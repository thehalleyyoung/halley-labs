"""
usability_oracle.taskspec.recorder — Record task steps from user events.

Converts raw browser / UI event logs into structured :class:`TaskSpec`
objects.  Supports:

* Manual step-by-step recording (``record_step``)
* Batch conversion from event logs (``from_event_log``)
* Keystroke merging (individual key events → ``type`` actions)
* Navigation filtering (removing noise events)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from usability_oracle.taskspec.models import TaskFlow, TaskSpec, TaskStep


# ---------------------------------------------------------------------------
# Recorded event representation
# ---------------------------------------------------------------------------

@dataclass
class RecordedEvent:
    """A single raw event captured during recording."""

    timestamp: float
    event_type: str              # "click", "keydown", "keyup", "input", "focus", "scroll", ...
    target_tag: str = ""         # HTML tag name
    target_id: str = ""          # DOM id attribute
    target_class: str = ""       # CSS class(es)
    target_role: str = ""        # ARIA role
    target_name: str = ""        # ARIA label / visible text
    target_selector: str = ""    # full CSS selector
    value: str = ""              # input value / key pressed
    position: Tuple[float, float] = (0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Role inference table
# ---------------------------------------------------------------------------

_TAG_TO_ROLE: Dict[str, str] = {
    "a": "link",
    "button": "button",
    "input": "textfield",
    "textarea": "textfield",
    "select": "combobox",
    "option": "option",
    "checkbox": "checkbox",
    "radio": "radio",
    "img": "image",
    "h1": "heading",
    "h2": "heading",
    "h3": "heading",
    "h4": "heading",
    "nav": "navigation",
    "dialog": "dialog",
    "table": "table",
    "tr": "row",
    "td": "cell",
    "li": "listitem",
    "ul": "list",
    "ol": "list",
}

_INPUT_TYPE_TO_ROLE: Dict[str, str] = {
    "text": "textfield",
    "password": "textfield",
    "email": "textfield",
    "number": "spinbutton",
    "checkbox": "checkbox",
    "radio": "radio",
    "submit": "button",
    "button": "button",
    "search": "searchbox",
    "url": "textfield",
    "tel": "textfield",
    "date": "textfield",
    "file": "button",
    "range": "slider",
}


# ---------------------------------------------------------------------------
# TaskRecorder
# ---------------------------------------------------------------------------


class TaskRecorder:
    """Accumulate user actions and produce a :class:`TaskSpec`.

    Usage — manual recording::

        rec = TaskRecorder(task_name="login")
        rec.record_step("click", {"role": "textfield", "name": "Username"})
        rec.record_step("type", {"role": "textfield", "name": "Username"}, value="alice")
        rec.record_step("click", {"role": "button", "name": "Sign In"})
        spec = rec.finish()

    Usage — from event log::

        events = [{"type": "click", "target_tag": "input", ...}, ...]
        spec = TaskRecorder.from_event_log(events, task_name="login")
    """

    def __init__(self, task_name: str = "recorded_task", task_description: str = "") -> None:
        self._task_name = task_name
        self._task_description = task_description
        self._steps: List[TaskStep] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._recording = True

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def step_count(self) -> int:
        return len(self._steps)

    # -- manual recording ----------------------------------------------------

    def record_step(
        self,
        action_type: str,
        target_info: Dict[str, str],
        *,
        value: Optional[str] = None,
        description: str = "",
        optional: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskStep:
        """Record a single step.

        Parameters
        ----------
        action_type : str
            One of the canonical action types.
        target_info : dict
            Must include ``role`` and/or ``name`` keys.
        value : str | None
            Input value for type/select actions.
        description : str
            Human-readable step description.

        Returns
        -------
        TaskStep
            The newly recorded step.

        Raises
        ------
        RuntimeError
            If recording has already been finalised.
        """
        if not self._recording:
            raise RuntimeError("Recording has already been finalised.")

        step = TaskStep(
            step_id=f"rec-{uuid.uuid4().hex[:8]}",
            action_type=action_type,
            target_role=target_info.get("role", ""),
            target_name=target_info.get("name", ""),
            target_selector=target_info.get("selector"),
            input_value=value,
            optional=optional,
            description=description or f"{action_type} on {target_info.get('name', '?')}",
            metadata=metadata or {},
        )
        self._steps.append(step)
        return step

    def undo_last(self) -> Optional[TaskStep]:
        """Remove and return the last recorded step, or None."""
        if self._steps:
            return self._steps.pop()
        return None

    def finish(self, *, success_criteria: Optional[List[str]] = None) -> TaskSpec:
        """Finalise the recording and return a :class:`TaskSpec`.

        After calling this, no more steps can be recorded.
        """
        self._recording = False
        flow = TaskFlow(
            flow_id=f"recorded-{uuid.uuid4().hex[:8]}",
            name=f"{self._task_name}_flow",
            steps=list(self._steps),
            success_criteria=success_criteria or [],
        )
        return TaskSpec(
            name=self._task_name,
            description=self._task_description,
            flows=[flow],
        )

    # -- event log conversion ------------------------------------------------

    @classmethod
    def from_event_log(
        cls,
        events: List[Dict[str, Any]],
        *,
        task_name: str = "recorded_task",
        merge_keystrokes: bool = True,
        filter_noise: bool = True,
    ) -> TaskSpec:
        """Convert a browser / UI event log into a :class:`TaskSpec`.

        Parameters
        ----------
        events : list[dict]
            Raw event dictionaries with keys like ``type``, ``target_tag``,
            ``target_id``, ``target_role``, ``target_name``, ``value``,
            ``timestamp``.
        merge_keystrokes : bool
            Merge consecutive ``keydown`` / ``input`` events into a single
            ``type`` action.
        filter_noise : bool
            Remove non-informative events (mousemove, focus, blur).

        Returns
        -------
        TaskSpec
        """
        parsed = [cls._dict_to_event(e) for e in events]

        if filter_noise:
            parsed = cls._filter_navigation(parsed)

        if merge_keystrokes:
            parsed = cls._merge_typing_events(parsed)

        recorder = cls(task_name=task_name)
        for ev in parsed:
            action = cls._event_type_to_action(ev.event_type)
            if action is None:
                continue
            role, name = cls._infer_target(ev)
            step_value = ev.value if action in ("type", "select") else None
            recorder.record_step(
                action,
                {"role": role, "name": name, "selector": ev.target_selector},
                value=step_value,
            )
        return recorder.finish()

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _dict_to_event(data: Dict[str, Any]) -> RecordedEvent:
        return RecordedEvent(
            timestamp=float(data.get("timestamp", 0)),
            event_type=data.get("type", data.get("event_type", "")),
            target_tag=data.get("target_tag", ""),
            target_id=data.get("target_id", ""),
            target_class=data.get("target_class", ""),
            target_role=data.get("target_role", ""),
            target_name=data.get("target_name", ""),
            target_selector=data.get("target_selector", ""),
            value=data.get("value", ""),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def _infer_target(event: RecordedEvent) -> Tuple[str, str]:
        """Infer (role, name) from a :class:`RecordedEvent`.

        Priority:
        1. Explicit ARIA role/name from the event
        2. HTML tag → role mapping + id/class heuristics
        """
        # Use explicit role if present
        role = event.target_role
        if not role:
            tag = event.target_tag.lower()
            input_type = event.metadata.get("input_type", "text")
            if tag == "input":
                role = _INPUT_TYPE_TO_ROLE.get(input_type, "textfield")
            else:
                role = _TAG_TO_ROLE.get(tag, "generic")

        # Use explicit name if present
        name = event.target_name
        if not name:
            # Fallback: use id, then class, then tag
            if event.target_id:
                # Convert camelCase / snake_case IDs to human-readable
                name = re.sub(r"([a-z])([A-Z])", r"\1 \2", event.target_id)
                name = name.replace("_", " ").replace("-", " ").title()
            elif event.target_class:
                first_class = event.target_class.split()[0]
                name = first_class.replace("-", " ").replace("_", " ").title()
            else:
                name = event.target_tag or "unknown"

        return role, name

    @staticmethod
    def _merge_typing_events(events: List[RecordedEvent]) -> List[RecordedEvent]:
        """Merge consecutive keystroke events into single ``input`` events.

        Consecutive ``keydown`` / ``keypress`` / ``input`` events targeting
        the same element are collapsed into a single event whose ``value``
        is the concatenated text.
        """
        merged: List[RecordedEvent] = []
        i = 0
        while i < len(events):
            ev = events[i]
            if ev.event_type in ("keydown", "keypress", "keyup", "input"):
                # Start accumulating
                target_sel = ev.target_selector or ev.target_id
                chars: List[str] = []
                last_ev = ev
                while (
                    i < len(events)
                    and events[i].event_type in ("keydown", "keypress", "keyup", "input")
                    and (events[i].target_selector or events[i].target_id) == target_sel
                ):
                    cur = events[i]
                    # Only accumulate from 'input' or single-char keydown
                    if cur.event_type == "input" and cur.value:
                        chars.append(cur.value)
                    elif cur.event_type == "keydown" and len(cur.value) == 1:
                        chars.append(cur.value)
                    last_ev = cur
                    i += 1

                if chars:
                    merged_event = RecordedEvent(
                        timestamp=ev.timestamp,
                        event_type="input",
                        target_tag=ev.target_tag,
                        target_id=ev.target_id,
                        target_class=ev.target_class,
                        target_role=ev.target_role,
                        target_name=ev.target_name,
                        target_selector=ev.target_selector,
                        value="".join(chars),
                        metadata=ev.metadata,
                    )
                    merged.append(merged_event)
                else:
                    merged.append(ev)
                    i += 1
            else:
                merged.append(ev)
                i += 1

        return merged

    @staticmethod
    def _filter_navigation(events: List[RecordedEvent]) -> List[RecordedEvent]:
        """Remove non-informative events that add noise.

        Filtered event types: ``mousemove``, ``mouseenter``, ``mouseleave``,
        ``focus``, ``blur``, ``resize``, ``pointerover``, ``pointerout``.
        """
        noise_types = frozenset({
            "mousemove", "mouseenter", "mouseleave", "mouseover", "mouseout",
            "focus", "blur", "focusin", "focusout",
            "resize", "orientationchange",
            "pointerover", "pointerout", "pointerenter", "pointerleave",
            "touchmove",
        })
        return [e for e in events if e.event_type not in noise_types]

    @staticmethod
    def _event_type_to_action(event_type: str) -> Optional[str]:
        """Map a raw DOM event type to a canonical action type."""
        mapping = {
            "click": "click",
            "mousedown": "click",
            "pointerdown": "click",
            "dblclick": "double_click",
            "contextmenu": "right_click",
            "input": "type",
            "change": "select",
            "scroll": "scroll",
            "wheel": "scroll",
            "keydown": "key_press",
            "submit": "click",
            "dragend": "drag",
            "drop": "drag",
        }
        return mapping.get(event_type)

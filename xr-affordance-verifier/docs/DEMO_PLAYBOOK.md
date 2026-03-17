# XR Affordance Verifier Demo Playbook

## Goal

Use the generated web dashboard as both the live product demo and the slide deck.

Primary artifact:
- [scene.dashboard.html](../scene.dashboard.html) or another dashboard generated with `xr-verify webapp`

## Best demo flow

### 1. Generate the showpiece scene

Use one of the stronger built-in scenes:

- `control-room` for breadth and visual variety
- `manufacturing` for workflow and dependency storytelling
- `accessibility` for showing lint failures and remediation opportunities

Suggested command:

`xr-verify demo control-room -o control_room.json`

### 2. Generate the dashboard

Suggested command:

`xr-verify webapp control_room.json -n 2000 --confidence 0.99 -o control_room.dashboard.html`

This produces:
- an interactive dashboard
- an on-the-fly certificate JSON beside it
- a built-in presentation narrative

### 3. Open the dashboard

Open the generated HTML file in a browser.

## Talk track

### Slide 1 — Motivation

Key message:
XR accessibility is not just UI polish. It is about whether people can physically reach, perceive, and complete spatial interactions.

What to point at:
- scene name and summary in the hero panel
- total affordances
- interaction-mode diversity
- why manual spot checks are insufficient

### Slide 2 — Method

Key message:
The tool combines fast linting with deeper certification so the audience sees both speed and rigor.

What to point at:
- certificate grade
- κ coverage
- uncertainty metrics
- verification insights panel

### Slide 3 — Scene story

Key message:
A single high-priority affordance can be inspected spatially, behaviorally, and in context.

What to point at:
- top/front/side projection buttons
- selected element panel
- dependency graph
- prioritized coverage list

### Slide 4 — Close

Key message:
This is both a flashy demo and a practical engineering artifact.

What to point at:
- demo readiness text
- violation callouts
- how the same dashboard can be handed to designers, PMs, or auditors

## Live interactions to perform

### Recommended sequence

1. Advance through the built-in presentation slides.
2. On the scene-story slide, use the “Focus highlighted affordance” action.
3. Switch between top/front/side projections.
4. Click the hero affordance in the scene explorer.
5. Show the dependency graph and selected-element notes.
6. Scroll to violations if you want an honest “what still needs work” finish.

## Keyboard shortcuts

- `←` / `→` — previous / next slide
- `PageUp` / `PageDown` — previous / next slide
- `n` — show or hide speaker notes

## Presenter tips

- Prefer `control-room` if you want the most visual and varied walkthrough.
- Prefer `manufacturing` if you want a stronger sequence/dependency story.
- Prefer `accessibility` if you want to show failure detection and remediation.
- Keep one browser tab on the dashboard and one terminal tab ready with the generation commands.

## Backup plan

If certification takes longer than expected, generate the scene first and keep a prebuilt dashboard file ready.

Then use:

`xr-verify webapp control_room.json --certificate control_room.dashboard.certificate.json -o control_room.dashboard.html`

## What success looks like

By the end of the demo, the audience should understand:
- why XR accessibility needs spatial verification
- how the verifier turns scenes into evidence
- which affordances are strongest or weakest
- that the HTML dashboard is portable and presentation-ready

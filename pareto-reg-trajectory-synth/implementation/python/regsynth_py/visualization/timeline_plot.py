"""Timeline and Gantt chart visualization using pure SVG generation.

Produces compliance roadmap timelines, milestone charts, and deadline calendars
with no external dependencies.
"""

from __future__ import annotations

import math
from typing import Optional

from regsynth_py.visualization.svg_utils import (
    PALETTE,
    RISK_COLORS,
    SVGElement,
    color_interpolate,
    render_svg,
    scale_linear,
    svg_circle,
    svg_document,
    svg_group,
    svg_line,
    svg_polygon,
    svg_rect,
    svg_text,
    svg_title,
)


class TimelinePlotter:
    """Generate SVG timeline / Gantt visualizations."""

    def __init__(self, width: int = 1000, height: int = 500, margin: int = 60) -> None:
        self.width = width
        self.height = height
        self.margin = margin

    # ------------------------------------------------------------------
    # Gantt chart
    # ------------------------------------------------------------------

    def plot_gantt(
        self,
        tasks: list[dict],
        title: str = "Compliance Roadmap",
    ) -> str:
        """Render a Gantt chart.

        Each *task* dict: ``{name, start, end, category, progress, color}``.
        ``start``/``end`` are ISO-like date strings (``YYYY-MM-DD``).
        ``progress`` is 0.0 – 1.0.
        """
        if not tasks:
            return render_svg(svg_document(self.width, self.height))

        all_dates = []
        for t in tasks:
            all_dates.append(self._parse_date(t["start"]))
            all_dates.append(self._parse_date(t["end"]))
        date_min = min(all_dates)
        date_max = max(all_dates)

        categories: list[str] = []
        for t in tasks:
            cat = t.get("category", "")
            if cat not in categories:
                categories.append(cat)
        cat_index = {c: i for i, c in enumerate(categories)}

        row_height = 32
        needed_h = self.margin * 2 + 50 + len(tasks) * row_height
        svg_h = max(self.height, needed_h)

        svg = svg_document(self.width, svg_h)
        svg.add_child(svg_rect(0, 0, self.width, svg_h, "#ffffff"))

        m = self.margin
        label_w = 160
        pw = self.width - m - label_w - m
        top = m + 40

        total_days = max(self._date_diff_days(date_min, date_max), 1)

        def date_to_x(ds: str) -> float:
            d = self._parse_date(ds)
            diff = self._date_diff_days(date_min, d)
            return label_w + m + (diff / total_days) * pw

        # Month gridlines
        months = self._month_ticks(date_min, date_max)
        for md in months:
            mx = label_w + m + (self._date_diff_days(date_min, md) / total_days) * pw
            svg.add_child(svg_line(mx, top, mx, top + len(tasks) * row_height, "#eee"))
            svg.add_child(svg_text(mx, top - 5, self._format_date_tuple(md), font_size=9, fill="#999", anchor="middle"))

        # Tasks
        for idx, t in enumerate(tasks):
            y = top + idx * row_height
            x1 = date_to_x(t["start"])
            x2 = date_to_x(t["end"])
            bar_w = max(x2 - x1, 4)
            color = t.get("color") or PALETTE[cat_index.get(t.get("category", ""), 0) % len(PALETTE)]
            progress = float(t.get("progress", 0))

            # Full bar background
            svg.add_child(svg_rect(x1, y + 4, bar_w, row_height - 8, color, rx=4, opacity=0.3))
            # Progress fill
            if progress > 0:
                svg.add_child(svg_rect(x1, y + 4, bar_w * min(progress, 1.0), row_height - 8, color, rx=4, opacity=0.85))

            # Task label
            svg.add_child(svg_text(m + 5, y + row_height / 2 + 4, t["name"], font_size=11, fill="#333"))

            # Percentage label inside bar
            pct = f"{int(progress * 100)}%"
            svg.add_child(svg_text(x1 + bar_w / 2, y + row_height / 2 + 4, pct, font_size=9, fill="#fff", anchor="middle", font_weight="bold"))

            bar_el = svg_rect(x1, y + 4, bar_w, row_height - 8, "transparent", rx=4)
            bar_el.add_child(svg_title(f"{t['name']}: {t['start']} → {t['end']} ({pct})"))
            svg.add_child(bar_el)

        # Today line
        today = self._today_tuple()
        if date_min <= today <= date_max:
            tx = label_w + m + (self._date_diff_days(date_min, today) / total_days) * pw
            svg.add_child(svg_line(tx, top - 10, tx, top + len(tasks) * row_height + 10, "#e15759", stroke_width=2, dash="4,3"))
            svg.add_child(svg_text(tx, top - 14, "Today", font_size=10, fill="#e15759", anchor="middle"))

        svg.add_child(svg_text(self.width / 2, 25, title, font_size=16, fill="#333", anchor="middle", font_weight="bold"))
        return render_svg(svg)

    # ------------------------------------------------------------------
    # Milestone timeline
    # ------------------------------------------------------------------

    def plot_milestone_timeline(
        self,
        milestones: list[dict],
        title: str = "Regulatory Timeline",
    ) -> str:
        """Render a milestone timeline.

        Each milestone: ``{date, label, category, importance}``.
        ``importance`` is 1–5 (maps to circle size).
        """
        if not milestones:
            return render_svg(svg_document(self.width, self.height))

        dates = [self._parse_date(ms["date"]) for ms in milestones]
        date_min, date_max = min(dates), max(dates)
        total_days = max(self._date_diff_days(date_min, date_max), 1)

        svg = svg_document(self.width, self.height)
        svg.add_child(svg_rect(0, 0, self.width, self.height, "#ffffff"))

        m = self.margin
        pw = self.width - 2 * m
        mid_y = self.height / 2

        # Axis
        svg.add_child(svg_line(m, mid_y, m + pw, mid_y, "#bbb", stroke_width=2))

        # Month labels
        months = self._month_ticks(date_min, date_max)
        for md in months:
            mx = m + (self._date_diff_days(date_min, md) / total_days) * pw
            svg.add_child(svg_line(mx, mid_y - 5, mx, mid_y + 5, "#aaa"))
            svg.add_child(svg_text(mx, mid_y + 22, self._format_date_tuple(md), font_size=9, fill="#999", anchor="middle"))

        # Category colors
        cats: dict[str, str] = {}
        for ms in milestones:
            cat = ms.get("category", "default")
            if cat not in cats:
                cats[cat] = PALETTE[len(cats) % len(PALETTE)]

        # Milestones (alternate above/below)
        for idx, ms in enumerate(milestones):
            d = self._parse_date(ms["date"])
            mx = m + (self._date_diff_days(date_min, d) / total_days) * pw
            above = idx % 2 == 0
            imp = int(ms.get("importance", 3))
            r = 4 + imp * 2
            color = cats.get(ms.get("category", "default"), PALETTE[0])

            if above:
                cy = mid_y - 35 - (idx % 3) * 22
                svg.add_child(svg_line(mx, mid_y, mx, cy + r, "#ddd", dash="2,2"))
            else:
                cy = mid_y + 35 + (idx % 3) * 22
                svg.add_child(svg_line(mx, mid_y, mx, cy - r, "#ddd", dash="2,2"))

            circ = svg_circle(mx, cy, r, color, stroke="#fff", opacity=0.9)
            circ.add_child(svg_title(f"{ms['label']} ({ms['date']})"))
            svg.add_child(circ)

            label_y = cy - r - 6 if above else cy + r + 14
            svg.add_child(svg_text(mx, label_y, ms["label"], font_size=10, fill="#444", anchor="middle"))

        svg.add_child(svg_text(self.width / 2, 25, title, font_size=16, fill="#333", anchor="middle", font_weight="bold"))
        return render_svg(svg)

    # ------------------------------------------------------------------
    # Deadline calendar
    # ------------------------------------------------------------------

    def plot_deadline_calendar(
        self,
        deadlines: list[dict],
        title: str = "Deadline Calendar",
    ) -> str:
        """Render a monthly calendar grid with deadline markers.

        Each deadline: ``{date, label, category}``.
        """
        if not deadlines:
            return render_svg(svg_document(self.width, self.height))

        dates = [self._parse_date(dl["date"]) for dl in deadlines]
        today = self._today_tuple()

        # Determine month range
        all_months = set()
        for y, mo, _ in dates:
            all_months.add((y, mo))
        sorted_months = sorted(all_months)
        if not sorted_months:
            sorted_months = [(today[0], today[1])]

        cols = min(len(sorted_months), 4)
        rows_grid = math.ceil(len(sorted_months) / cols)
        cell = 22
        month_w = cell * 7 + 20
        month_h = cell * 7 + 40
        total_w = max(self.width, cols * month_w + 40)
        total_h = max(self.height, rows_grid * month_h + 80)

        svg = svg_document(total_w, total_h)
        svg.add_child(svg_rect(0, 0, total_w, total_h, "#ffffff"))

        deadline_map: dict[tuple[int, int, int], list[dict]] = {}
        for dl in deadlines:
            d = self._parse_date(dl["date"])
            deadline_map.setdefault(d, []).append(dl)

        for mi, (yr, mo) in enumerate(sorted_months):
            col = mi % cols
            row = mi // cols
            ox = 20 + col * month_w
            oy = 60 + row * month_h

            svg.add_child(svg_text(ox + month_w / 2 - 10, oy, f"{yr}-{mo:02d}", font_size=13, fill="#333", font_weight="bold"))

            days_in = self._days_in_month(yr, mo)
            first_dow = self._day_of_week(yr, mo, 1)

            for day_label_i, dl in enumerate(["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]):
                svg.add_child(svg_text(ox + day_label_i * cell + cell / 2, oy + 20, dl, font_size=9, fill="#999", anchor="middle"))

            for day in range(1, days_in + 1):
                dow = (first_dow + day - 1) % 7
                week = (first_dow + day - 1) // 7
                dx = ox + dow * cell
                dy = oy + 28 + week * cell
                date_tuple = (yr, mo, day)

                fill = "#fafafa"
                text_color = "#333"
                if date_tuple in deadline_map:
                    diff = self._date_diff_days(today, date_tuple)
                    if diff < 0:
                        fill = RISK_COLORS["critical"]
                        text_color = "#fff"
                    elif diff <= 7:
                        fill = RISK_COLORS["high"]
                        text_color = "#fff"
                    elif diff <= 30:
                        fill = RISK_COLORS["medium"]
                    else:
                        fill = RISK_COLORS["low"]
                        text_color = "#fff"

                r = svg_rect(dx, dy, cell - 2, cell - 2, fill, stroke="#eee", rx=3)
                if date_tuple in deadline_map:
                    tip_lines = "; ".join(d["label"] for d in deadline_map[date_tuple])
                    r.add_child(svg_title(tip_lines))
                svg.add_child(r)
                svg.add_child(svg_text(dx + cell / 2 - 1, dy + cell / 2 + 3, str(day), font_size=9, fill=text_color, anchor="middle"))

        svg.add_child(svg_text(total_w / 2, 30, title, font_size=16, fill="#333", anchor="middle", font_weight="bold"))
        return render_svg(svg)

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(date_str: str) -> tuple[int, int, int]:
        parts = date_str.strip().split("-")
        return int(parts[0]), int(parts[1]), int(parts[2])

    @staticmethod
    def _date_diff_days(d1: tuple[int, int, int], d2: tuple[int, int, int]) -> int:
        def to_jdn(y: int, m: int, d: int) -> int:
            a = (14 - m) // 12
            y2 = y + 4800 - a
            m2 = m + 12 * a - 3
            return d + (153 * m2 + 2) // 5 + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045
        return to_jdn(*d2) - to_jdn(*d1)

    @staticmethod
    def _format_date(date_str: str, fmt: str = "short") -> str:
        parts = date_str.strip().split("-")
        if fmt == "short":
            return f"{parts[1]}/{parts[2]}"
        return date_str

    @staticmethod
    def _format_date_tuple(d: tuple[int, int, int]) -> str:
        return f"{d[0]}-{d[1]:02d}"

    @staticmethod
    def _days_in_month(year: int, month: int) -> int:
        if month in (1, 3, 5, 7, 8, 10, 12):
            return 31
        if month in (4, 6, 9, 11):
            return 30
        if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
            return 29
        return 28

    @staticmethod
    def _day_of_week(year: int, month: int, day: int) -> int:
        """Zeller-like formula.  Returns 0=Mon .. 6=Sun."""
        if month < 3:
            month += 12
            year -= 1
        k = year % 100
        j = year // 100
        h = (day + (13 * (month + 1)) // 5 + k + k // 4 + j // 4 - 2 * j) % 7
        return (h + 5) % 7  # convert Zeller (0=Sat) to 0=Mon

    def _date_to_x(self, date_str: str, date_range: tuple[tuple, tuple]) -> float:
        d = self._parse_date(date_str)
        total = max(self._date_diff_days(date_range[0], date_range[1]), 1)
        frac = self._date_diff_days(date_range[0], d) / total
        return self.margin + frac * (self.width - 2 * self.margin)

    @staticmethod
    def _today_tuple() -> tuple[int, int, int]:
        import time
        t = time.localtime()
        return (t.tm_year, t.tm_mon, t.tm_mday)

    def _month_ticks(
        self, d_min: tuple[int, int, int], d_max: tuple[int, int, int],
    ) -> list[tuple[int, int, int]]:
        ticks: list[tuple[int, int, int]] = []
        y, m = d_min[0], d_min[1]
        while (y, m) <= (d_max[0], d_max[1]):
            ticks.append((y, m, 1))
            m += 1
            if m > 12:
                m = 1
                y += 1
        return ticks

    def save(self, svg_content: str, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(svg_content)

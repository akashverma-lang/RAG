"""Choosing how to draw a query result, and keeping that drawable.

The chart type is decided from the *shape* of the result plus the column roles the
importer already profiled -- not by asking the model.  Shape is something we can
check, so the choice is predictable and free, and a bar chart never appears with a
hallucinated axis.

Two invariants make the picker safe to override:

* the *natural* kind is always computed from the data, never from the user's choice,
  so the list of alternatives never collapses -- picking "table" used to be a
  one-way door out of every other option;
* every chart carries the raw rows as well as its series, so switching to "table"
  always has something to show and CSV export never needs a second query.

Every guard here exists because of a way a real result breaks a chart: identifiers
with 29,206 distinct values, categories in the thousands, dates that are NULL,
series that would need one colour per row.  A result that cannot be drawn honestly
falls back to a table rather than being forced into a shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from app import config

_ISO_DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?([ T]\d{2}:\d{2}(:\d{2})?)?$")
BLANK = "(blank)"
TABLE_DISPLAY_ROWS = 100

KINDS = ("stat", "stats", "bar", "hbar", "sbar", "line", "area", "donut",
         "scatter", "table")


@dataclass
class Chart:
    kind: str = "table"          # what to draw now (may be a user override)
    natural: str = "table"       # what the data shape suggests, always
    x: str = ""
    y: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    series: list[dict] = field(default_factory=list)   # [{name, data:[...]}]
    points: list[list[float]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    total: float = 0.0

    def as_dict(self) -> dict:
        return {
            "kind": self.kind, "natural": self.natural, "x": self.x, "y": self.y,
            "labels": self.labels, "series": self.series, "points": self.points,
            "columns": self.columns, "rows": self.rows, "notes": self.notes,
            "total": self.total,
        }


# --------------------------------------------------------------------------- typing
def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_date(v: Any) -> bool:
    return isinstance(v, str) and bool(_ISO_DATE.match(v.strip()))


def _column_kind(values: Sequence[Any]) -> str:
    filled = [v for v in values if v is not None and v != ""]
    if not filled:
        return "category"
    if all(_is_number(v) for v in filled):
        return "number"
    if all(_is_date(v) for v in filled):
        return "date"
    return "category"


def _label(v: Any) -> str:
    return BLANK if v is None or v == "" else str(v)


# --------------------------------------------------------------------------- guards
def _cap_categories(labels: list[str], series: list[dict],
                    notes: list[str]) -> tuple[list[str], list[dict]]:
    """Keep the largest N categories and say so.

    Deliberately no "Others" bucket: summing the tail is only correct for additive
    measures, and nothing in a bare result says whether the column is a count or an
    average.  Quietly averaging averages would be a wrong number on a chart, which
    is worse than a shorter chart with a note.
    """
    limit = config.CHART_MAX_CATEGORIES
    if len(labels) <= limit:
        return labels, series
    primary = series[0]["data"] if series else []
    order = sorted(range(len(labels)),
                   key=lambda i: abs(primary[i]) if i < len(primary)
                   and primary[i] is not None else 0.0, reverse=True)
    keep = sorted(order[:limit])
    notes.append(f"showing the {limit} largest of {len(labels)} categories")
    return ([labels[i] for i in keep],
            [{"name": s["name"], "data": [s["data"][i] for i in keep]} for s in series])


def _downsample(labels: list[str], series: list[dict],
                notes: list[str]) -> tuple[list[str], list[dict]]:
    limit = config.CHART_MAX_POINTS
    if len(labels) <= limit:
        return labels, series
    step = len(labels) / limit
    idx = sorted({min(len(labels) - 1, int(i * step)) for i in range(limit)})
    idx = sorted(set(idx) | {len(labels) - 1})
    notes.append(f"{len(labels)} points thinned to {len(idx)} for display")
    return ([labels[i] for i in idx],
            [{"name": s["name"], "data": [s["data"][i] for i in idx]} for s in series])


# --------------------------------------------------------------------------- build
def _natural(columns: list[str], rows: list[list[Any]],
             roles: dict[str, str]) -> Chart:
    """The chart the data itself suggests, with no user preference involved."""
    notes: list[str] = []
    if not columns or not rows:
        return Chart(notes=["the query returned no rows"])

    cols = list(zip(*rows))
    kinds = [_column_kind(c) for c in cols]
    nums = [i for i, k in enumerate(kinds) if k == "number"]
    dates = [i for i, k in enumerate(kinds) if k == "date"]
    # An identifier axis is never a chart: 29,206 serial numbers is not a bar chart,
    # it is a table that happens to be tall.
    cats = [i for i, k in enumerate(kinds)
            if k == "category" and roles.get(columns[i], "") != "id"]

    if not nums:
        return Chart(notes=["nothing numeric to plot"])

    if len(rows) == 1:
        if len(nums) == 1 and len(columns) == 1:
            return Chart(kind="stat", y=[columns[nums[0]]],
                         series=[{"name": columns[nums[0]], "data": [rows[0][nums[0]]]}])
        if len(nums) == len(columns):
            return Chart(kind="stats", y=[columns[i] for i in nums],
                         series=[{"name": columns[i], "data": [rows[0][i]]} for i in nums])

    if dates and nums:
        xi = dates[0]
        pairs = [(r[xi], r) for r in rows if r[xi]]
        dropped = len(rows) - len(pairs)
        if dropped:
            notes.append(f"{dropped} row(s) with no {columns[xi]} left out")
        if not pairs:
            return Chart(notes=notes + ["every date was empty"])
        pairs.sort(key=lambda p: str(p[0]))
        labels = [str(p[0]) for p in pairs]
        series = [{"name": columns[i], "data": [pr[1][i] for pr in pairs]} for i in nums]
        labels, series = _downsample(labels, series, notes)
        return Chart(kind="line", x=columns[xi], y=[columns[i] for i in nums],
                     labels=labels, series=series, notes=notes)

    if len(cats) >= 2 and nums:
        xi, si, vi = cats[0], cats[1], nums[0]
        x_vals, s_vals = [], []
        for r in rows:
            if _label(r[xi]) not in x_vals:
                x_vals.append(_label(r[xi]))
            if _label(r[si]) not in s_vals:
                s_vals.append(_label(r[si]))
        if len(s_vals) <= 8 and len(x_vals) * len(s_vals) <= 400:
            grid = {(_label(r[xi]), _label(r[si])): r[vi] for r in rows}
            series = [{"name": s, "data": [grid.get((x, s), 0) for x in x_vals]}
                      for s in s_vals]
            x_vals, series = _cap_categories(x_vals, series, notes)
            return Chart(kind="bar", x=columns[xi], y=[columns[vi]],
                         labels=x_vals, series=series, notes=notes)
        notes.append(f"{len(s_vals)} series is too many to colour; showing the table")
        return Chart(notes=notes)

    if cats and nums:
        xi = cats[0]
        labels = [_label(r[xi]) for r in rows]
        series = [{"name": columns[i], "data": [r[i] for r in rows]} for i in nums]
        if len(series) == 1:                       # biggest first reads better
            order = sorted(range(len(labels)),
                           key=lambda i: series[0]["data"][i]
                           if series[0]["data"][i] is not None else 0, reverse=True)
            labels = [labels[i] for i in order]
            series = [{"name": s["name"], "data": [s["data"][i] for i in order]}
                      for s in series]
        labels, series = _cap_categories(labels, series, notes)
        longest = max((len(x) for x in labels), default=0)
        kind = "hbar" if longest > 16 or len(labels) > 8 else "bar"
        total = sum(v for v in series[0]["data"] if _is_number(v))
        return Chart(kind=kind, x=columns[xi], y=[columns[i] for i in nums],
                     labels=labels, series=series, notes=notes, total=float(total))

    if len(nums) >= 2:
        xi, yi = nums[0], nums[1]
        pts = [[r[xi], r[yi]] for r in rows
               if _is_number(r[xi]) and _is_number(r[yi])][:config.CHART_MAX_POINTS]
        if pts:
            return Chart(kind="scatter", x=columns[xi], y=[columns[yi]],
                         points=pts, notes=notes)

    return Chart(notes=notes)


def infer(columns: Sequence[str], rows: Sequence[Sequence[Any]],
          roles: dict[str, str] | None = None, prefer: str = "") -> Chart:
    """Pick a chart for this result and prepare the data a renderer needs.

    ``roles`` maps column name -> role from the table catalogue; it is what stops an
    identifier being charted.  ``prefer`` overrides the drawn kind, but never the
    natural one, so the alternatives on offer stay the same.
    """
    columns = list(columns)
    rows = [list(r) for r in rows]
    chart = _natural(columns, rows, roles or {})
    chart.natural = chart.kind

    # Raw rows travel with every chart, so "table" is always renderable and CSV
    # export works from whatever is on screen.
    chart.columns = columns
    chart.rows = rows[:TABLE_DISPLAY_ROWS]
    if len(rows) > TABLE_DISPLAY_ROWS:
        chart.notes.append(f"table view shows {TABLE_DISPLAY_ROWS} of {len(rows)} rows")

    if prefer and prefer != chart.kind and prefer in options_for(chart):
        chart.kind = prefer
    return chart


def options_for(chart: Chart) -> list[str]:
    """Chart kinds it is honest to offer for this result.

    Derived from the natural kind, never the current one -- otherwise choosing
    "table" removes every other option and there is no way back.
    """
    base = chart.natural or chart.kind
    if base in {"stat", "stats"}:
        return [base, "table"]
    if base in {"line", "area"}:
        return ["line", "area", "bar", "table"]
    if base == "scatter":
        return ["scatter", "line", "table"]
    if base in {"bar", "hbar", "sbar", "donut"}:
        multi = len(chart.series) > 1
        opts = ["bar", "hbar"]
        if multi:
            opts.append("sbar")               # stacking only means something for parts
        else:
            # A donut is only honest for a handful of non-negative parts of a whole.
            data = chart.series[0]["data"] if chart.series else []
            if 2 <= len(chart.labels) <= 8 and all((v or 0) >= 0 for v in data):
                opts.append("donut")
        return opts + ["table"]
    return ["table"]


# --------------------------------------------------------------------------- titles
_STOP = {"select", "from", "where", "group", "order", "by", "limit", "desc", "asc",
         "as", "and", "or", "count", "sum", "avg", "min", "max", "round", "distinct"}


def humanize(name: str) -> str:
    text = re.sub(r"[_\s]+", " ", str(name or "").strip())
    return text[:1].upper() + text[1:] if text else ""


def suggest_title(chart: Chart, question: str = "") -> str:
    """A readable panel name from the result itself, so nobody has to invent one.

    Built from the columns rather than the SQL text: the aliases the model chose are
    already the best short description of what was measured.
    """
    measure = humanize(chart.y[0]) if chart.y else ""
    dimension = humanize(chart.x) if chart.x else ""
    if chart.natural in {"stat", "stats"} and chart.series:
        return humanize(chart.series[0]["name"]) or "Value"
    if measure and dimension:
        return f"{measure} by {dimension}"
    if measure:
        return measure
    if chart.columns:
        return " / ".join(humanize(c) for c in chart.columns[:2])
    q = (question or "").strip().rstrip("?")
    return (q[:1].upper() + q[1:])[:80] if q else "Panel"

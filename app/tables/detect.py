"""Deciding whether a sheet is a real table, and what each column holds.

A spreadsheet only earns a SQL table when it is genuinely tabular: one header row,
a stable column count, mostly-filled rows.  Pivot layouts, title banners and
several tables stacked on one sheet fail the check and keep the old text-chunking
path, which does not care about structure.

Type inference is deliberately conservative.  The expensive mistake is turning an
identifier into a number: an ENGINE SERIAL NUMBER of 0701290 would become 701290
and never match a lookup again, so anything with a leading zero stays TEXT.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from app import config

# Column names that are labels, not quantities -- summing them is always a bug.
# Only genuinely identifier-ish words belong here: entity words like "claim" or
# "invoice" also head measures ("CLAIM AMOUNT"), so they would misclassify money.
ID_HINT = re.compile(r"\b(id|ids|no|nos|num|number|code|serial|ref|uid|guid|pin|zip)\b",
                     re.IGNORECASE)
# A measure always wins over an identifier: "GENERATED CLAIM AMOUNT" must stay REAL
# or SUM() over it silently turns into string concatenation.
MEASURE_HINT = re.compile(
    r"\b(amount|amt|value|price|cost|total|sum|qty|quantity|mandays|days|hours|hrs|"
    r"reading|rate|discount|tax|weight|volume|count|score|percent|pct)\b", re.IGNORECASE)
_NUM_RE = re.compile(r"^[-+]?\d{1,3}(,\d{3})*(\.\d+)?$|^[-+]?\d*\.?\d+$")
_IDENT_BAD = re.compile(r"[^0-9a-zA-Z_]+")

_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d-%b-%Y", "%Y/%m/%d",
)


@dataclass
class Column:
    ord: int
    label: str            # header text exactly as it appears in the sheet
    name: str             # sanitised SQL identifier
    type: str = "TEXT"    # TEXT | INTEGER | REAL
    role: str = "text"    # id | date | number | category | text
    n_filled: int = 0


@dataclass
class SheetProbe:
    sheet: str
    header_row: int                     # 1-based row where the header sits
    columns: list[Column]
    sampled: int = 0
    ok: bool = True
    reason: str = ""
    fingerprint: str = ""

    @property
    def labels(self) -> list[str]:
        return [c.label for c in self.columns]


@dataclass
class FileProbe:
    path: Path
    sheets: list[SheetProbe] = field(default_factory=list)
    rejected: list[SheetProbe] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.sheets)


# --------------------------------------------------------------------------- naming
def sanitize(label: str, taken: set[str], fallback: str = "col") -> str:
    """Turn a header label into a SQL identifier that is still readable."""
    name = _IDENT_BAD.sub("_", (label or "").strip()).strip("_").lower()
    if not name or name[0].isdigit():
        name = f"{fallback}_{name}" if name else fallback
    base, n = name, 2
    while name in taken or name in {"rowid", "oid", "_rowid_"}:
        name = f"{base}_{n}"
        n += 1
    taken.add(name)
    return name


def table_name_for(path: Path, sheet: str, single_sheet: bool) -> str:
    stem = sanitize(path.stem, set(), "sheet")
    if single_sheet:
        return stem[:56]
    return f"{stem[:40]}_{sanitize(sheet, set(), 's')}"[:56]


# --------------------------------------------------------------------------- values
def to_iso(v: Any) -> str:
    if isinstance(v, dt.datetime):
        if v.hour or v.minute or v.second:
            return v.strftime("%Y-%m-%d %H:%M:%S")
        return v.strftime("%Y-%m-%d")
    if isinstance(v, dt.date):
        return v.strftime("%Y-%m-%d")
    return str(v).strip()


def as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dt.datetime, dt.date)):
        return to_iso(v)
    return str(v).strip()


def parse_date(text: str) -> str | None:
    t = text.strip()
    if not 8 <= len(t) <= 26:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return to_iso(dt.datetime.strptime(t, fmt))
        except ValueError:
            continue
    return None


def parse_number(text: str) -> float | None:
    t = text.strip()
    if not t or not _NUM_RE.match(t):
        return None
    try:
        return float(t.replace(",", ""))
    except ValueError:
        return None


def has_leading_zero(text: str) -> bool:
    t = text.strip().lstrip("+-")
    return len(t) > 1 and t[0] == "0" and t[1] != "."


# --------------------------------------------------------------------------- inference
def infer_column(col: Column, values: Sequence[Any]) -> Column:
    """Assign a storage type and a semantic role from sampled values."""
    texts = [as_text(v) for v in values]
    filled = [t for t in texts if t]
    col.n_filled = len(filled)
    if not filled:
        col.type, col.role = "TEXT", "text"
        return col

    native = sum(1 for v in values if isinstance(v, (dt.datetime, dt.date)))
    dates = native + sum(1 for t in filled if parse_date(t))
    if dates >= len(filled) * 0.9:
        col.type, col.role = "TEXT", "date"      # ISO strings: SQLite date funcs work
        return col

    measure = MEASURE_HINT.search(col.label) is not None
    if any(has_leading_zero(t) for t in filled) or (ID_HINT.search(col.label) and not measure):
        # Identifiers stay text even when they look numeric -- they exist for matching,
        # never arithmetic, and leading zeros have to survive the round trip.
        col.type, col.role = "TEXT", "id"
        return col

    parsed = [parse_number(t) for t in filled]
    if all(p is not None for p in parsed):
        col.type = "INTEGER" if all(float(p).is_integer() for p in parsed) else "REAL"
        col.role = "number"
        return col

    distinct = len(set(filled))
    avg_len = sum(len(t) for t in filled) / len(filled)
    col.type = "TEXT"
    if distinct <= max(config.TABLE_ENUM_MAX, 2) and avg_len <= 60:
        col.role = "category"
    else:
        col.role = "text"
    return col


# --------------------------------------------------------------------------- shape
def _find_header(rows: list[list[Any]]) -> int:
    """Index (0-based) of the header row, or -1.

    The header is the first wide, fully textual row followed by rows of a similar
    width -- which is what separates it from a title banner or a stray note.
    """
    widths = [sum(1 for v in r if as_text(v)) for r in rows]
    if not widths:
        return -1
    body_width = max(widths)
    for i, row in enumerate(rows[:30]):
        cells = [as_text(v) for v in row]
        filled = sum(1 for c in cells if c)
        if filled < max(config.TABLE_MIN_COLS, body_width * 0.6):
            continue
        texty = sum(1 for c in cells if c and parse_number(c) is None and not parse_date(c))
        if texty < filled * 0.7:
            continue
        following = widths[i + 1:i + 6]
        if following and sum(1 for w in following if w >= filled * 0.5) >= len(following) * 0.6:
            return i
    return -1


def fingerprint(labels: Sequence[str]) -> str:
    """Identity of a schema: sheets sharing a fingerprint can be UNIONed."""
    norm = "|".join(re.sub(r"\s+", " ", (x or "").strip().lower()) for x in labels)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def probe_rows(sheet: str, rows: list[list[Any]]) -> SheetProbe:
    """Decide whether a block of sampled rows is a table, and type its columns."""
    if len(rows) < config.TABLE_MIN_ROWS + 1:
        return SheetProbe(sheet, 0, [], ok=False, reason="too few rows")

    h = _find_header(rows)
    if h < 0:
        return SheetProbe(sheet, 0, [], ok=False, reason="no single header row found")

    header = [as_text(v) for v in rows[h]]
    while header and not header[-1]:
        header.pop()
    n_cols = len(header)
    if n_cols < config.TABLE_MIN_COLS:
        return SheetProbe(sheet, h + 1, [], ok=False, reason=f"only {n_cols} columns")
    if sum(1 for c in header if c) < n_cols * 0.7:
        return SheetProbe(sheet, h + 1, [], ok=False, reason="header has gaps")

    body = [r for r in rows[h + 1:] if any(as_text(v) for v in r)]
    if len(body) < config.TABLE_MIN_ROWS:
        return SheetProbe(sheet, h + 1, [], ok=False, reason="too few data rows")
    consistent = sum(1 for r in body
                     if sum(1 for v in r[:n_cols] if as_text(v)) >= n_cols * 0.4)
    if consistent < len(body) * 0.7:
        return SheetProbe(sheet, h + 1, [], ok=False, reason="rows do not match the header")

    taken: set[str] = set()
    cols: list[Column] = []
    for i, label in enumerate(header):
        label = label or f"column_{i + 1}"
        col = Column(ord=i, label=label, name=sanitize(label, taken, f"col_{i + 1}"))
        cols.append(infer_column(col, [r[i] if i < len(r) else None for r in body]))

    probe = SheetProbe(sheet, h + 1, cols, sampled=len(body))
    probe.fingerprint = fingerprint(probe.labels)
    return probe


# --------------------------------------------------------------------------- readers
def sheet_names(path: Path) -> list[str]:
    if path.suffix.lower() in {".csv", ".tsv"}:
        return [path.stem]
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    try:
        return list(wb.sheetnames)
    finally:
        wb.close()


def iter_rows(path: Path, sheet: str) -> Iterator[list[Any]]:
    """Stream a sheet row by row so a 100k-row workbook never lands in memory."""
    if path.suffix.lower() in {".csv", ".tsv"}:
        import csv

        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            sample = fh.read(64 * 1024)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except Exception:                                   # noqa: BLE001
                dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
            for row in csv.reader(fh, dialect):
                yield list(row)
        return

    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    try:
        for row in wb[sheet].iter_rows(values_only=True):
            yield list(row)
    finally:
        wb.close()


def _probe_rows_per_sheet(path: Path) -> Iterator[tuple[str, list[list[Any]], str]]:
    """The first TABLE_PROBE_ROWS rows of each sheet, from ONE open of the workbook.

    Opening it per sheet -- as listing the names and then streaming each one did --
    parses an 11 MB xlsx from scratch once for the names and once more for every
    sheet in it, which on a multi-sheet export is most of the detection cost.
    """
    limit = config.TABLE_PROBE_ROWS
    if path.suffix.lower() in {".csv", ".tsv"}:
        rows: list[list[Any]] = []
        for row in iter_rows(path, path.stem):
            rows.append(row)
            if len(rows) >= limit:
                break
        yield path.stem, rows, ""
        return

    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    try:
        for name in list(wb.sheetnames):
            try:
                rows = []
                for row in wb[name].iter_rows(values_only=True):
                    rows.append(list(row))
                    if len(rows) >= limit:
                        break
                yield name, rows, ""
            except Exception as exc:                            # noqa: BLE001
                yield name, [], f"{type(exc).__name__}: {exc}"
    finally:
        wb.close()


def probe_file(path: Path) -> FileProbe:
    """Read the first rows of every sheet and report which ones are tables."""
    out = FileProbe(path=path)
    try:
        sheets = list(_probe_rows_per_sheet(path))
    except Exception as exc:                                    # noqa: BLE001
        out.rejected.append(SheetProbe("", 0, [], ok=False, reason=f"unreadable: {exc}"))
        return out

    for name, rows, error in sheets:
        if error:
            probe = SheetProbe(name, 0, [], ok=False, reason=error)
        else:
            try:
                probe = probe_rows(name, rows)
            except Exception as exc:                            # noqa: BLE001
                probe = SheetProbe(name, 0, [], ok=False,
                                   reason=f"{type(exc).__name__}: {exc}")
        if probe.ok:
            out.sheets.append(probe)
        else:
            out.rejected.append(probe)
    return out

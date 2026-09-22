"""The semantic layer: what the columns *mean*, and which direction is good.

A profiled catalogue knows that ``claim_status`` contains 'Rejected'.  It does not
know that rejection is bad, that ``generated_claim_amount`` should be summed while
``hour_meter_reading`` should not, or that a user asking about "breakdowns" means
the rows where ``claim_type`` starts with 'CM'.  Every one of those is a judgement
about the business, and no amount of profiling recovers it from the values alone.

So it is written down, in one editable YAML file, and used in four places:

* **trends** -- which outcome counts as failure, instead of guessing from a word list;
* **the analyst agent** -- so "where are we lagging" has a definition of lagging;
* **text-to-SQL** -- so a metric is computed the same way every time it is asked for;
* **document retrieval** -- the glossary is a synonym table, which is what lets a
  question about "breakdowns" find a page that only ever says "corrective maintenance".

Nobody wants to write this from scratch, so ``draft()`` produces a first version from
the catalogue with one model call, and the file is plain YAML to edit afterwards.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app import config
from app.tables import catalog

_lock = threading.RLock()
_cache: dict | None = None
_mtime: float = 0.0

HIGHER, LOWER = "higher_is_better", "lower_is_better"


@dataclass
class Metric:
    name: str
    sql: str = ""
    direction: str = HIGHER
    means: str = ""
    format: str = "number"

    @property
    def good_when(self) -> str:
        return "higher is better" if self.direction == HIGHER else "lower is better"


@dataclass
class Outcome:
    column: str
    good: list[str] = field(default_factory=list)
    bad: list[str] = field(default_factory=list)

    @property
    def worst(self) -> str:
        return self.bad[0] if self.bad else ""


@dataclass
class TableSemantics:
    name: str
    description: str = ""
    grain: str = ""
    domain: str = ""
    dimensions: dict[str, str] = field(default_factory=dict)
    outcomes: list[Outcome] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    glossary: list["Glossary"] = field(default_factory=list)

    def outcome_for(self, column: str) -> Outcome | None:
        return next((o for o in self.outcomes if o.column == column), None)


@lru_cache(maxsize=2048)
def word_pattern(word: str) -> re.Pattern:
    """Match a term as a whole word, plural included.

    Without the optional 's', "how many breakdowns" misses a glossary entry that
    reads 'Breakdown' -- which is most of how people actually type.

    Cached because every question recompiles this once per glossary term per
    alternative wording, and the set of terms barely changes between questions.
    """
    return re.compile(rf"\b{re.escape(str(word).lower())}s?\b", re.IGNORECASE)


@dataclass
class Glossary:
    term: str
    means: str = ""
    also: list[str] = field(default_factory=list)

    @property
    def words(self) -> list[str]:
        """Every wording of this concept, including what it stands for.

        ``means`` belongs here and not just in the description: an entry reading
        ``BD = Breakdown`` exists precisely so that somebody typing "breakdown"
        finds the rows and pages that only ever say "BD".
        """
        out = [self.term, *self.also]
        if self.means and len(self.means.split()) <= 4:
            out.append(self.means)
        return [str(w).strip() for w in out if str(w).strip()]

    def matches(self, text: str) -> bool:
        low = text.lower()
        return any(word_pattern(w).search(low) for w in self.words)


def parse_forgiving(text: str, max_drop: int = 10) -> tuple[dict, list[str]]:
    """Parse YAML a model wrote, dropping the lines it got wrong.

    Real jargon breaks naive YAML: a term like ``"9+1" topology`` is an unquoted
    scalar after a quoted one, and PyYAML rejects the whole document for it. Losing
    twenty-four good glossary entries because of the twenty-fifth is the wrong
    trade, so the offending line is removed and the parse retried.
    """
    dropped: list[str] = []
    lines = text.splitlines()
    for _ in range(max_drop + 1):
        try:
            data = yaml.safe_load("\n".join(lines)) or {}
            return (data if isinstance(data, dict) else {}), dropped
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            if mark is None or not (0 <= mark.line < len(lines)):
                raise
            # Remove the whole list item the bad line belongs to, not just the line,
            # or its continuation lines become orphans and fail in turn.
            start = mark.line
            while start > 0 and not lines[start].lstrip().startswith("- "):
                if not lines[start].strip():
                    break
                start -= 1
            end = start + 1
            while (end < len(lines) and lines[end].strip()
                   and not lines[end].lstrip().startswith("- ")):
                end += 1
            dropped.append(lines[start].strip()[:80])
            del lines[start:end]
    raise ValueError("could not parse the model's YAML")


def path() -> Path:
    return Path(config.STORAGE_DIR) / "semantics.yaml"


# --------------------------------------------------------------------------- load
def _parse(raw: dict) -> dict:
    tables: dict[str, TableSemantics] = {}
    for name, body in (raw.get("tables") or {}).items():
        body = body or {}
        info = TableSemantics(
            name=name,
            description=str(body.get("description", "")),
            grain=str(body.get("grain", "")),
            dimensions={k: str(v.get("means", "") if isinstance(v, dict) else v)
                        for k, v in (body.get("dimensions") or {}).items()},
        )
        for column, states in (body.get("outcomes") or {}).items():
            states = states or {}
            info.outcomes.append(Outcome(
                column=column,
                good=[str(x) for x in (states.get("good") or [])],
                bad=[str(x) for x in (states.get("bad") or [])]))
        info.domain = str(body.get("domain", ""))
        for entry in (body.get("glossary") or []):
            if isinstance(entry, dict) and entry.get("term"):
                info.glossary.append(Glossary(
                    term=str(entry["term"]), means=str(entry.get("means", "")),
                    also=[str(x) for x in (entry.get("also") or [])]))
        for metric, spec in (body.get("metrics") or {}).items():
            spec = spec or {}
            direction = str(spec.get("direction", HIGHER))
            info.metrics.append(Metric(
                name=metric, sql=str(spec.get("sql", "")),
                direction=direction if direction in (HIGHER, LOWER) else HIGHER,
                means=str(spec.get("means", "")),
                format=str(spec.get("format", "number"))))
        tables[name] = info

    glossary = []
    for entry in (raw.get("glossary") or []):
        if isinstance(entry, dict) and entry.get("term"):
            glossary.append(Glossary(
                term=str(entry["term"]), means=str(entry.get("means", "")),
                also=[str(x) for x in (entry.get("also") or [])]))
    return {"tables": tables, "glossary": glossary,
            "domain": str(raw.get("domain", ""))}


def load(force: bool = False) -> dict:
    """The semantic layer, reloaded whenever the file changes on disk."""
    global _cache, _mtime
    file = path()
    with _lock:
        try:
            stamp = file.stat().st_mtime
        except OSError:
            _cache, _mtime = {"tables": {}, "glossary": [], "domain": ""}, 0.0
            return _cache
        if _cache is None or force or stamp != _mtime:
            try:
                raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
                _cache = _parse(raw if isinstance(raw, dict) else {})
            except (yaml.YAMLError, OSError) as exc:
                _cache = {"tables": {}, "glossary": [], "domain": "",
                          "error": f"{type(exc).__name__}: {exc}"}
            _mtime = stamp
        return _cache


def exists() -> bool:
    return path().exists()


def save(text: str) -> None:
    """Write the file as given, after checking it parses."""
    yaml.safe_load(text)                       # raises on malformed YAML
    file = path()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    load(force=True)


def raw_text() -> str:
    try:
        return path().read_text(encoding="utf-8")
    except OSError:
        return ""


# --------------------------------------------------------------------------- use
def table(name: str) -> TableSemantics | None:
    return load()["tables"].get(name)


def bad_values(table_name: str, column: str) -> list[str]:
    """States that count as failure for this column, if anyone said so."""
    info = table(table_name)
    outcome = info.outcome_for(column) if info else None
    return list(outcome.bad) if outcome else []


def glossary_for(table_name: str = "") -> list[Glossary]:
    """Terms in scope: the shared ones, plus this table's own.

    Scoping matters as soon as there is more than one dataset. 'PM' means
    preventive maintenance in a service export and project manager in an HR one;
    a single global list would tell the query planner the wrong thing about
    whichever dataset it was not written for.
    """
    data = load()
    out = list(data["glossary"])
    if table_name:
        info = data["tables"].get(table_name)
        if info:
            out = list(info.glossary) + out      # the table's own wins on a clash
    else:
        for info in data["tables"].values():
            out += info.glossary
    seen: set[str] = set()
    unique: list[Glossary] = []
    for g in out:
        key = g.term.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(g)
    return unique


def relevant_glossary(text: str, limit: int = 12,
                      table_name: str = "") -> list[Glossary]:
    """Glossary entries the wording of this question touches."""
    return [g for g in glossary_for(table_name) if g.matches(text)][:limit]


def synonyms(text: str, table_name: str = "") -> list[str]:
    """Alternative wordings for a question, from the glossary alone.

    This is the cheap half of query expansion: no model call, and it captures the
    vocabulary the documents actually use rather than the vocabulary the user has.
    """
    out: list[str] = []
    for entry in relevant_glossary(text, table_name=table_name):
        words = entry.words
        for word in words:
            pattern = word_pattern(word)     # same matcher, so plurals swap too
            if not pattern.search(text):
                continue
            for alt in words:
                if alt.lower() == word.lower():
                    continue
                swapped = pattern.sub(alt, text)
                if swapped != text and swapped not in out:
                    out.append(swapped)
    return out[:6]


def render(table_name: str = "", question: str = "") -> str:
    """The semantic layer as prompt text: only the parts that are relevant."""
    data = load()
    if not data["tables"] and not data["glossary"]:
        return ""
    lines: list[str] = []
    if data.get("domain"):
        lines.append(f"Domain: {data['domain']}")

    for name, info in data["tables"].items():
        if table_name and name != table_name:
            continue
        block = [f"TABLE {name}"]
        if info.description:
            block.append(f"  what it holds: {info.description}")
        if info.grain:
            block.append(f"  one row is: {info.grain}")
        for column, means in info.dimensions.items():
            if means:
                block.append(f"  {column}: {means}")
        for outcome in info.outcomes:
            good = ", ".join(repr(v) for v in outcome.good) or "-"
            bad = ", ".join(repr(v) for v in outcome.bad) or "-"
            block.append(f"  {outcome.column}: GOOD = {good}; BAD = {bad}")
        for metric in info.metrics:
            block.append(f"  metric {metric.name} = {metric.sql}"
                         f"  ({metric.good_when}"
                         + (f"; {metric.means}" if metric.means else "") + ")")
        if len(block) > 1:
            lines.append("\n".join(block))

    entries = relevant_glossary(question) if question else load()["glossary"][:20]
    if entries:
        lines.append("Terms:\n" + "\n".join(
            f"  {g.term} = {g.means}"
            + (f" (also: {', '.join(g.also)})" if g.also else "")
            for g in entries))
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- draft
DRAFT_SYSTEM = """You are writing a semantic layer for a dataset: what the columns mean
in business terms, and which direction is good.

{schema}

Reply with ONLY a YAML document in exactly this shape:

domain: <one line naming the business domain this data is from>
tables:
  <table_name>:
    description: <what this table holds, one sentence>
    grain: <what one row represents>
    dimensions:
      <column>: {{means: <what this column identifies, a few words>}}
    outcomes:
      <status-like column>:
        good: [<values that mean success>]
        bad: [<values that mean failure>]
    metrics:
      <snake_case_metric_name>:
        sql: <a SQL aggregate over this table, e.g. SUM(amount)>
        direction: higher_is_better | lower_is_better
        means: <one short line>
glossary:
  - term: <abbreviation or jargon that appears in this data>
    means: <what it stands for>
    also: [<other words people use for the same thing>]

Rules:
- Use ONLY column names and literal values that appear in the schema above. Never
  invent a column or a value.
- outcomes: only for columns whose values genuinely describe success or failure.
  A value that is merely a stage ("Submitted", "Pending") is neither good nor bad -
  leave it out of both lists.
- metrics: 3 to 8 of them, the ones somebody would actually ask for. Never aggregate
  a column described as an identifier.
- direction: think about what the business wants. A cost or a rejection rate is
  lower_is_better; volume or approved value is higher_is_better.
- glossary: include the abbreviations that appear in the data values themselves
  (for example a claim type code), and the everyday words a person would use for
  them instead. This is what lets a plain-English question match the documents.
- No commentary, no code fences, only the YAML."""


def draft(model: str | None = None, only: list[str] | None = None) -> str:
    """Ask the model for a description, from the catalogue. One call.

    ``only`` restricts it to named tables, which is what makes an incremental
    update possible: describe the new spreadsheet, leave the rest of the file alone.
    """
    from app.core import router

    infos = catalog.list_tables()
    if only:
        infos = [t for t in infos if t.name in set(only)]
    schema = catalog.render_schema(infos)
    if not schema:
        raise ValueError("no tables have been imported yet")
    ask = ("Write the semantic layer for these tables: " + ", ".join(only)
           if only else "Write the semantic layer.")
    text, _used = router.complete(
        [{"role": "system", "content": DRAFT_SYSTEM.format(schema=schema)},
         {"role": "user", "content": ask}],
        role="answer", model=model, temperature=0.1,
        # A layer for a 23-column table does not fit in the default answer budget,
        # and a truncated YAML document is not recoverable.
        max_tokens=4000, timeout=180.0)
    body = re.sub(r"^```(?:ya?ml)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    data, dropped = parse_forgiving(body)
    if dropped:
        print(f"[semantics] skipped {len(dropped)} malformed entr(ies): "
              f"{'; '.join(dropped[:3])}", flush=True)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True,
                          default_flow_style=False)


DOC_TERMS_SYSTEM = """You are building a glossary from someone's documents.

Below are excerpts from files they have indexed. Find the abbreviations, acronyms and
domain jargon that a reader outside this organisation would not know.

Reply with ONLY a YAML document:

glossary:
  - term: <the abbreviation or jargon exactly as it appears>
    means: <what it stands for, a few words>
    also: [<the everyday words someone would use instead>]

Rules:
- Only terms that actually appear in the excerpts. Never invent one.
- Skip ordinary English. A term earns its place only if not knowing it would stop
  someone finding the right page.
- "also" is the important field: it is what lets a plain-English question match a
  page written in jargon. For MTBF, that is ["mean time between failures",
  "reliability", "failure interval"].
- At most 25 entries, the most useful first. No commentary, no code fences.
- Always wrap term and means in single quotes, so a value containing a quote, colon
  or bracket stays valid YAML."""


def draft_document_terms(model: str | None = None, sample: int = 40) -> str:
    """Read the indexed documents and write a glossary of their jargon.

    Spreadsheet values give you the codes in the data; they say nothing about the
    vocabulary of a PDF. Those terms are exactly the ones that make a question miss:
    somebody asks about "downtime" and the report only ever writes "MDT".
    """
    from app.core import router
    from app.core.store import get_store

    store = get_store()
    rows = store._q(                                            # noqa: SLF001
        "SELECT c.text, f.name FROM chunks c JOIN files f ON f.id = c.file_id "
        "WHERE c.n_chars > 400 ORDER BY RANDOM() LIMIT ?", (sample,))
    if not rows:
        raise ValueError("no document text has been indexed yet")
    excerpt = "\n\n".join(f"[{r['name']}]\n{r['text'][:900]}" for r in rows)[:24000]
    text, _used = router.complete(
        [{"role": "system", "content": DOC_TERMS_SYSTEM},
         {"role": "user", "content": excerpt}],
        role="answer", model=model, temperature=0.1,
        max_tokens=2000, timeout=180.0)
    body = re.sub(r"^```(?:ya?ml)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    data, dropped = parse_forgiving(body)
    if dropped:
        print(f"[semantics] skipped {len(dropped)} malformed term(s)", flush=True)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True,
                          default_flow_style=False)


# --------------------------------------------------------------------------- staleness
def described() -> set[str]:
    return set(load()["tables"])


def missing_tables() -> list[str]:
    """Imported tables nobody has described yet."""
    try:
        names = [t.name for t in catalog.list_tables(with_columns=False)]
    except Exception:                                           # noqa: BLE001
        return []
    return [n for n in names if n not in described()]


def status() -> dict:
    data = load()
    missing = missing_tables()
    # n_tables, not "tables": the API also returns the table *list* under that name,
    # and the collision silently dropped the count.
    return {"exists": exists(), "n_tables": len(data["tables"]),
            "terms": len(glossary_for()), "missing": missing,
            "stale": bool(missing), "error": data.get("error", "")}


# --------------------------------------------------------------------------- merge
def _block_end(lines: list[str], start: int) -> int:
    """Index just past the indented block that begins after ``lines[start]``."""
    i = start + 1
    while i < len(lines):
        line = lines[i]
        if line.strip() and not line.startswith((" ", "\t")):
            return i
        i += 1
    return len(lines)


def _block_indent(lines: list[str], start: int, end: int, default: int = 2) -> int:
    """How far the existing entries under a key are indented."""
    for line in lines[start + 1:end]:
        if line.strip():
            return len(line) - len(line.lstrip())
    return default


def _reindent(body: list[str], indent: int) -> list[str]:
    """Line up a dumped fragment with the block it is being spliced into.

    PyYAML writes a sequence under a mapping key at column zero, so pasting one
    under an existing indented block produces a top-level token and a parse error.
    """
    if not body:
        return body
    first = next((l for l in body if l.strip()), "")
    current = len(first) - len(first.lstrip())
    shift = indent - current
    if shift == 0:
        return body
    if shift > 0:
        return [(" " * shift + l) if l.strip() else l for l in body]
    return [l[-shift:] if l[:-shift].strip() == "" else l.lstrip() for l in body]


def merge(existing: str, addition: str) -> str:
    """Splice new table and glossary entries into a file, textually.

    Deliberately not a parse-and-redump: PyYAML discards comments and reorders
    keys, so a round trip would quietly destroy the notes somebody wrote to explain
    *why* a metric points the way it does. Existing text is never rewritten -- new
    blocks are inserted, and a table that is already described is skipped entirely.
    """
    try:
        new_raw = yaml.safe_load(addition) or {}
        old_raw = yaml.safe_load(existing) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"could not merge: {exc}") from exc
    if not isinstance(new_raw, dict) or not isinstance(old_raw, dict):
        raise ValueError("both documents must be YAML mappings")

    have_tables = set((old_raw.get("tables") or {}))
    add_tables = {k: v for k, v in (new_raw.get("tables") or {}).items()
                  if k not in have_tables}
    have_terms = {str(e.get("term", "")).lower()
                  for e in (old_raw.get("glossary") or []) if isinstance(e, dict)}
    add_terms = [e for e in (new_raw.get("glossary") or [])
                 if isinstance(e, dict) and str(e.get("term", "")).lower() not in have_terms]
    if not add_tables and not add_terms:
        return existing

    text = existing if existing.endswith("\n") or not existing else existing + "\n"
    lines = text.splitlines()

    if add_tables:
        fragment = yaml.safe_dump({"tables": add_tables}, sort_keys=False,
                                  allow_unicode=True, default_flow_style=False)
        body = fragment.splitlines()[1:]                    # drop the "tables:" line
        at = next((i for i, l in enumerate(lines)
                   if re.match(r"^tables:\s*$", l)), -1)
        if at >= 0:
            end = _block_end(lines, at)
            lines[end:end] = _reindent(body, _block_indent(lines, at, end))
        else:
            lines += ["tables:"] + body

    if add_terms:
        fragment = yaml.safe_dump({"glossary": add_terms}, sort_keys=False,
                                  allow_unicode=True, default_flow_style=False)
        body = fragment.splitlines()[1:]
        at = next((i for i, l in enumerate(lines)
                   if re.match(r"^glossary:\s*$", l)), -1)
        if at >= 0:
            end = _block_end(lines, at)
            lines[end:end] = _reindent(body, _block_indent(lines, at, end))
        else:
            lines += ["glossary:"] + body

    out = "\n".join(lines).rstrip() + "\n"
    yaml.safe_load(out)                                     # never write a broken file
    return out


def update(model: str | None = None) -> dict:
    """Describe whatever is new, keeping every existing line as it was."""
    new = missing_tables()
    if not new:
        return {"added": [], "text": raw_text(), "changed": False}
    addition = draft(model=model, only=new)
    merged = merge(raw_text(), addition) if exists() else addition
    return {"added": new, "text": merged, "changed": merged != raw_text()}
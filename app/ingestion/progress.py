"""Live throughput and a remaining-time estimate for an index run.

Why this is not simply ``files_done / elapsed``
-----------------------------------------------
Files are not comparable units of work. A 2 KB note and a 40 MB scanned contract
both count as "one file", so a file-count estimate swings by an order of magnitude
as the folder's mix changes, and a cumulative average never recovers from the first
few files being unrepresentative. Measured against a recorded run of this pipeline
it was wrong by 68% of the run on average, and by 50% even in its second half.

Bytes are not the answer either, and on this pipeline they are actively worse: a
spreadsheet is streamed into SQL at about a megabyte per second while a text file
has to be chunked and embedded, which is hundreds of times slower per byte. Size and
cost are close to anti-correlated, and a bytes-per-second estimate scored worse than
useless on the same recording.

So cost is measured rather than assumed:

* **weight** -- how expensive a byte of each file type has actually turned out to
  be, in seconds per byte, kept as a moving average and learned as the run goes.
  Size times weight converts a file into *work*, which is comparable across formats.
* **throughput** -- how much work the pipeline retires per wall-clock second, over a
  sliding window of the last few seconds rather than over the whole run. Being
  wall-clock it already accounts for the worker pool; eight threads simply retire
  work eight times faster.

The estimate is ``work outstanding / work retired per second``.

The one subtlety that matters: both sides are priced with the weights believed
*now*. An earlier version credited each file at the weight believed when it
finished, so the backlog was denominated in current weights while the rate was
denominated in stale ones, and every time a new file type revised a weight the
estimate lurched -- it swung by ±175s on a 470s run. Repricing the window on every
read costs nothing at these sizes and keeps the two halves of the division in the
same unit.

A file type nothing has been measured for is costed at the most expensive type that
has been. The two errors are not symmetric: an estimate that starts long and comes
down as the run proves it wrong is ordinary, while one that promises forty seconds
and then climbs to seven minutes is what people mean when they say a progress bar
lies.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Iterable

# Seconds of history the throughput is measured over. Shorter reacts faster but gets
# noisy: at 15s the error roughly doubled against a recorded run.
WINDOW = 45.0
# Weight of a new per-file measurement against the running one for that type.
WEIGHT_ALPHA = 0.3
# Weight of a new estimate against the displayed one, so the number does not jitter
# between refreshes while still being free to move when the truth moves.
ETA_ALPHA = 0.3
# Until this much has gone by, any rate computed is noise, and an estimate is worse
# than no estimate -- so none is offered and the bar says "estimating".
MIN_ELAPSED = 1.5
MIN_SAMPLES = 3
# Only ever used to rank one never-seen file type against another.
DEFAULT_WEIGHT = 1e-6


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} GB"


def human_time(seconds: float | None) -> str:
    """A duration the way a download dialog writes one."""
    if seconds is None:
        return "estimating"
    seconds = max(0.0, float(seconds))
    if seconds < 1:
        return "less than a second"
    if seconds < 60:
        return f"{int(round(seconds))}s"
    if seconds < 3600:
        m, s = divmod(int(round(seconds)), 60)
        return f"{m}m {s:02d}s"
    h, rest = divmod(int(round(seconds)), 3600)
    m = rest // 60
    return f"{h}h {m:02d}m"


class Meter:
    """Tracks progress of one index run. Safe to read from another thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = time.monotonic()
        self.total_bytes = 0
        self.done_bytes = 0
        self.total_files = 0
        self.done_files = 0
        self._remaining: dict[str, int] = {}      # ext -> bytes still to do
        self._done: dict[str, int] = {}           # ext -> bytes retired
        self._weight: dict[str, float] = {}       # ext -> seconds of work per byte
        self._seen: dict[str, int] = {}           # ext -> measurements taken
        self._window: deque = deque()             # (monotonic, ext, size) recently done
        self._eta: float | None = None
        self._rate = 0.0                          # bytes/sec, for display only

    # ------------------------------------------------------------------ setup
    def plan(self, files: Iterable[tuple[str, int]]) -> None:
        """Declare the work: (extension, size in bytes) for every file to be visited."""
        with self._lock:
            for ext, size in files:
                ext = (ext or "").lower()
                size = max(int(size), 1)
                self.total_files += 1
                self.total_bytes += size
                self._remaining[ext] = self._remaining.get(ext, 0) + size
            self._window.clear()

    def plan_bulk(self, groups: Iterable[tuple[str, int, int]]) -> None:
        """Declare the work as (extension, file count, total bytes) per type.

        The same information as ``plan``, from a GROUP BY rather than from a list of
        every file -- which is the difference between constant memory and several
        gigabytes at twenty million files.
        """
        with self._lock:
            for ext, n_files, n_bytes in groups:
                ext = (ext or "").lower()
                n_files, n_bytes = int(n_files), max(int(n_bytes), int(n_files))
                self.total_files += n_files
                self.total_bytes += n_bytes
                self._remaining[ext] = self._remaining.get(ext, 0) + n_bytes
            self._window.clear()

    # ------------------------------------------------------------------ weights
    def _weight_for(self, ext: str) -> float:
        """Seconds of work per byte for this file type; the worst known if untried."""
        known = self._weight.get(ext)
        if known is not None and self._seen.get(ext, 0) >= 2:
            return known
        measured = [w for e, w in self._weight.items() if self._seen.get(e, 0) >= 2]
        if known is not None:
            return max([known] + measured) if measured else known
        return max(measured) if measured else DEFAULT_WEIGHT

    def _record_cost(self, ext: str, size: int, seconds: float) -> None:
        observed = max(float(seconds), 0.0) / size
        previous = self._weight.get(ext)
        self._weight[ext] = (observed if previous is None else
                             previous * (1 - WEIGHT_ALPHA) + observed * WEIGHT_ALPHA)
        self._seen[ext] = self._seen.get(ext, 0) + 1

    # ------------------------------------------------------------------ progress
    def complete(self, ext: str, size: int, seconds: float) -> None:
        """One file is done with: how big it was and how long it actually cost.

        A skipped file measures as almost free, which is correct -- if the rest of
        the run is skips too, it really will be quick.
        """
        ext = (ext or "").lower()
        size = max(int(size), 1)
        now = time.monotonic()
        with self._lock:
            self._remaining[ext] = max(0, self._remaining.get(ext, 0) - size)
            self._done[ext] = self._done.get(ext, 0) + size
            self.done_files += 1
            self.done_bytes += size
            self._record_cost(ext, size, seconds)
            self._window.append((now, ext, size))
            while len(self._window) > 1 and now - self._window[0][0] > WINDOW:
                self._window.popleft()

    def observe(self, ext: str, size: int, seconds: float) -> None:
        """More cost for a file already counted as done, without retiring it twice."""
        if seconds <= 0:
            return
        with self._lock:
            self._record_cost((ext or "").lower(), max(int(size), 1), seconds)

    # ------------------------------------------------------------------ readout
    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            elapsed = now - self.started
            # Everything below is priced with the weights believed right now, so the
            # backlog and the rate are always in the same unit.
            outstanding = sum(b * self._weight_for(e)
                              for e, b in self._remaining.items() if b > 0)
            retired = sum(b * self._weight_for(e) for e, b in self._done.items())

            rate_work = 0.0
            if len(self._window) >= 2:
                span = now - self._window[0][0]
                if span > 0.5:
                    work = sum(sz * self._weight_for(e) for _t, e, sz in self._window)
                    bytes_in = sum(sz for _t, _e, sz in self._window)
                    rate_work = work / span
                    self._rate = bytes_in / span

            if outstanding <= 0 and self.done_files:
                # Nothing left is a fact, not an estimate. Easing into it through the
                # moving average left the bar promising a few more seconds of work
                # that had already finished.
                self._eta = 0.0
            elif (elapsed >= MIN_ELAPSED and self.done_files >= MIN_SAMPLES
                    and rate_work > 0):
                raw = outstanding / rate_work
                self._eta = (raw if self._eta is None else
                             self._eta * (1 - ETA_ALPHA) + raw * ETA_ALPHA)

            pct = 100.0 * retired / (retired + outstanding) if retired + outstanding else 0.0
            return {
                "files_done": self.done_files,
                "files_total": self.total_files,
                "bytes_done": self.done_bytes,
                "bytes_total": self.total_bytes,
                "bytes_per_sec": round(self._rate, 1),
                "pct": round(min(100.0, max(0.0, pct)), 1),
                "eta": None if self._eta is None else round(max(0.0, self._eta), 1),
                "elapsed": round(elapsed, 1),
            }

    def finish(self) -> None:
        with self._lock:
            self._eta = 0.0
            self._remaining.clear()


# --------------------------------------------------------------------------- console
def _can_print(sample: str, stream) -> bool:
    """Will this terminal actually render these characters?

    cmd.exe on a default code page cannot, and a UnicodeEncodeError halfway through
    a progress line is a worse outcome than a plainer bar.
    """
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        sample.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


class ConsoleBar:
    """A one-line progress bar that redraws in place, for ``python rag.py index``.

    ``interactive`` is false when output is being piped or redirected: a bar that
    redraws four times a second turns a log file into a megabyte of carriage
    returns, so there it prints a plain line now and then instead.
    """

    def __init__(self, stream=None, width: int | None = None,
                 interactive: bool | None = None) -> None:
        import shutil
        import sys

        self.stream = stream or sys.stdout
        if interactive is None:
            try:
                interactive = bool(self.stream.isatty())
            except Exception:                                 # noqa: BLE001
                interactive = False
        self.interactive = interactive
        self._last_line_at = 0.0
        self.fancy = _can_print("\u2588\u2591", self.stream)
        self.full, self.empty = ("\u2588", "\u2591") if self.fancy else ("#", "-")
        try:
            self.width = width or shutil.get_terminal_size((100, 24)).columns
        except Exception:                                     # noqa: BLE001
            self.width = width or 100
        self._last = ""

    def _bar(self, pct: float, size: int) -> str:
        filled = int(round(size * max(0.0, min(100.0, pct)) / 100.0))
        return self.full * filled + self.empty * (size - filled)

    def render(self, snap: dict, phase: str = "", current: str = "") -> str:
        """One line that fits the terminal, dropping the least useful part first.

        A narrow window is the normal case on Windows, and truncating the line from
        the right threw away the remaining time -- the one number somebody watching
        a long run actually wants. So the filename goes first, then the throughput,
        and the estimate is the last thing to be given up.
        """
        pct = float(snap.get("pct") or 0.0)
        done, total = snap.get("files_done", 0), snap.get("files_total", 0)
        rate = float(snap.get("bytes_per_sec") or 0.0)

        eta = f"ETA {human_time(snap.get('eta'))}"
        files = f"{done:,}/{total:,} files"
        speed = f"{human_bytes(rate)}/s" if rate > 0 else "-- B/s"
        # Least important first: each is dropped in turn until the line fits.
        optional = []
        if current:
            optional.append(current)
        if phase and phase not in ("indexing", "done"):
            optional.append(phase)
        optional += [speed, files]

        limit = max(24, self.width - 1)
        for drop in range(len(optional) + 1):
            kept = optional[drop:]
            size = 22 if self.width >= 90 else (14 if self.width >= 60 else 8)
            head = f"  [{self._bar(pct, size)}] {pct:5.1f}%"
            line = "   ".join([head, eta] + list(reversed(kept)))
            if len(line) <= limit:
                return line
            # The filename is the one piece worth shortening rather than dropping.
            if drop == 0 and current:
                room = limit - (len(line) - len(current))
                if room > 12:
                    cut = current[:room - 1] + ("\u2026" if self.fancy else "")
                    if not self.fancy:
                        cut = current[:room - 3] + "..."
                    optional[0] = cut
                    line = "   ".join([head, eta] + list(reversed(optional)))
                    if len(line) <= limit:
                        return line
        # Even the short form does not fit. Cutting it would leave "ETA 3m " on
        # screen, so the bar goes rather than half a number.
        short = f"  [{self._bar(pct, 8)}] {pct:5.1f}%   {eta}"
        return short if len(short) <= limit else f"  {pct:5.1f}%  {eta}"[:limit]

    def draw(self, snap: dict, phase: str = "", current: str = "") -> None:
        if not self.interactive:
            now = time.monotonic()
            if now - self._last_line_at < 15.0:
                return
            self._last_line_at = now
            self.stream.write(self.render(snap, phase, current).strip() + "\n")
            self.stream.flush()
            return
        line = self.render(snap, phase, current)
        pad = " " * max(0, len(self._last) - len(line))
        self.stream.write("\r" + line + pad)
        self.stream.flush()
        self._last = line

    def done(self, message: str = "") -> None:
        if self.interactive:
            self.stream.write("\r" + " " * (len(self._last) + 2) + "\r")
        if message:
            self.stream.write(message + "\n")
        self.stream.flush()
        self._last = ""

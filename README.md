# Local RAG

Retrieval-augmented chat over your own documents. Point it at a folder, index it, and
ask questions in the browser. Reading, indexing, search and every chart happen on your
machine; only the question you type and the passages it selected go to the model that
writes the answer.

## Install

One command. It fetches the app, sets up a private Python environment and starts it.

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/akashverma-lang/RAG/main/install.ps1 | iex
```

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/akashverma-lang/RAG/main/install.sh | bash
```

On first run a setup screen asks for the folder holding your documents and a free API
key — [Groq](https://console.groq.com/keys) or
[Gemini](https://aistudio.google.com/apikey). Either one is enough.

From a clone instead:

```powershell
setup.bat        # once: create the environment and install packages
run.bat          # start it; the setup screen handles the rest
```

- Installing by hand, or when the installer fails → **[SETUP.md](SETUP.md)**

### Why it installs rather than runs in the browser

The app reads a folder on your own computer. A page on the web cannot do that — the
browser forbids it — so a hosted copy could only work by having you upload your
documents to someone else's server first. Installing keeps the documents where they
already are. What is shared is the link to `site/`, a static page that explains the
app and hands over the command above; see [site/README.md](site/README.md) for
putting it on a free host.

## Layout

```
RAG/
├─ rag.py                     single entry point: doctor | index | serve | test
├─ launcher.py                starts the server and opens the browser
├─ setup.bat / index.bat / run.bat      double-click equivalents
├─ install.ps1 / install.sh   the one-line installers the README quotes
├─ .env                       your settings - optional; the setup screen writes them
├─ requirements.txt
│
├─ app/
│  ├─ config.py               every setting, read from .env
│  ├─ main.py                 FastAPI routes (chat, index, health, files)
│  │
│  ├─ core/                   the engine room
│  │  ├─ store.py             SQLite metadata + FTS5/BM25 + float32 vector file
│  │  ├─ ann.py               IVF-PQ approximate search, for corpora too big to scan
│  │  ├─ embed.py             local embedding & reranking backends
│  │  ├─ intent.py            is this a greeting, or a follow-up?
│  │  ├─ router.py            model per job, pacing, and provider failover
│  │  └─ llm.py               chat streaming across providers (Groq, Gemini, …)
│  │
│  ├─ ingestion/              documents in
│  │  ├─ indexer.py           incremental crawl, parallel parse, batch embed
│  │  ├─ progress.py          live throughput and the remaining-time estimate
│  │  ├─ chunk.py             structure-aware splitting
│  │  ├─ ocr.py               reads text out of pictures
│  │  └─ extractors/          one module per document family
│  │     ├─ documents.py         PDF, Word, RTF
│  │     ├─ presentations.py     PowerPoint
│  │     ├─ spreadsheets.py      Excel, CSV/TSV
│  │     ├─ textual.py           text, Markdown, JSON, YAML, HTML, XML
│  │     ├─ mail.py              .eml, Outlook .msg, ePub
│  │     ├─ images.py            standalone images
│  │     ├─ legacy.py            .doc/.xls/.ppt via installed Office
│  │     └─ common.py            shared helpers
│  │
│  ├─ retrieval/
│  │  ├─ expand.py            rewrites a question into the words the documents use
│  │  └─ search.py            hybrid search, fusion, reranking, context building
│  │
│  ├─ analysis/               finding what is wrong, not just answering what was asked
│  │  ├─ semantics.py        what the columns mean, and which direction is good
│  │  ├─ diagnose.py         data-quality scanning, in SQL, no model
│  │  ├─ cleaning.py         proposes fixes, applies them as a non-destructive view
│  │  ├─ performance.py      scale, leaders, laggards, concentration, momentum
│  │  ├─ trends.py           movement and significant outlier rates
│  │  └─ agent.py            the investigation loop, then the write-up
│  │
│  └─ tables/                 spreadsheets as SQL, for questions retrieval cannot answer
│     ├─ detect.py            is this sheet a table? what type is each column?
│     ├─ importer.py          streaming load, whole-table profiling, union views
│     ├─ catalog.py           what exists, and the schema description the model sees
│     ├─ sql.py               question to SQL, and the read-only execution guard
│     ├─ qa.py                runs the query and packages the result as context
│     ├─ charts.py            picks a chart from the result shape, and the guards
│     ├─ filters.py           injects a dashboard date range into a panel's SQL
│     └─ dashboard.py         saved panels, and running them live
│
├─ frontend/index.html        the whole UI, one self-contained file
├─ site/index.html            the public landing page people are sent to
├─ tests/                     self-test + sample document generator
└─ storage/                   the index (gitignored)
   ├─ index.db                files, chunks, full-text index
   ├─ vectors.f32             one float32 row per chunk
   ├─ tables.db               spreadsheets loaded as real tables
   ├─ dashboards.db           saved panels (survives a rebuild)
   ├─ semantics.yaml          what the data means (edit this)
   ├─ cleaning.yaml           the fixes in force (edit this)
   └─ models/                 downloaded ONNX models
```

## What it can read

| Type | Extensions | Notes |
|---|---|---|
| PDF | `.pdf` | per page, uses the bookmark tree for section titles |
| Word | `.docx` `.doc`\* | paragraphs, headings, tables, headers/footers |
| PowerPoint | `.pptx` `.ppt`\* | per slide, incl. tables, grouped shapes, speaker notes |
| Excel | `.xlsx` `.xlsm` `.xls` | per sheet, header-aware, 40 rows per chunk |
| Data | `.csv` `.tsv` `.json` `.jsonl` `.yaml` | header-aware records |
| Text | `.txt` `.md` `.rst` `.log` `.rtf` + source code | Markdown headings become sections |
| Web | `.html` `.htm` `.xml` `.epub` | scripts/styles stripped |
| Mail | `.eml` `.msg` | headers, body, attachment names |
| Images | `.png` `.jpg` `.tiff` `.bmp` `.webp` `.gif` | read with OCR |

\* Legacy `.doc/.ppt/.xls` are converted automatically **if** Microsoft Office and
`pywin32` are installed; otherwise they are reported as failed in the Files tab.

## Reading text inside pictures

OCR is on by default and covers four separate cases:

| Case | What happens |
|---|---|
| An image file (`invoice.png`) | read directly |
| A **scanned PDF** with no text layer | the page is rendered and read |
| A chart or screenshot **inside** a normal PDF page | that picture is read on its own |
| A picture pasted into **Word, PowerPoint or Excel** | read in place, so its text sits next to the surrounding paragraph or slide |

The engine is [RapidOCR](https://github.com/RapidAI/RapidOCR) — ONNX-based and
installed by plain `pip`, with no separate system installer. Tesseract is used
instead if you set `OCR_ENGINE=tesseract` and have it installed.

Because OCR is slow, it protects itself: images smaller than `OCR_IMAGE_MIN_PX`
are skipped as icons, repeated images (logos, watermarks) are read once, at most
`OCR_MAX_IMAGES_PER_FILE` images are read per document, results that look like
noise are discarded, and at most `OCR_WORKERS` engines run at a time. To turn it
off entirely, set `OCR_ENABLED=false` — indexing gets substantially faster.

## How it works

Two paths, chosen per question. Documents are embedded and searched; spreadsheets
that are real tables are loaded into SQLite and queried with SQL.

```
                    ┌─ spreadsheet, tabular ─► SQLite table + column profile
your folder ──► ────┤
                    └─ everything else ─► extract (+OCR) ──► chunk ──► embed (ONNX, local)

                    ┌─ "how many / total / top 5" ─► generated SQL ─► exact result ─┐
question ──► ───────┤                                                               ├─► answer
                    └─ "what does X say about Y" ─► hybrid search ─► passages ──────┘
                                                    (dense + BM25 → RRF → rerank)
```

* **Structured questions get exact answers** — counting, summing, ranking and
  filtering run as SQL over *every* row. Retrieval can only ever show the model the
  top handful of chunks, so it cannot answer those correctly at any `TOP_K`. The
  generated query is shown with the answer, so a number can always be checked.
* **Hybrid search** — dense vectors catch paraphrases ("what do we earn" → "revenue"),
  BM25 catches exact tokens (part numbers, names, error codes). RRF merges both.
* **Reranking** — a small cross-encoder rescores the top candidates against the
  question. This is the single biggest accuracy win; turn it off for more speed.
* **Incremental indexing** — files are fingerprinted by size+mtime, then by SHA-256.
  Unchanged files are skipped, edited files are re-read, deleted files drop out.
* **Grounded answers** — the model answers only from the retrieved passages, cites
  them as `[1]`, and says so when the answer isn't there.

## Spreadsheets

A sheet with a header row and more than a handful of rows is loaded into
`storage/tables.db` as a real table instead of being embedded. This is both far more
accurate and far faster: 200,000 rows import in about a minute, where embedding them
would take most of a day and still could not answer "how many" or "what is the total".

* **Types are inferred conservatively.** Amounts become `REAL`, dates become ISO text,
  and anything that looks like an identifier stays `TEXT` — a part number like
  `0041290` must never lose its leading zero.
* **Columns are profiled against the whole table**, so the model is told that
  `claim_status` is one of `'Approved'`, `'Rejected'`, … That is what lets it turn
  "how many were rejected" into a query that matches the real spelling in your data.
* **Sheets that share a schema are unioned** into a `*_all` view automatically, so a
  question about the whole year does not silently answer about one quarter. The
  `_source_file` column says which file a row came from.
* **Queries are read-only.** Generated SQL is parsed, rejected unless it is a single
  `SELECT`, run on a read-only connection under a timeout, and capped in rows.
* Sheets that are not tabular — pivot layouts, banners, a handful of rows — fall back
  to the normal text path, which does not care about structure.

The **Tables** tab lists what was loaded. `POST /api/sql` runs a `SELECT` yourself.

## Dashboards

Ask a question about your spreadsheets, check the query under the answer, then press
**Pin to dashboard**. The panel stores that SQL and re-runs it every time you open
the dashboard, so a panel is always live rather than a snapshot that quietly ages.

Dashboards live in `storage/dashboards.db`, separate from `tables.db`, because a
full rebuild resets the tables and a dashboard has to survive that.

* **The chart is chosen from the result, not by the model.** One number becomes a
  stat tile, a date axis a line, a category and a measure a bar (horizontal when the
  labels are long), two categories a grouped bar, two measures a scatter. Anything
  that cannot be drawn honestly stays a table. You can override the type per panel.
* **Charts are hand-drawn SVG** — no library, no CDN, so the page still works with no
  network and both themes are driven by the same tokens as the rest of the UI.
* **Guards that keep a panel readable and true:**
  * an identifier axis is never charted (29,206 serial numbers is a table, not a bar chart)
  * a categorical axis is capped at `CHART_MAX_CATEGORIES`, keeping the largest and
    saying how many were left out — there is deliberately no "Others" bucket, because
    summing the tail of an average would put a wrong number on a chart
  * long lines are thinned to `CHART_MAX_POINTS`
  * rows with a null date are dropped from a line and the count reported
  * a panel reading one file when a combined `*_all` view exists says so
  * saved SQL is re-validated on **every** run, not trusted because it was safe when pinned
  * each panel scrolls inside itself, so a wide result never widens the page

Panel names are suggested from the result itself ("Claims by Zone name", "Total
value"), so pinning does not start with an empty box.

**Viewing** — 1, 2 or 3 panels per row, remembered between sessions; ⤢ expands one
panel to fill the window (Esc to leave); the Manage sidebar closes automatically so
it does not eat a panel's width.

**Chart types** — the picker offers only what is honest for that result, and always
including `auto` and `table`. Switching is never a one-way door: the natural kind is
kept alongside your override, and the raw rows travel with every chart, so `table`
always has something to show and you can switch straight back. Beyond the basics
there is a **stacked** bar for multi-series results and an **area** line; single-series
bars get value labels when there is room.

**Date range** — one range in the toolbar applies to every panel that has a date
column, with quick presets. It is injected into each panel's SQL at run time rather
than stored with it, so clearing it restores exactly the query that was pinned. Only
a plain ISO date is ever accepted into the SQL, table names inside string literals
are left alone, and a panel with no date column says the filter does not apply to it.

**Export** — the whole dashboard as a standalone HTML page (charts included, opens
anywhere with no server) or as one CSV with every panel's rows and query. Each panel
also has its own **SQL** and **CSV** buttons.

## What the data means

The profile knows `claim_status` contains `'Rejected'`. It does not know that
rejection is *bad*, that a question about "breakdowns" means the rows whose
`claim_type` contains `BD`, or that a rate should go down while a total should go up.
Those are judgements about your business, and no amount of profiling recovers them
from the values.

So they are written down once, in `storage/semantics.yaml`, editable in the
**Meaning** tab. **Draft from data** writes a first version from the catalogue with
one model call — then correct it, because only you know which direction is good.

```yaml
outcomes:
  claim_status: {good: ['Approved'], bad: ['Rejected', 'QA Reject']}
metrics:
  rejection_rate: {sql: "...", direction: lower_is_better}
glossary:
  - {term: BD, means: Breakdown, also: ['Failure']}
```

It is used in four places: **trends** take the declared failure states instead of
guessing from a word list; the **analyst** gets a definition of "lagging"; **SQL**
computes a metric the same way every time; and the **glossary doubles as a synonym
table** for document search.

### When you add another file

The Meaning tab tells you when it is out of date — *"2 new tables not described
yet"* — and gives you three buttons:

| Button | What it does |
|---|---|
| **Add new tables** | describes only what is new, and splices it in |
| **Add document terms** | reads your PDFs and slides and adds their jargon to the glossary |
| **Redraft all** | rewrites everything, losing your edits — rarely what you want |

The merge is textual, not a parse-and-redump: your comments, ordering and hand-written
lines survive byte-for-byte, a table already described is never touched, and running
it twice changes nothing. Terms are **scoped per table** as well as shared, because
`PM` means preventive maintenance in a service export and project manager in an HR
one — a single global list would tell the query planner the wrong thing about
whichever dataset it was not written for.

Malformed YAML from the model is salvaged rather than rejected: a term like
`"9+1" topology` is dropped and reported, and the other twenty-four entries are kept.

## Finding things the documents word differently

A question rarely uses the words the files use. Someone asks about "machine
breakdowns"; the document only says "corrective maintenance", and BM25 has no shared
token to match on at all.

So the question is rewritten before searching — from the glossary (free and exact)
and from the fast model (a few alternative phrasings). Each variant is searched
separately and the ranked lists are fused, so a passage reachable through only one
phrasing still surfaces. Reranking happens against your **original** wording, because
that is the question that has to be answered.

| You ask | It also searches |
|---|---|
| how many breakdown jobs | `claim_type LIKE '%BD%'` — 25,821 rows |
| how do we deal with machine breakdowns | "BD handling procedure", "Breakdown Corrective Maintenance" |
| scheduled servicing policy | "Preventive Maintenance policy", "PM service guidelines" |

Set `EXPAND_QUERIES=0` to turn it off, or `EXPAND_WITH_MODEL=false` to keep only the
free glossary half.

## Answering from chosen files

The **All files** button above the composer limits a question to files you pick, and
every citation has an **only this file** link to narrow to what you can already see
the answer came from.

Scoping happens inside the index, before the shortlist is cut — not by filtering
results afterwards. That difference matters: filtering after the fact returns nothing
whenever the chosen file was not already winning, so a two-page note could never
compete with a 300-page report. Choosing only documents also skips the SQL path, so
the answer never comes from a spreadsheet you just excluded.

## Documents plus general knowledge

`ANSWER_MODE=blended` (the default) lets an answer use what the model knows as well
as what your files say — but the two are never mixed. Facts from your documents are
cited `[1]`; anything else appears under a **Beyond your documents** heading with no
citation, so you can always tell which is which. When nothing in your files matches,
it says so in one sentence and then answers from general knowledge rather than
refusing.

`ANSWER_MODE=grounded` restores the strict behaviour: documents only, and a refusal
when they do not contain the answer.

### Saying why, not just no

A question is rarely all-or-nothing, and a refusal that does not explain itself is
close to useless. Two rules:

* **Half a question still gets answered.** Asked for total claim value *and* average
  resolution time, it returns the totals and adds: *"Average claim resolution time
  cannot be computed: the table records sr_close_date but has no open date, so a
  duration cannot be calculated."* Declining the whole question because one clause is
  impossible leaves you with nothing.
* **A refusal names the column.** When the question really was about your data but a
  needed column is absent, that is said first — before any document search results —
  naming the column that is missing and the closest one that exists. "The data does
  not support this" is explicitly rejected as an answer, because it tells you nothing
  you did not already suspect.

A question about a document's wording sets no missing-column signal, so a good
document answer is never prefaced with an irrelevant note about SQL.

## Cleaning

Reporting that a column is 82% empty is where most tools stop. The **Clean** tab goes
further: **Examine data** proposes fixes, each measured before you accept it —

```
map_value      sd_name       10,678 rows   merge 'Gangpur Sales & Services Pvt. Ltd.'
                                           into 'Gangpur Sales and Services'
null_outlier   hour_meter       663 rows   set aside values above 389,355,000
null_sentinel  quantity     135,451 rows   treat 0 as missing, not as a measurement
```

Untick anything you disagree with — the `quantity = 0` rule above is a judgement call,
because zero parts on a labour-only line is a real value — then **Apply**.

**Nothing is destroyed.** Fixes become a SQL view named `<table>_clean`; the raw table
is never modified, so an unwanted rule is undone by unticking it and rebuilding, not by
re-importing 200,000 rows. Questions then prefer the cleaned view automatically, and
the raw one stays available for when you want the uncorrected values.

The effect is not cosmetic. On the claims data:

| | raw | cleaned |
|---|---|---|
| distinct dealers | 34 | 33 |
| average hour meter | 21,935,218,835 | 410,069 |

The rules — trim, blank-to-null, merge spellings, null a placeholder, set aside an
extreme value, set aside a future date — come from the profile and the diagnostics,
not from a model's impression of the data.

## Analysis

The **Analyse** button investigates the data instead of waiting to be asked. Two
halves, shown side by side:

**Measured** runs first and uses no model at all, so it is exact and instant. It is
in three parts, and the order is deliberate — a report whose evidence opens with
fourteen null-count findings writes about null counts:

*How the business is doing* — scale, direction of travel, and who is at each end of
every dimension, ranked on the metrics the semantic layer declares. The `direction`
field is what turns a ranking into a judgement: the top of a `higher_is_better` list
is the leaders, the top of a `lower_is_better` list is the problem. Concentration is
reported as a finding in its own right, because "the top 5 of 34 dealers carry 71%
of the value" is where the business actually is — and where the exposure sits.

*What changed* — month-over-month movement and segment outlier rates.

*Caveats* — last, and only what would change a conclusion:

* *data quality* - missing values, columns that never vary, numeric outliers judged
  against the 99th percentile (not the mean, which a zero-heavy column destroys),
  placeholder values, negatives, future dates, files whose date span disagrees with
  their name, near-duplicate category labels, and duplicated row keys;
* *movement* - each measure per month, and which segment moved between the last two
  **complete** months (the final month of an export is usually partial);
* *outlier rates* - where a failure state is more common than the overall rate,
  tested for significance rather than reported on a raw multiple, so a small segment
  has to be genuinely extreme before it counts.

Findings feed each other, but narrowly. A column is banned from totals only when it
is genuinely corrupt — an hour-meter reaching 154 million times its own 99th
percentile has a destroyed mean. Six extreme rows in two hundred thousand move a
total by a fraction of a percent and get a caveat instead: excluding revenue over
that would leave the analysis ranking segments by labour days, describing entirely
the wrong thing.

Every measured signal is **drawn as well as described** — a ranking becomes a bar
chart, a month series a line — with its query one click away.

**Investigation** then hands those measurements to the model, which follows up with
its own queries - is a spike one segment or all of them, one month or the whole
period - and writes the report. Every query it runs is shown as it runs, and goes
through the same read-only guard as everything else, so any claim in the report can
be traced to something that actually executed.

Every finding carries the SQL that produced it. `GET /api/diagnostics` returns the
measured half on its own, with no model call and no waiting.

### Two models, on purpose

| Job | Model | Why |
|---|---|---|
| each investigation step | `AGENT_MODEL` (Gemini flash-lite) | ~2s, and latency multiplies by the step count |
| the final report | `ANSWER_MODEL` (Groq gpt-oss-120b) | one long piece of prose, quality matters |

Gemini's newer *thinking* Flash models are a trap here: measured at 30-42s per call,
returning `finish_reason: length` with **zero** content tokens because the whole
budget went to hidden reasoning. `flash-lite` answers the same structured prompt in
about two seconds. Calls are paced client-side (`PROVIDER_RPM`) and fall over to the
other provider when one rate-limits, because a free tier will refuse mid-loop.

## Tuning

Everything is in `.env`.

**Answers are too slow** — response time is dominated by how much text the model
has to read:

| Setting | Faster | Better answers |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:1.5b-instruct` | `qwen2.5:7b-instruct`, `qwen2.5:14b` |
| `TOP_K` | `3` | `8` |
| `MAX_CONTEXT_CHARS` | `4000` | `12000` |
| `NUM_CTX` | `4096` | `8192` |
| `RERANK_ENABLED` | `false` | `true` |

`KEEP_ALIVE=30m` keeps the model in RAM between questions — without it every
question pays the model load time again.

**Indexing is too slow** — `OCR_ENABLED=false` is the big one; then lower
`OCR_MAX_IMAGES_PER_FILE`, or raise `INGEST_WORKERS` if the CPU is idle. Large
spreadsheets no longer contribute: they are imported as tables, not embedded.

**Spreadsheet answers are wrong or the query looks odd**

| Setting | Effect |
|---|---|
| `TABLE_ENUM_MAX` | how many distinct values per column the model is shown (raise it when a column's real values are being guessed) |
| `TABLE_MAX_ROWS` | rows a query may return to the model |
| `TABLE_MIN_ROWS` | below this a sheet stays on the text path |
| `TABLE_SQL_RETRIES` | re-prompts after a SQL error |
| `TABLES_ENABLED=false` | turn the whole path off |

The query is shown under every answer — if it filtered on the wrong column, the fix
is usually a clearer column name in the sheet, or a higher `TABLE_ENUM_MAX`.

**A question seems to hang**

Press **Esc** or the ■ button to stop it — a request is always cancellable, and the
status line shows the elapsed seconds once a step takes longer than three.

A spreadsheet question costs two model round trips: one to write the query, one to
phrase the answer. Only the second one streams, so a slow provider shows as a pause
before any text appears. The budgets that bound it:

| Setting | Bounds |
|---|---|
| `CONNECT_TIMEOUT` | reaching the provider at all |
| `READ_TIMEOUT` | the gap between streamed chunks |
| `TABLE_PLAN_TIMEOUT` | writing the query; on overrun the question falls back to document search |
| `REQUEST_TIMEOUT` | the whole answer |

If the provider is unreachable the question drops to document search immediately
rather than waiting. On a rate-limited free cloud tier, `TABLES_ENABLED=false`
halves the number of calls.

**Answers miss things**

* raise `TOP_K` and `MAX_CONTEXT_CHARS`
* set `NEIGHBOR_WINDOW=1` to also feed the chunks either side of each hit
* raise `CANDIDATES_DENSE` / `CANDIDATES_BM25` (cheap — only affects search)
* raise `RERANK_CANDIDATES` so a passage ranked low by fusion can still be promoted
  (it defaults to `4 × TOP_K`; lowering it is the fastest way to cut answer latency)
* for other languages: `EMBED_MODEL=intfloat/multilingual-e5-small`, then
  `index.bat --rebuild`

## How it holds up as the corpus grows

Exact search multiplies the whole vector matrix by the query. That is the right
answer while the matrix is small -- one BLAS call, and exact -- but its cost *is*
the size of the corpus:

| documents | chunks | float32 vectors | one exact query must read |
|---|---|---|---|
| 100,000 | 500,000 | 0.7 GB | 0.7 GB |
| 1,000,000 | 5,000,000 | 7.2 GB | 7.2 GB |
| 20,000,000 | 100,000,000 | 143 GB | 143 GB |

At the bottom row a single query reads 143 GB that does not fit in memory. Two
things have to change, and they are independent.

**Fewer vectors looked at.** Vectors are clustered once; a query is compared against
the handful of clusters whose centroid it resembles. The work per query stops
growing with the corpus and starts growing with the number of cells probed, which is
a number you choose (`ANN_NPROBE`).

**Smaller vectors.** Each 384-dimensional vector is cut into 32 slices and each
slice replaced by the index of the nearest of 256 learned prototypes: 32 bytes
instead of 1536. At 20M documents that is 4.5 GB instead of 143 GB -- the difference
between an index that lives in memory and one that cannot.

Both are in [`app/core/ann.py`](app/core/ann.py). Two details decide whether it
works at all, and both were found by measuring rather than by assuming:

* **The rotation.** Real embeddings are not spread evenly across their dimensions.
  Measured on bge-small, the top 10 of 384 directions carry 58% of the variance and
  the effective dimensionality is about 23. Sliced as they come, a few slices carry
  nearly all the information and cannot express it with 256 prototypes. Recall was
  0.54 that way. Rotating onto the principal axes first and dealing those axes out
  so each slice gets a comparable share of the variance took it to 0.89.
* **The re-score.** The approximate pass picks a shortlist; the shortlist is then
  scored exactly against the original float32 vectors -- a few hundred 1.5 KB reads.
  Approximation decides what to look at, arithmetic decides the order, and the score
  the rest of the pipeline sees is a true inner product. This is also why 32 slices
  are enough: 64 and 96 measured identical recall while costing two and three times
  the space.

Nothing is thrown away to make this work. The float32 vectors stay the source of
truth, the index is a derived artifact rebuilt at the end of an index run, and rows
added since the last build are searched exactly and merged in -- so a document is
searchable the moment it is indexed, and deleting `storage/vectors_ann/` falls back
to exact search with no loss.

### Measured

Two million chunks, vectors drawn to match the real embedding space, 100 queries,
on this machine (12 cores, 16 GB):

| | exact | approximate |
|---|---|---|
| per query | 267.4 ms | **5.3 ms** at 64 probes, 20.0 ms at 256 |
| read per query | 2.86 GB | 25 MB |
| index size | 2,930 MB | **92 MB** |
| recall@40 vs exact | 1.000 | 0.742 at 64 probes, 0.920 at 256 |
| finds the chunk the query came from | 1.000 | **0.980** at 64 probes |

The last row is the one that matters for answering questions and the row above it is
the one that looks alarming. They disagree because recall@40-against-exact punishes
reordering among near-ties: at these similarities the 40th result and the 200th are
separated by thousandths, so swapping them costs recall without costing the answer
anything. What the retrieval actually has to do -- surface the passage the question
is about -- it does 98% of the time while reading 0.9% as much data.

### The setting that would have gone wrong quietly

Accuracy tracks the *fraction* of the corpus a query examines, and the cell count
grows with the corpus -- so a fixed `ANN_NPROBE` examines a smaller share as the
index grows. Measured: 64 probes covered 8% of a 60k corpus and returned recall
0.98, but only 1.1% of a 2M corpus, where it returned 0.74. Left alone, accuracy
would have decayed with scale and nothing would have reported it.

So `ANN_NPROBE` is a floor and `ANN_SCAN_FRACTION` (default 2%) holds the share
constant. The trade this makes is explicit: a constant fraction means query cost
grows with the corpus. Constant cost is available -- set `ANN_SCAN_FRACTION=0` and
the floor is used alone -- but then recall falls away as the index grows. No setting
holds both, and the default prefers the answer staying right.

Below `ANN_MIN_VECTORS` (200k) none of this is used, because exact search is both
faster and exact at that size.

### Where the disk actually goes

At 100M chunks, the vectors were never the biggest thing on disk:

| | before | after |
|---|---|---|
| float32 vectors | 143 GB | 143 GB (source of truth, not read at query time) |
| approximate index | - | 4.5 GB |
| full-text index | 328 GB | **161 GB** |

The full-text half was storing every chunk's text twice -- once in `chunks`, once
again inside a standalone FTS5 table. It is now external-content, so FTS5 indexes
`chunks.text` in place instead of copying it, with `detail=none` because nothing
here issues a phrase or NEAR query. An index written before this change is detected
and keeps working on the old layout; a rebuild is what reclaims the space.

`python rag.py test` reports measured recall against exact search.

## Watching an index run

`index.bat` draws a live bar:

```
  [████████░░░░░░░░░░░░░░]  36.4%   ETA 2m 59s   64/78 files   220.6 KB/s   report_08.md
```

The remaining time is measured, not extrapolated from a file count. Files are not
comparable units of work — a 2 KB note and a 40 MB scanned contract both count as
one — and neither are bytes: on this pipeline a spreadsheet streams into SQL at
about a megabyte a second while text has to be chunked and embedded, which is
hundreds of times slower per byte. Size and cost are close to anti-correlated.

So the cost of each file *type* is measured as the run goes, in seconds per byte,
and the estimate is the outstanding work divided by the work the pipeline is
actually retiring per second over the last 45 seconds. Measured against a recorded
run, the old files-done/elapsed estimate was wrong by 68% of the run on average and
50% even in its second half; this one scores 28% and 10%.

It moves when conditions move, the way a download estimate does — a folder that
turns from notes into scanned contracts really does get slower, and the number
should say so. A file type nothing has been measured for yet is costed at the most
expensive type that has been, so the estimate starts long and comes down rather than
promising forty seconds and climbing to seven minutes.

The percentage is work-weighted, so it can sit behind the file count when the
expensive files are the ones left. `index.bat --quiet` turns the bar off, and it
turns itself off when output is redirected to a file.

## If it feels slow

Two things dominate, and both are fixed in the code rather than configurable:

**Dead IPv6 paths.** Google's API resolves to eight IPv6 addresses before any IPv4
one. On a network without working IPv6, each attempt hangs until it times out, so a
single model-list call took **81 seconds** where `curl` took 0.7 — and `/api/health`,
which the page polls and which probes every provider, took a minute and a half. That
is also why a configured provider can seem to be "missing": the dropdown is waiting
on a request that has not come back. IPv6 is probed once at startup and, when it does
not work, outbound requests are pinned to IPv4.

**Fresh connections.** A new HTTPS client per call pays TLS setup every time — 80s
against 0.4s for the second request on the same client. Each provider now keeps one
long-lived client, probes run in parallel, and they are warmed in the background at
startup, so the first page load is not waiting on three handshakes.

Measured on this machine: `/api/health` went from **105s to 2.6s cold and 0.01s warm**.

If it is still slow, the remaining time is the models themselves. A spreadsheet
question costs two calls (write the query, phrase the answer) and a document question
adds query expansion and reranking; `EXPAND_WITH_MODEL=false` and
`RERANK_ENABLED=false` each remove one step, at a cost in answer quality.

## Model providers

Several providers can be active at once. Every model they offer is listed in
**Manage → Settings**, grouped by provider, and any question can be sent to any of
them — no restart, no config change.

| Provider | Setting | Notes |
|---|---|---|
| **Groq** | `GROQ_API_KEY` | free tier, very fast, open-source models |
| **Gemini** | `GEMINI_API_KEY` | free tier, large context, speaks the OpenAI protocol |
| Anything OpenAI-compatible | `OPENAI_BASE_URL` + `OPENAI_API_KEY` | LM Studio, OpenRouter, Cerebras, vLLM — including a model running on your own machine |

Both keys are entered on the setup screen; nothing needs editing by hand. Models are
addressed as `provider::model`, e.g. `groq::openai/gpt-oss-120b`, and `LLM_MODEL` only
sets which one a new chat starts with.

> **What leaves the machine.** A question sent to a provider carries the passages the
> search selected, so those passages leave with it. Indexing, embedding, OCR, keyword
> search, reranking and every chart are computed locally and send nothing.
>
> For material that must not leave at all, point `OPENAI_BASE_URL` at a model served
> on your own machine — LM Studio and similar expose exactly this protocol on
> `localhost`, and the dropdown then treats it like any other provider.

Speech-to-text, text-to-speech and safety-classifier models that providers also
host are filtered out of the dropdown, since they cannot answer a question.

## The UI

One page, three panels behind the **Manage** button:

* **Index** — live progress, per-file log, incremental or full rebuild
* **Settings** — model, passages per answer, creativity, reranking
* **Files** — everything indexed, with failures and their reasons

Click any `[1]` in an answer to jump to the passage it came from; click the file
name to open the original document.

## Commands

| Command | Purpose |
|---|---|
| `python rag.py doctor` | check the install, report what is missing |
| `python rag.py index` | read `DATA_DIR` (`--rebuild` to start over) |
| `python rag.py serve` | start the web UI (`--open` to launch a browser) |
| `python rag.py test` | self-test on generated samples, including OCR |

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | SSE stream: `sources`, `token`, `done`, `error`, `notice`; `sources` and `done` carry `route` (`sql`, `vector` or `chat`). Pass `files: [...]` to restrict the answer to chosen files |
| `POST /api/search` | retrieval only, no LLM — useful for debugging |
| `POST /api/index` | start indexing (`{"rebuild": false}`) |
| `GET /api/index/status` | live progress |
| `GET /api/health` | LLM, OCR, index and table status |
| `GET /api/files` | indexed files |
| `GET /api/tables` | imported tables, their columns, and the schema the model is shown |
| `POST /api/sql` | run a read-only `SELECT` yourself (`{"sql": "..."}`) |
| `POST /api/analyze` | SSE: `evidence`, `step`, `report_token`, `done` - the full investigation |
| `GET /api/diagnostics` | the measured findings only, no model call |
| `GET/PUT /api/semantics` | read or save the semantic layer |
| `POST /api/semantics/draft` | rewrite the whole layer from the catalogue |
| `POST /api/semantics/update` | describe only the new tables, keeping your edits |
| `POST /api/semantics/document-terms` | add glossary terms read from your documents |
| `POST /api/cleaning/propose` | measure what is worth fixing in a table |
| `POST /api/cleaning/apply` | save the rules and rebuild the cleaned view |
| `GET /api/cleaning/impact` | what the fixes actually changed |
| `GET /api/dashboards` | list dashboards |
| `GET /api/dashboards/{id}` | run every panel and return chart-ready data |
| `POST /api/panels` | pin a panel (`{"title", "sql"}`) |
| `PATCH /api/panels/{id}` | change title, chart type, width or position |
| `GET /api/docs` | interactive API docs |

Deleting `storage/` and re-running `index.bat` rebuilds everything from your
documents. The index holds extracted text, so keep it as confidential as the
documents themselves.

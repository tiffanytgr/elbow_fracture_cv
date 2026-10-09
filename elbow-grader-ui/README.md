# Elbow Grader UI — Next.js 14

Next.js 14 (App Router) frontend for the KKH paediatric elbow fracture grading pipeline.

## Prerequisites

- Node.js 18+
- Python 3.10+ with the `elbow-grader` package installed (`pip install -e .` from `KKH_Elbow/`)

## Development

**1. Start the Python / FastAPI backend** (from `KKH_Elbow/`):

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

**2. Start the Next.js dev server** (from `KKH_Elbow/elbow-grader-ui/`):

```bash
npm install
npm run dev
```

Open <http://localhost:3000>.

## Production build

```bash
npm run build
npm start
```

## Environment

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | FastAPI service URL |
| `STUDY_LOG_PATH` | `logs/study-records.jsonl` | Where reader-study records are written on the local device (absolute path recommended) |

Set `BACKEND_URL` in `.env.local` to point at a remote backend.

## Reader study: timer and answer capture

The page runs as an AI-assisted reader-study tool. The reader enters their name
in the **Study Session** bar, records a grade **before** the AI is shown and
**Submit**s it — this captures the control timing, `pre_grade_time_seconds`, the
time from when the X-ray appeared (case load) to that submission — then rates
confidence and locks the pre-AI read, runs the AI analysis, reviews it (Grad-CAM,
Baumann, AHL, bone profile), and records a **post-AI** read in two steps (below).
The Analyse button stays disabled until the pre-AI read is locked, and the AI
result is only revealed afterwards.

The decision read is submitted in two steps:

1. **Submit Gartland grade** — the reader picks a Gartland grade and submits it.
   This records two timings from the stopwatch (paused time excluded):
   `grade_time_seconds`, the time from **X-ray upload (case load) to this
   submission**, and `decision_time_seconds`, the slice of that **after the AI
   results appeared** (re-running the AI before submitting restarts the decision
   clock).
2. **Submit assessment** — the reader then rates confidence (1–5) and answers
   the follow-up questions (currently an optional comments box), and submits.
   This saves the case.

**Control arm (no AI).** After the pre-AI grade is submitted and confidence
rated, the reader can instead click **Submit without AI (control)** to save the
case straight from the pre-AI read, without locking, running the analysis, or
the post-AI steps. Control records store only the pre-AI read (`pre_grade`,
`pre_confidence`, `pre_grade_time_seconds`) with the post-AI and `ai_*` fields
null, and are tagged `arm: "control"` (AI-assisted records are `arm: "ai"`).

**Cases.** Uploading new X-ray images (or selecting another example case)
starts a new case: the stopwatch, answers, and AI result reset. Adding the
missing second view to an unsaved case (e.g. the LAT after the AP) completes
the current case instead; rotating or removing an image never starts a new one.

**Case identifier.** Before grading, the reader enters a **Case ID** (required;
selecting an example case pre-fills it with the example's ID). It is logged as
`case_id` in both the study log and the backend's `predictions.log`, and the
field clears after each submitted case. The X-ray file paths are logged
separately as `ap_path` / `lat_path`: example cases use their bundled path (e.g.
`/demo/grade-2a/a145-ap.png`); for uploads, browsers do not expose a file's
absolute path, so it is the folder-relative path when available and otherwise
the file name.

A per-case **stopwatch** auto-starts when a case loads and has a **Pause**
button for stepping away. **Submit assessment** appends one
record to a JSON Lines file **on the machine running the app** (written server
side via `POST /api/study-log`), so it works for a locally-hosted deployment on
a remote laptop. Grades use the AI label space (`Normal`, `Grade 1`,
`Grade 2a`, `Grade 2b`, `Grade 3`) with a 1–5 confidence scale.

Each line is a JSON object, e.g.:

```json
{"reviewer":"TT","case_id":"a145","ap_path":"/demo/grade-2a/a145-ap.png","lat_path":"/demo/grade-2a/a145-lat.png","input_mode":"demo","arm":"ai","pre_grade":"Grade 2a","pre_confidence":3,"pre_grade_time_seconds":18.9,"post_grade":"Grade 2b","post_confidence":4,"ai_gartland_grade":"Grade 2b","ai_cnn_grade":"Grade 2b","ai_geometric_grade":"Grade 2b","ai_confidence":0.81,"ai_processing_time_seconds":7.3,"notes":null,"decision_started_at":"…","grade_submitted_at":"…","grade_time_seconds":41.7,"decision_time_seconds":12.4,"elapsed_seconds":63.2,"elapsed_hms":"00:01:03","started_at":"…","ended_at":"…","logged_at":"…"}
```

`ai_gartland_grade` is the pipeline's final Gartland grade; `ai_cnn_grade` and
`ai_geometric_grade` are the CNN and geometric (AHL) grades it was derived
from. `ai_processing_time_seconds` is the backend model run time for the AI
result the reader saw. After **Submit assessment**, the panel shows the full
path of the study log and of the backend's `predictions.log`. The default location is
`elbow-grader-ui/logs/study-records.jsonl` (git-ignored); override with
`STUDY_LOG_PATH`.

The backend's `logs/predictions.log` (one line per AI run) uses the same
`case_id` (falling back to the file paths if none was sent) and also records `gartland_grade`, `cnn_grade` and
`geometric_grade`.

`GET /api/study-log` reads the log back:

| Query | Returns |
|---|---|
| `?limit=N` | N most recent records as JSON (shown under "Recently logged") |
| `?format=csv` | Every record as a CSV download |
| `?summary=1` | Grouped timing stats (count, mean/median/min/max/total seconds, mean/median pre-AI grade time, mean/median grade time, mean/median decision time) as JSON |
| `?summary=1&format=csv` | The same grouped stats as a CSV download |
| `&group_by=…` | Grouping field for the summary — `reviewer` (default), `arm`, `pre_grade`, `post_grade`, or `input_mode` |

The summary always includes an `ALL` row alongside the per-group rows. The UI
exposes **Download log (CSV)**, **Summary (CSV)**, and an inline summary table.

## Architecture

```
elbow-grader-ui/
├── app/
│   ├── layout.tsx           Root layout
│   ├── page.tsx             Main upload + results page (client component)
│   └── api/predict/route.ts Next.js API route — proxies to FastAPI
├── components/
│   ├── Sidebar.tsx          Model status + config controls
│   ├── FileUploader.tsx     Drag-and-drop image upload
│   ├── ResultsBanner.tsx    Grade result banner + CNN/geometric comparison
│   └── tabs/
│       ├── SummaryTab.tsx
│       ├── PerExperimentTab.tsx  Prob bars + Grad-CAM overlays
│       ├── BaumannTab.tsx
│       ├── GeometricTab.tsx      SAM2 overlay + AHL bisection
│       ├── CorticalWidthTab.tsx
│       └── AuditTab.tsx          JSON export
├── components/ui/           shadcn/ui base components (Radix-based)
└── lib/
    ├── types.ts             TypeScript types mirroring FastAPI schema
    └── utils.ts             cn() helper
```

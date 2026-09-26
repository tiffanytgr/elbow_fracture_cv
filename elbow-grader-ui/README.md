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
| `BACKEND_URL` | `http://localhost:8000` | FastAPI service URL (only needed for the AI arm) |
| `STUDY_LOG_PATH` | `logs/study-records.jsonl` | Where reader-study records are written on the local device (absolute path recommended) |

Set `BACKEND_URL` in `.env.local` to point at a remote backend.

## Reader study: arms, timer, and answer capture

The page runs as a reader-study tool with two arms, chosen per case in the
**Study Session** bar (reviewer name + arm). The arm is **locked once a case is
loaded** and unlocks after the case is submitted, so it can't be flipped mid-case.

- **AI-assisted arm** — the reader records a grade + confidence **before** the AI
  is shown, locks it, runs the AI analysis, reviews it (Grad-CAM, Baumann, AHL,
  bone profile), then records a **post-AI** read in two steps (below). The
  Analyse button stays disabled until the pre-AI read is locked, and the AI
  result is only revealed afterwards.
- **Control arm** — no AI at all: the Sidebar, Analyse step, and results are
  hidden, so the reader sees only the AP/LAT images and records **one** read
  per case in the same two steps.

The decision read is submitted in two steps:

1. **Submit grade** — the reader picks a Gartland grade and submits it. This
   stops the **decision time** clock, which measures how long the clinician
   took to decide on a grade **after the AI results appeared** (AI arm) or
   after the case loaded (control arm). It uses the stopwatch, so paused time
   is excluded; re-running the AI before submitting restarts the clock.
2. **Submit assessment** — the reader then rates confidence (1–5) and answers
   the follow-up questions (currently an optional comments box), and submits.
   This saves the case.

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

Both arms share a per-case **stopwatch** that auto-starts when a case loads and
has a **Pause** button for stepping away. **Submit assessment** appends one
record to a JSON Lines file **on the machine running the app** (written server
side via `POST /api/study-log`), so it works for a locally-hosted deployment on
a remote laptop. Grades use the AI label space (`Normal`, `Grade 1`,
`Grade 2a`, `Grade 2b`, `Grade 3`) with a 1–5 confidence scale.

Each line is a JSON object, e.g.:

```json
{"reviewer":"TT","mode":"ai","case_id":"a145","ap_path":"/demo/grade-2a/a145-ap.png","lat_path":"/demo/grade-2a/a145-lat.png","input_mode":"demo","pre_grade":"Grade 2a","pre_confidence":3,"post_grade":"Grade 2b","post_confidence":4,"ai_gartland_grade":"Grade 2b","ai_cnn_grade":"Grade 2b","ai_geometric_grade":"Grade 2b","ai_confidence":0.81,"notes":null,"decision_started_at":"…","grade_submitted_at":"…","decision_time_seconds":12.4,"elapsed_seconds":63.2,"elapsed_hms":"00:01:03","started_at":"…","ended_at":"…","logged_at":"…"}
```

`ai_gartland_grade` is the pipeline's final Gartland grade; `ai_cnn_grade` and
`ai_geometric_grade` are the CNN and geometric (AHL) grades it was derived
from. In control records `post_grade`, `post_confidence` and the `ai_*` fields
are `null`. The default location is
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
| `?summary=1` | Grouped timing stats (count, mean/median/min/max/total seconds, mean/median decision time) as JSON |
| `?summary=1&format=csv` | The same grouped stats as a CSV download |
| `&group_by=…` | Grouping field for the summary — `mode` (default), `reviewer`, `pre_grade`, or `input_mode` |

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

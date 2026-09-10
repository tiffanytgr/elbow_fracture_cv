# Elbow Control UI — Next.js 14

Clinician-facing **control app** for the KKH paediatric elbow fracture study.
It presents the same X-ray cases as the AI grader but provides **no AI
assistance** — the clinician reads the images and records their own Gartland
classification. It runs fully client + Next.js server side, with **no Python
backend**, so it can be hosted on a local laptop.

Its purpose is the control arm: measuring unaided clinician grade, confidence,
and time per case for comparison against the AI-assisted `elbow-grader-ui`.

## Prerequisites

- Node.js 18+

## Development

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

## What it captures

For each case the clinician records a Gartland **grade** (Normal, I, IIA, IIB,
III, IV), a 1–5 **confidence** rating, and optional **notes**. A per-case timer
auto-starts when a case loads and stops on save. **Save assessment** appends one
record to a JSON Lines file **on the machine running the app** (written server
side via `POST /api/case-log`).

## Assessment log

| Variable | Default | Description |
|---|---|---|
| `CONTROL_LOG_PATH` | `logs/assessments.jsonl` | Where assessments are written on the local device (absolute path recommended) |

Each line is a JSON object, e.g.:

```json
{"case_id":"a145","clinician":"AK","grade":"IIA","confidence":4,"notes":null,"input_mode":"demo","elapsed_seconds":51.2,"elapsed_hms":"00:00:51","started_at":"…","ended_at":"…","logged_at":"…"}
```

`GET /api/case-log` reads it back:

| Query | Returns |
|---|---|
| `?limit=N` | N most recent records as JSON (shown under "Recently logged") |
| `?format=csv` | Every record as a CSV download |
| `?summary=1` | Grouped stats (count, mean confidence, mean/median/min/max/total seconds) as JSON |
| `?summary=1&format=csv` | The same grouped stats as a CSV download |
| `&group_by=…` | Grouping field for the summary — `grade` (default), `clinician`, or `input_mode` |

The summary always includes an `ALL` row alongside the per-group rows. The UI
exposes **Download log (CSV)**, **Summary (CSV)**, and an inline summary table.

## Architecture

```
elbow-control-ui/
├── app/
│   ├── layout.tsx            Root layout
│   ├── page.tsx              Choose case → review & grade (client component)
│   └── api/case-log/route.ts Local JSONL logging + CSV/summary export
├── components/
│   ├── DemoCaseSelector.tsx  Example-case picker (shared image set)
│   ├── FileUploader.tsx      Drag-and-drop upload for own images
│   ├── CaseImageBoard.tsx    Read-only AP/LAT viewer with expand
│   └── AssessmentPanel.tsx   Grade + confidence + notes + timer + save/export
├── components/ui/            shadcn/ui base components
└── lib/
    ├── types.ts              Gartland grades, confidence levels, record shape
    └── utils.ts              cn() helper
```

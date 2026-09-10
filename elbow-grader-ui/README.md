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
| `TIMER_LOG_PATH` | `logs/case-timings.jsonl` | Where case review timings are written on the local device (absolute path recommended) |

Set `BACKEND_URL` in `.env.local` to point at a remote backend.

## Case review timer

The main page includes a **Case Review Timer** that measures how long a reviewer
spends on each case. The timer auto-starts and resets whenever a new case is
loaded (demo selection or upload), and supports pause/resume/reset. Clicking
**Save time to log** appends one record to a JSON Lines file **on the machine
running the app** (the Next.js server writes it via `POST /api/timer-log`), so
it works for a locally-hosted deployment on a remote laptop.

Each line is a JSON object, e.g.:

```json
{"case_id":"a145","reviewer":"TT","notes":null,"input_mode":"demo","final_grade":"2a","confidence":0.83,"elapsed_seconds":42.6,"elapsed_hms":"00:00:43","started_at":"2026-09-10T…","ended_at":"2026-09-10T…","logged_at":"2026-09-10T…"}
```

The default location is `elbow-grader-ui/logs/case-timings.jsonl` (git-ignored).
Override it with `TIMER_LOG_PATH`. `GET /api/timer-log?limit=N` returns the most
recent records, which the UI shows under "Recently logged".

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

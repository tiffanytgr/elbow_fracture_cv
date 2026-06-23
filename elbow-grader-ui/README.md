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

Set `BACKEND_URL` in `.env.local` to point at a remote backend.

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

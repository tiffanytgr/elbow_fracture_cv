# Elbow Grader

Gartland classification pipeline for paediatric supracondylar elbow fractures. Processes AP and LAT elbow X-rays to assign fracture severity grades (Normal, Grade 1, Grade 2a, Grade 2b, Grade 3) using a combination of deep learning classifiers and geometric analysis.

## Features

- **Cascading CNN classification** — 4 sequential experiments (Normal/Fractured → Grade 3 → Grade 1/2 → Grade 2a/2b)
- **Geometric grading** — SAM2 bone segmentation + anterior humeral line (AHL) bisection + cortical width analysis
- **Baumann angle measurement** — Automated angle computation with normal-range flagging
- **OOD detection** — DRUE (Dual Reconstruction Uncertainty Estimation) scoring per experiment
- **Grad-CAM explainability** — Visual attention overlays for each classifier
- **CNN vs Geometric comparison** — Flags discordant results for clinical review

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn |
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS, shadcn/ui |
| ML | PyTorch ≥ 2.0, TorchVision ≥ 0.15 |
| Segmentation | SAM2 (sam2_hiera_large.pt) |
| Image Processing | OpenCV, scikit-image, Pillow |

## Project Structure

```
KKH_Elbow/
├── elbow_grader/         # Python package (ElbowGrader, GraderConfig, GraderResult)
├── pipeline/             # Internal ML pipeline (config, inference, alignment, SAM2)
├── backend/              # FastAPI server
├── elbow-grader-ui/      # Next.js frontend
├── checkpoints/          # Model weights (sam2_hiera_large.pt)
├── experiments/          # Training notebooks and evaluation scripts
└── outputs/              # Experiment results
```

## Model Weights

Model weights are not included in this repository due to file size. Download them before running the app.

**[Download all weights (Google Drive)](https://drive.google.com/your-link-here)**

After downloading, place the files as follows:

```
checkpoints/
└── sam2_hiera_large.pt

experiments/
└── checkpoints/
    ├── exp1/best_model_for_analysis.pth
    ├── exp2/best_model_for_analysis.pth
    ├── exp3/best_model_for_analysis.pth
    ├── exp4/best_model_for_analysis.pth
    └── cap_regressor/best_model.pth
```

DRUE OOD decoder weights (optional — required for OOD scoring):

```
experiments/
├── drue_outputs_exp1/drue_decoders.pth
├── drue_outputs_exp2/drue_decoders.pth
├── drue_outputs_exp3/drue_decoders.pth
└── drue_outputs_exp4/drue_decoders.pth
```

`sam2_hiera_large.pt` can also be downloaded directly from the [Meta SAM2 releases page](https://github.com/facebookresearch/sam2/releases).

## Prerequisites

- Python ≥ 3.10 with CUDA-capable PyTorch
- Node.js ≥ 18
- Model weights downloaded and placed as described above

## Setup & Running

### Backend

From `KKH_Elbow/`:

```bash
pip install -e .
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Backend serves on **http://localhost:8000**.

### Frontend

From `KKH_Elbow/elbow-grader-ui/`:

```bash
npm install
npm run dev
```

Frontend serves on **http://localhost:3000**.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | `http://localhost:8000` | FastAPI backend URL (set in `elbow-grader-ui/.env.local`) |

### CLI

After `pip install -e .`:

```bash
elbow-grader --ap path/to/ap.png --lat path/to/lat.png
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server liveness check + checkpoint status |
| `POST` | `/predict` | Run full grading pipeline |

### `POST /predict`

**Parameters** (multipart form):

| Field | Type | Description |
|-------|------|-------------|
| `ap_file` | file | AP X-ray image (optional) |
| `lat_file` | file | LAT X-ray image (optional) |
| `confidence_threshold` | float | Minimum probability to accept prediction (0.50–0.95) |
| `run_full_lat_alignment` | bool | Run 6-step LAT alignment for raw images |
| `run_sam2` | bool | Run SAM2 segmentation + geometric grading |

**Response** includes: `final_grade`, `cnn_grade`, `geometric_grade`, `discordant`, `confidence`, `is_ood`, `baumann_angle`, per-experiment results, and base64-encoded plots (Grad-CAM, geometric overlay, cortical width, Baumann).

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.5 | Minimum winning-class probability [0.5, 1.0) |
| `run_full_lat_alignment` | True | Run 6-step LAT alignment |
| `run_sam2` | True | Run geometric grading track |
| `device` | "auto" | Compute device: "auto", "cpu", or "cuda" |

## Grading Methodology

1. **AP cascade**: Normal/Fractured → Grade 3 vs Grade 1/2
2. **LAT cascade** (if not Grade 3): Grade 1 vs Grade 2 → Grade 2a vs 2b
3. **Geometric track**: SAM2 segmentation → AHL bisection → cortical width profiling
4. **Baumann angle**: Shaft axis + physis line → angle computation → normal-range check
5. **Grade fusion**: CNN + geometric results combined; discordance flagged for review

# Industrial Waste Classification — WaRP-C

[![API smoke test](https://github.com/SiddiqueSahb/Industrial-waste-ai/actions/workflows/api-smoke-test.yml/badge.svg)](https://github.com/SiddiqueSahb/Industrial-waste-ai/actions/workflows/api-smoke-test.yml)

End-to-end computer-vision system that classifies industrial waste images into
the 28 fine-grained **WaRP-C** (Waste Recycling Plant) categories, with a
YOLOv11 detection extension on WaRP-D. Five modern architectures are trained
and compared, and the best models are served through a production-style stack:
**MLflow experiment tracking → evaluation reports → versioned GCS model
registry → FastAPI inference service → Docker → Cloud Run**, with a CI smoke
test on every push.

## Architecture

## Project Output

### Industrial Waste Classification

<p align="center">
</p><img width="1468" height="836" alt="Screenshot 2026-06-26 at 8 47 18 PM" src="https://github.com/user-attachments/assets/3a24b322-6ca7-4258-ac41-d02919a69a49" />
<img width="1457" height="834" alt="Screenshot 2026-06-26 at 8 48 30 PM" src="https://github.com/user-attachments/assets/87c52f20-7424-42a8-8226-a88cabff8b47" />
<img width="1465" height="830" alt="Screenshot 2026-06-26 at 8 54 26 PM" src="https://github.com/user-attachments/assets/055dc460-6a0f-467b-8136-e428f31c77d0" />
<img width="1468" height="830" alt="Screenshot 2026-06-26 at 8 55 26 PM" src="https://github.com/user-attachments/assets/f77a95ca-4cfb-4825-96b3-745bc36870b3" />
<img width="1470" height="837" alt="Screenshot 2026-06-26 at 8 56 29 PM" src="https://github.com/user-attachments/assets/d53bb4b1-1ec0-4f16-a126-bac4c9c894d0" />


---

```
                        ┌────────────────────────────────────────────┐
                        │                 TRAINING                   │
 WaRP-C dataset ──────► │  src/classification/train.py               │
 (8.8k train /          │  ConvNeXt · ConvNeXtV2 · ViT-B/16          │
  1.5k test,            │  Swin-V2-T · MaxViT-T                      │
  28 classes)           │        │                                   │
                        │        ├──► MLflow (params, per-epoch      │
                        │        │     metrics, model artifacts)     │
                        │        └──► results/metrics/*.pkl|.npz     │
                        └────────┬───────────────────────────────────┘
                                 │
                ┌────────────────┴──────────────────┐
                ▼                                   ▼
   ┌───────────────────────────┐      ┌───────────────────────────────┐
   │        EVALUATION         │      │       MODEL REGISTRY          │
   │ src/classification/       │      │ gs://bucket/models/           │
   │   evaluate.py             │      │   <name>/<version>/           │
   │ confusion matrix,         │      │ model.pth + metadata.json     │
   │ per-class P/R/F1,         │      │ + summary.json + config       │
   │ cross-model table         │      └──────────────┬────────────────┘
   └───────────────────────────┘                     │ MODEL_URI
                                                     ▼
   ┌───────────────────────────┐      ┌───────────────────────────────┐
   │      STREAMLIT DEMO       │      │      INFERENCE SERVICE        │
   │ app/streamlit_app.py      │      │ api/main.py (FastAPI)         │
   │ live predictions,         │      │ POST /predict, GET /health    │
   │ error analysis dashboards │      │ Docker → Cloud Run            │
   └───────────────────────────┘      │ CI smoke test on every push   │
                                      └───────────────────────────────┘
```

Both front-ends share the same model factory ([src/classification/models.py](src/classification/models.py)),
transforms, and config ([configs/classification.yaml](configs/classification.yaml)),
so demo and production predictions always agree.

## Results

Test-set performance on WaRP-C (1,583 images, 28 classes), computed by
`python -m src.classification.evaluate --all`:

| Model | Top-1 | Top-5 | Macro Precision | Macro Recall | Macro F1 | Macro AUC (OvR) |
|:--|--:|--:|--:|--:|--:|--:|
| ViT-B/16 | **0.7537** | **0.9678** | **0.7748** | 0.7169 | 0.7262 | 0.9794 |
| Swin-V2-T | 0.7453 | 0.9652 | 0.7643 | **0.7244** | **0.7345** | **0.9813** |
| ConvNeXt-Tiny | 0.7002 | 0.9587 | 0.7005 | 0.7085 | 0.6942 | 0.9802 |
| ConvNeXtV2-Tiny | — | — | — | — | — | — |
| MaxViT-T | — | — | — | — | — | — |

*ConvNeXtV2-Tiny and MaxViT-T were trained in the Colab experiments
([notebooks/](notebooks/)); their prediction artifacts have not been exported
to this repo yet, so they are excluded from the verified table. Full per-class
precision/recall/F1, confusion matrices, and the machine-readable comparison
live in [results/evaluation/](results/evaluation/).*

The detection extension trains YOLOv11-m on WaRP-D (5 super-classes) — see
[notebooks/DetectionWaRP.ipynb](notebooks/DetectionWaRP.ipynb).

## Quickstart (local)

```bash
git clone https://github.com/SiddiqueSahb/Industrial-waste-ai.git
cd Industrial-waste-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Place the WaRP-C crops under `data/raw/warp_c/Warp-C/{train_crops,test_crops}/`
and trained checkpoints under `models/classification/` (or pull them from the
[model registry](docs/model_registry.md)).

**Streamlit demo** — live predictions plus per-model and cross-model analysis:

```bash
streamlit run app/streamlit_app.py
```

**Inference API**:

```bash
uvicorn api.main:app --port 8000
# then:
curl -X POST localhost:8000/predict -F "file=@path/to/image.jpg"
```

```json
{
  "model_name": "convnext_tiny",
  "predicted_class": "milk-cardboard",
  "confidence": 0.4815,
  "top_k": [{"class_name": "milk-cardboard", "confidence": 0.4815}, ...],
  "latency_ms": 33.8
}
```

Configure with env vars: `MODEL_NAME` (any model in the config), `WEIGHTS_PATH`,
`MODEL_URI` (auto-download from GCS/HTTP), `DEVICE`, `TOP_K`.

## Training with MLflow tracking

```bash
python -m src.classification.train --model convnext_tiny --epochs 30
mlflow ui   # inspect runs at http://localhost:5000
```

Every run logs hyperparameters, per-epoch train/val loss & accuracy, learning
rate, epoch time, final test metrics, and the best checkpoint as an artifact.
A quick pipeline check without touching real artifacts:

```bash
python -m src.classification.train --model convnext_tiny --epochs 1 --limit-per-class 4
```

## Evaluation

```bash
python -m src.classification.evaluate --model vit_b_16   # one model
python -m src.classification.evaluate --all              # comparison table too
```

Produces per model: `summary.json`, `per_class_metrics.csv` (precision /
recall / F1 / support for all 28 classes), `confusion_matrix.png`,
`per_class_f1.png`, and a text classification report — plus
`results/evaluation/model_comparison.{csv,md}` across models.

## Docker

```bash
docker build -t warp-api .
docker run -p 8080:8080 warp-api                      # weights baked at build time
# or pull weights from the registry at startup:
docker run -p 8080:8080 -e MODEL_URI=gs://BUCKET/models/convnext_tiny/v1/model.pth warp-api
curl localhost:8080/health
```

CPU-only image (small torch wheels), non-root user, Cloud Run-compatible
(`PORT` env respected).

## CI

[.github/workflows/api-smoke-test.yml](.github/workflows/api-smoke-test.yml)
runs on every push:

1. **api-smoke** — boots the service on the runner and validates `/health`,
   `/predict` (schema, sorted top-k, confidence bounds) and rejection of
   non-image uploads, via [tests/smoke_test.py](tests/smoke_test.py).
2. **docker-smoke** — builds the Docker image and runs the same smoke test
   against the container.

## MLOps docs

- [docs/model_registry.md](docs/model_registry.md) — versioned model artifacts
  on GCS (one immutable folder per model version, with metrics + config snapshot)
- [docs/deployment.md](docs/deployment.md) — step-by-step Cloud Run deployment
  with Cloud Build, service accounts, traffic splitting and rollback

## Project structure

```
├── api/                    # FastAPI inference service (+ its slim requirements)
├── app/streamlit_app.py    # Streamlit demo & analysis dashboard
├── configs/                # single source of truth for classes/models/transforms
├── docs/                   # model registry + Cloud Run deployment guides
├── notebooks/              # Colab experiments (classification + YOLOv11 detection)
├── scripts/                # training helpers, GCS registry push
├── src/
│   ├── classification/     # dataset, models, train (MLflow), evaluate, inference
│   ├── analysis/           # misclassification & comparison tooling
│   └── config/             # YAML → typed config loader
├── tests/smoke_test.py     # API smoke test (local, CI, or against Cloud Run)
├── Dockerfile
└── .github/workflows/api-smoke-test.yml
```

## Dataset

[WaRP — Waste Recycling Plant dataset](https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset):
industrial conveyor imagery. WaRP-C provides cropped objects across 28
recyclable categories (bottles, cans, cardboard, detergent, glass); classes are
highly imbalanced, which is why evaluation emphasises macro F1 and per-class
behaviour over plain accuracy.

## Contributors

| Model / Workstream | Contributor |
|---|---|
| Vision Transformer (ViT-B/16), per-class & confusion analysis | Mohammad Asim Siddique |
| ConvNeXt V1, occlusion-sensitivity explainability | Mohammad Arshad Siddique |
| ConvNeXt V2, occlusion-sensitivity explainability | Unmesh Pawar |
| Swin Transformer (Swin-V2-T) | Suhaib Ahmed Khan |
| MaxViT-T | Mohd Yasir Ansari |
| YOLOv11-m detection extension | Arshad · Unmesh · Asim |

Originally built for EEEM068 Applied Machine Learning (University of Surrey),
then extended with the production/MLOps stack.

## License

MIT — see [LICENSE](LICENSE).
This project is developed for academic, research, and educational purposes.

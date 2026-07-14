# Deploying the inference service to Google Cloud Run

The FastAPI service ships as a CPU-only Docker image. Weights are **not** baked
into the deployed image — Cloud Run pulls them from the
[GCS model registry](model_registry.md) at startup via `MODEL_URI`, so you can
roll a new model version by redeploying with a different URI, without rebuilding.

All commands below are for you to run yourself. Placeholders:

| Placeholder | Meaning |
|---|---|
| `YOUR_PROJECT_ID` | GCP project id |
| `YOUR_BUCKET_NAME` | registry bucket from [model_registry.md](model_registry.md) |
| `europe-west2` | region (London) — change if you prefer |

## 0. Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated
- Model pushed to the registry (e.g. `models/convnext_tiny/v1/model.pth`)

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable the required services (one-time)
gcloud services enable run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com
```

## 1. Create an Artifact Registry repository (one-time)

```bash
gcloud artifacts repositories create warp-images \
    --repository-format=docker \
    --location=europe-west2 \
    --description="WaRP-C inference service images"
```

## 2. Build and push the image with Cloud Build

No local Docker needed — Cloud Build builds from your source tree:

```bash
gcloud builds submit \
    --tag europe-west2-docker.pkg.dev/YOUR_PROJECT_ID/warp-images/warp-api:v1 \
    .
```

(`.dockerignore` keeps the upload small — data, venv, notebooks and results
are excluded. Local weights *are* uploaded if present; harmless, but you can
move them aside first if you want the image weight-free.)

## 3. Create a service account that can read the model bucket (one-time)

```bash
gcloud iam service-accounts create warp-api-runner \
    --display-name="WaRP API Cloud Run runtime"

gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET_NAME \
    --member="serviceAccount:warp-api-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
```

## 4. Deploy to Cloud Run

```bash
gcloud run deploy warp-api \
    --image=europe-west2-docker.pkg.dev/YOUR_PROJECT_ID/warp-images/warp-api:v1 \
    --region=europe-west2 \
    --service-account=warp-api-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --set-env-vars=MODEL_NAME=convnext_tiny,MODEL_URI=gs://YOUR_BUCKET_NAME/models/convnext_tiny/v1/model.pth,DEVICE=cpu \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=3 \
    --allow-unauthenticated
```

Notes:

- `--memory=2Gi` — torch + ConvNeXt weights need well over the 512Mi default.
- `--min-instances=0` scales to zero (free when idle); the first request after
  idle pays a cold start (container boot + weights download, ~30–60 s).
  Set `--min-instances=1` to keep one instance warm if you demo it live.
- Drop `--allow-unauthenticated` if the endpoint should require IAM auth.

## 5. Smoke-test the deployment

```bash
SERVICE_URL=$(gcloud run services describe warp-api \
    --region=europe-west2 --format='value(status.url)')

curl "$SERVICE_URL/health"

curl -X POST "$SERVICE_URL/predict" \
    -F "file=@data/raw/warp_c/Warp-C/test_crops/cans/$(ls data/raw/warp_c/Warp-C/test_crops/cans | head -1)"
```

Or reuse the CI smoke test against production:

```bash
BASE_URL="$SERVICE_URL" python tests/smoke_test.py
```

## 6. Rolling out a new model version

```bash
# push v2 to the registry, then:
gcloud run services update warp-api \
    --region=europe-west2 \
    --update-env-vars=MODEL_URI=gs://YOUR_BUCKET_NAME/models/convnext_tiny/v2/model.pth
```

Cloud Run creates a new revision; `--to-revisions` lets you split traffic or
roll back instantly:

```bash
gcloud run services update-traffic warp-api \
    --region=europe-west2 --to-revisions=PREVIOUS_REVISION=100
```

## Cost guardrails

- Scale-to-zero + 3-instance cap keeps this inside/near the free tier for
  demo traffic.
- Delete everything with:
  `gcloud run services delete warp-api --region=europe-west2` and
  `gcloud artifacts repositories delete warp-images --location=europe-west2`.

# Model registry — versioned artifacts on Google Cloud Storage

Trained checkpoints are too large for git (they are gitignored). Instead, every
released model version is pushed to a GCS bucket with an immutable layout:

```
gs://<bucket>/models/
├── convnext_tiny/
│   ├── v1/
│   │   ├── model.pth              # trained weights
│   │   ├── metadata.json          # git sha, timestamp, file size
│   │   ├── summary.json           # evaluation metrics for this checkpoint
│   │   └── classification.yaml    # config snapshot used at training time
│   └── v2/ ...
└── vit_b_16/
    └── v1/ ...
```

One folder per model per version, each carrying the metrics and config that
produced it — a lightweight model registry: any version can be pulled,
compared, or served, and the push script refuses to overwrite an existing
version.

## One-time setup

Run these yourself (requires the [gcloud CLI](https://cloud.google.com/sdk/docs/install)):

```bash
# 1. Authenticate and pick your project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Create the bucket (europe-west2 = London)
gcloud storage buckets create gs://YOUR_BUCKET_NAME \
    --location=europe-west2 \
    --uniform-bucket-level-access

# 3. (Recommended) enable object versioning as an extra safety net,
#    so even an accidental overwrite is recoverable
gcloud storage buckets update gs://YOUR_BUCKET_NAME --versioning
```

## Pushing a model version

```bash
export GCS_BUCKET=YOUR_BUCKET_NAME
bash scripts/push_model_to_gcs.sh convnext_tiny v1
```

The script uploads the weights, a `metadata.json` (git commit, UTC timestamp,
file size), the evaluation `summary.json` if you have run
`python -m src.classification.evaluate --model convnext_tiny`, and a snapshot
of `configs/classification.yaml`.

## Consuming a version

The inference service downloads weights at startup when `MODEL_URI` is set:

```bash
MODEL_URI=gs://YOUR_BUCKET_NAME/models/convnext_tiny/v1/model.pth \
MODEL_NAME=convnext_tiny \
uvicorn api.main:app --port 8000
```

This is also how the Cloud Run deployment pulls weights — see
[deployment.md](deployment.md).

## Inspecting the registry

```bash
gcloud storage ls gs://YOUR_BUCKET_NAME/models/            # all models
gcloud storage ls gs://YOUR_BUCKET_NAME/models/convnext_tiny/   # all versions
gcloud storage cat gs://YOUR_BUCKET_NAME/models/convnext_tiny/v1/metadata.json
```

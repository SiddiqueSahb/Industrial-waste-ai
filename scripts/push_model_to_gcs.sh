#!/usr/bin/env bash
# Push a trained model (weights + metrics + config snapshot) to the versioned
# GCS model registry:
#
#     gs://$GCS_BUCKET/models/<model_name>/<version>/
#         model.pth
#         metadata.json      (git sha, date, metrics summary)
#         summary.json       (evaluation metrics, if present)
#         classification.yaml
#
# Usage (from the project root, after `gcloud auth login`):
#     export GCS_BUCKET=your-warp-models-bucket
#     bash scripts/push_model_to_gcs.sh convnext_tiny v1
#
# Requires: gcloud CLI (https://cloud.google.com/sdk/docs/install)
set -euo pipefail

MODEL_NAME="${1:?Usage: push_model_to_gcs.sh <model_name> <version>}"
VERSION="${2:?Usage: push_model_to_gcs.sh <model_name> <version>}"
BUCKET="${GCS_BUCKET:?Set GCS_BUCKET to your bucket name (no gs:// prefix)}"

# Resolve the weights file (matches configs/classification.yaml conventions)
WEIGHTS=""
for candidate in "models/classification/${MODEL_NAME}_best.pth" \
                 "models/classification/${MODEL_NAME}.pth"; do
    if [[ -f "$candidate" && $(stat -f%z "$candidate" 2>/dev/null || stat -c%s "$candidate") -gt 10000 ]]; then
        WEIGHTS="$candidate"
        break
    fi
done
[[ -n "$WEIGHTS" ]] || { echo "ERROR: no valid weights found for ${MODEL_NAME}"; exit 1; }

DEST="gs://${BUCKET}/models/${MODEL_NAME}/${VERSION}"

# Refuse to silently overwrite an existing version — versions are immutable
if gcloud storage ls "${DEST}/model.pth" >/dev/null 2>&1; then
    echo "ERROR: ${DEST} already exists. Bump the version instead of overwriting."
    exit 1
fi

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
SUMMARY="results/evaluation/${MODEL_NAME}/summary.json"

METADATA=$(mktemp)
cat > "$METADATA" <<EOF
{
  "model_name": "${MODEL_NAME}",
  "version": "${VERSION}",
  "git_sha": "${GIT_SHA}",
  "pushed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "weights_file": "$(basename "$WEIGHTS")",
  "weights_bytes": $(stat -f%z "$WEIGHTS" 2>/dev/null || stat -c%s "$WEIGHTS")
}
EOF

echo "Pushing ${MODEL_NAME} ${VERSION} -> ${DEST}"
gcloud storage cp "$WEIGHTS" "${DEST}/model.pth"
gcloud storage cp "$METADATA" "${DEST}/metadata.json"
gcloud storage cp configs/classification.yaml "${DEST}/classification.yaml"
[[ -f "$SUMMARY" ]] && gcloud storage cp "$SUMMARY" "${DEST}/summary.json"
rm -f "$METADATA"

echo
echo "Done. Registry contents for ${MODEL_NAME}:"
gcloud storage ls -l "gs://${BUCKET}/models/${MODEL_NAME}/**"
echo
echo "Serve this version with:"
echo "  MODEL_URI=${DEST}/model.pth"

"""Pydantic response models for the inference API."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel


class TopKEntry(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    model_name: str
    predicted_class: str
    confidence: float
    top_k: List[TopKEntry]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_name: str
    device: str
    weights_loaded: bool
    num_classes: int

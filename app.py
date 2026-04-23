# app.py
# Minimal FastAPI wrapper around the self-pruning network.
#
# Endpoints:
#   GET  /          → health check
#   POST /train     → runs training for a given lambda value
#   GET  /metrics   → returns latest accuracy and sparsity
#
# Run with:
#   uvicorn app:app --reload

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = logging.getLogger("uvicorn.error")

# In-memory store for the latest training result.
# Replace with a database or file-based store for anything production-grade.
_latest_metrics: dict = {}


#  App lifecycle 

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Self-Pruning NN API is starting up.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Self-Pruning Neural Network API",
    description=(
        "Train a self-pruning CNN on CIFAR-10 and query the results. "
        "Built as a case study for the Tredence AI Engineering Internship."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


#  Schemas 

class TrainRequest(BaseModel):
    lambda_val: float = 5.0
    num_epochs: int   = 20

class TrainResponse(BaseModel):
    lambda_val: float
    accuracy:   float
    sparsity:   float
    message:    str

class MetricsResponse(BaseModel):
    lambda_val: float | None
    accuracy:   float | None
    sparsity:   float | None


#  Endpoints 

@app.get("/", summary="Health check")
def health():
    return {"status": "ok", "service": "self-pruning-nn"}


@app.post("/train", response_model=TrainResponse, summary="Run training")
def train_endpoint(req: TrainRequest):
    """
    Triggers a full training run for the given lambda value.

    Warning: this is a blocking call — training takes several minutes on CPU.
    For production use, offload to a background task queue (Celery, RQ, etc.)
    and return a job ID instead.
    """
    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    from config import DEVICE
    from model import SelfPruningNet
    from evaluate import evaluate
    from utils import get_data_loaders, compute_sparsity_loss

    log.info(f"Training request received  lambda={req.lambda_val}  epochs={req.num_epochs}")

    try:
        train_loader, test_loader = get_data_loaders()
        model     = SelfPruningNet().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, req.num_epochs + 1):
            model.train()
            for data, target in train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                logits  = model(data)
                loss    = F.cross_entropy(logits, target) + req.lambda_val * compute_sparsity_loss(model)
                loss.backward()
                optimizer.step()
            log.info(f"Epoch {epoch}/{req.num_epochs} done")

        acc, sparsity = evaluate(model, test_loader, DEVICE)

        _latest_metrics.update({
            "lambda_val": req.lambda_val,
            "accuracy":   round(acc, 4),
            "sparsity":   round(sparsity, 4),
        })

        return TrainResponse(
            lambda_val=req.lambda_val,
            accuracy=round(acc, 2),
            sparsity=round(sparsity, 2),
            message="Training complete.",
        )

    except Exception as e:
        log.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse, summary="Get latest metrics")
def get_metrics():
    """Returns the accuracy and sparsity from the most recent training run."""
    if not _latest_metrics:
        return MetricsResponse(lambda_val=None, accuracy=None, sparsity=None)
    return MetricsResponse(**_latest_metrics)

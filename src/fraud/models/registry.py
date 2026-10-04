"""Minimal file-based model registry with champion/challenger promotion.

Mirrors what MLflow / SageMaker Model Registry give you:
  model_registry/fraud/<version>/model.joblib
  model_registry/fraud/<version>/metadata.json
  model_registry/fraud/<version>/reference.parquet   (training snapshot for drift)
  model_registry/fraud/PRODUCTION                    (pointer to live version)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

MODEL_NAME = "fraud"


class ModelRegistry:
    def __init__(self, root: str):
        self.root = Path(root) / MODEL_NAME
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, model, metadata: dict, reference: pd.DataFrame) -> str:
        version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S_%f")
        d = self.root / version
        d.mkdir()
        joblib.dump(model, d / "model.joblib")
        reference.to_parquet(d / "reference.parquet", index=False)
        metadata = {**metadata, "version": version,
                    "registered_at": datetime.now(timezone.utc).isoformat()}
        (d / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        return version

    def production_version(self) -> str | None:
        p = self.root / "PRODUCTION"
        return p.read_text().strip() if p.exists() else None

    def promote(self, version: str) -> None:
        (self.root / "PRODUCTION").write_text(version)

    def load(self, version: str | None = None):
        version = version or self.production_version()
        if version is None:
            raise FileNotFoundError("No production model. Run `make train` first.")
        d = self.root / version
        meta = json.loads((d / "metadata.json").read_text())
        return joblib.load(d / "model.joblib"), meta, pd.read_parquet(d / "reference.parquet")

    def maybe_promote(self, version: str, metric: str) -> bool:
        """Challenger replaces champion only if it is at least as good on the test window."""
        champ = self.production_version()
        if champ is None:
            self.promote(version)
            return True
        _, champ_meta, _ = self.load(champ)
        _, chall_meta, _ = self.load(version)
        if chall_meta["test_metrics"][metric] >= champ_meta["test_metrics"][metric]:
            self.promote(version)
            return True
        return False

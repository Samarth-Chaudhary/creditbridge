"""
CreditBridge - Phase 1 Experiment Engine & Single Source of Truth
Path: src/experiment_engine.py

Provides the versioned experiment contract and persistence layer.
Every model run produces a self-contained, auditable experiment directory:
experiments/<experiment_id>/
  - config.json
  - dataset_manifest.json
  - feature_schema.json
  - metrics.json
  - calibration.json
  - fairness.json
  - deciles.csv
  - predictions.csv
  - model_metadata.json
  - artifact_hash.txt
  - README.md
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def get_git_commit_hash(repo_dir: Optional[Path] = None) -> str:
    """Retrieves current git commit hash if available, otherwise returns 'uncommitted_workspace'."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        cwd = repo_dir if repo_dir else Path(__file__).resolve().parent.parent
        res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "uncommitted_or_git_unavailable"


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file."""
    p = Path(filepath)
    if not p.exists():
        return "file_not_found"
    sha = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compute_dataframe_sha256(df: pd.DataFrame) -> str:
    """Computes deterministic hash of a pandas DataFrame."""
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


@dataclass
class DatasetManifest:
    dataset_version: str
    dataset_hash: str
    dataset_path: str
    row_count: int
    column_count: int
    target_column: str
    class_0_count: int
    class_1_count: int
    default_rate: float
    train_rows: int
    val_rows: int
    test_rows: int
    oot_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureSchema:
    schema_version: str
    feature_count: int
    numeric_features: List[str]
    categorical_features: List[str]
    derived_features: List[str]
    self_reported_features: List[str]
    imputed_features: List[str]
    target: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetadata:
    model_name: str
    model_type: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    preprocessing_version: str
    artifact_path: str
    artifact_hash: str
    coefficients: Optional[Dict[str, float]] = None
    feature_importances: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricSummary:
    roc_auc: float
    pr_auc: float
    ks_statistic: float
    gini: float
    brier_score: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]
    calibration_intercept: float
    calibration_slope: float
    ci_95: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRun:
    experiment_id: str
    timestamp: str
    git_commit: str
    random_seed: int
    status: str  # "CHAMPION", "CHALLENGER", "BASELINE", "REJECTED"
    dataset_manifest: DatasetManifest
    feature_schema: FeatureSchema
    model_metadata: ModelMetadata
    train_metrics: MetricSummary
    val_metrics: MetricSummary
    test_metrics: MetricSummary
    oot_metrics: MetricSummary
    calibration_metrics: Dict[str, Any]
    fairness_metrics: Dict[str, Any]
    champion_selection: Dict[str, Any]
    economic_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "random_seed": self.random_seed,
            "status": self.status,
            "dataset_manifest": self.dataset_manifest.to_dict(),
            "feature_schema": self.feature_schema.to_dict(),
            "model_metadata": self.model_metadata.to_dict(),
            "train_metrics": self.train_metrics.to_dict(),
            "val_metrics": self.val_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
            "oot_metrics": self.oot_metrics.to_dict(),
            "calibration_metrics": self.calibration_metrics,
            "fairness_metrics": self.fairness_metrics,
            "champion_selection": self.champion_selection,
            "economic_metrics": self.economic_metrics or {},
        }


def save_experiment_run(
    run: ExperimentRun,
    deciles_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    base_experiments_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Persists an experiment run into experiments/<experiment_id>/ following the single source of truth contract.
    """
    if base_experiments_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        experiments_root = project_root / "experiments"
    else:
        experiments_root = Path(base_experiments_dir)

    exp_dir = experiments_root / run.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. config.json
    config_data = {
        "experiment_id": run.experiment_id,
        "timestamp": run.timestamp,
        "git_commit": run.git_commit,
        "random_seed": run.random_seed,
        "status": run.status,
        "model_name": run.model_metadata.model_name,
        "hyperparameters": run.model_metadata.hyperparameters,
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # 2. dataset_manifest.json
    with open(exp_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run.dataset_manifest.to_dict(), f, indent=2)

    # 3. feature_schema.json
    with open(exp_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(run.feature_schema.to_dict(), f, indent=2)

    # 4. metrics.json (all splits)
    metrics_data = {
        "train": run.train_metrics.to_dict(),
        "validation": run.val_metrics.to_dict(),
        "test": run.test_metrics.to_dict(),
        "oot": run.oot_metrics.to_dict(),
        "champion_selection": run.champion_selection,
        "economic_decisioning": run.economic_metrics or {},
    }
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # 5. calibration.json
    with open(exp_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(run.calibration_metrics, f, indent=2)

    # 6. fairness.json
    with open(exp_dir / "fairness.json", "w", encoding="utf-8") as f:
        json.dump(run.fairness_metrics, f, indent=2)

    # 7. deciles.csv
    deciles_df.to_csv(exp_dir / "deciles.csv", index=False)

    # 8. predictions.csv
    predictions_df.to_csv(exp_dir / "predictions.csv", index=False)

    # 9. model_metadata.json
    with open(exp_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run.model_metadata.to_dict(), f, indent=2)

    # 10. artifact_hash.txt (hashes of all files in this experiment directory)
    hashes: Dict[str, str] = {}
    for item in sorted(os.listdir(exp_dir)):
        if item != "artifact_hash.txt" and item != "README.md":
            item_path = exp_dir / item
            if item_path.is_file():
                hashes[item] = compute_file_sha256(item_path)

    with open(exp_dir / "artifact_hash.txt", "w", encoding="utf-8") as f:
        for fname, h in hashes.items():
            f.write(f"{h}  {fname}\n")

    # 11. README.md
    readme_content = f"""# Experiment Report: {run.experiment_id}

- **Model Name**: {run.model_metadata.model_name}
- **Status**: {run.status}
- **Timestamp**: {run.timestamp}
- **Git Commit**: `{run.git_commit}`
- **Random Seed**: {run.random_seed}

## Headline Performance Summary
| Metric | Out-of-Time (OOT) | Test Set | Validation | Train |
| :--- | :--- | :--- | :--- | :--- |
| **ROC-AUC** | **{run.oot_metrics.roc_auc:.4f}** | {run.test_metrics.roc_auc:.4f} | {run.val_metrics.roc_auc:.4f} | {run.train_metrics.roc_auc:.4f} |
| **PR-AUC** | **{run.oot_metrics.pr_auc:.4f}** | {run.test_metrics.pr_auc:.4f} | {run.val_metrics.pr_auc:.4f} | {run.train_metrics.pr_auc:.4f} |
| **KS-Statistic (%)** | **{run.oot_metrics.ks_statistic:.2f}%** | {run.test_metrics.ks_statistic:.2f}% | {run.val_metrics.ks_statistic:.2f}% | {run.train_metrics.ks_statistic:.2f}% |
| **Gini** | **{run.oot_metrics.gini:.4f}** | {run.test_metrics.gini:.4f} | {run.val_metrics.gini:.4f} | {run.train_metrics.gini:.4f} |
| **Brier Score** | **{run.oot_metrics.brier_score:.4f}** | {run.test_metrics.brier_score:.4f} | {run.val_metrics.brier_score:.4f} | {run.train_metrics.brier_score:.4f} |
| **Precision** | **{run.oot_metrics.precision:.4f}** | {run.test_metrics.precision:.4f} | {run.val_metrics.precision:.4f} | {run.train_metrics.precision:.4f} |
| **Recall** | **{run.oot_metrics.recall:.4f}** | {run.test_metrics.recall:.4f} | {run.val_metrics.recall:.4f} | {run.train_metrics.recall:.4f} |
| **F1-Score** | **{run.oot_metrics.f1:.4f}** | {run.test_metrics.f1:.4f} | {run.val_metrics.f1:.4f} | {run.train_metrics.f1:.4f} |

## Champion Selection Status
- Decision: `{run.champion_selection.get("decision", "N/A")}`
- Rationale: {run.champion_selection.get("rationale", "N/A")}

All figures generated deterministically and verifiable against `artifact_hash.txt`.
"""
    with open(exp_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Also update experiments/latest_champion.json pointer if champion
    if run.status == "CHAMPION":
        pointer_data = {
            "latest_champion_id": run.experiment_id,
            "timestamp": run.timestamp,
            "model_name": run.model_metadata.model_name,
            "oot_auc": run.oot_metrics.roc_auc,
            "oot_ks": run.oot_metrics.ks_statistic,
            "oot_pr_auc": run.oot_metrics.pr_auc,
            "experiment_dir": str(exp_dir.resolve()),
        }
        with open(experiments_root / "latest_champion.json", "w", encoding="utf-8") as f:
            json.dump(pointer_data, f, indent=2)

    return exp_dir


def load_experiment_run(experiment_id_or_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads all persisted artifacts for an experiment into a structured dictionary."""
    p = Path(experiment_id_or_path)
    if not p.is_dir():
        project_root = Path(__file__).resolve().parent.parent
        p = project_root / "experiments" / str(experiment_id_or_path)

    if not p.exists():
        raise FileNotFoundError(f"Experiment directory not found at: {p}")

    res: Dict[str, Any] = {}
    for json_file in ["config.json", "dataset_manifest.json", "feature_schema.json", "metrics.json", "calibration.json", "fairness.json", "model_metadata.json"]:
        jf = p / json_file
        if jf.exists():
            with open(jf, "r", encoding="utf-8") as f:
                res[json_file.replace(".json", "")] = json.load(f)

    if (p / "deciles.csv").exists():
        res["deciles"] = pd.read_csv(p / "deciles.csv")

    if (p / "predictions.csv").exists():
        res["predictions"] = pd.read_csv(p / "predictions.csv")

    return res


def load_latest_champion(experiments_root: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    """Loads the current latest champion experiment if one exists."""
    if experiments_root is None:
        project_root = Path(__file__).resolve().parent.parent
        root = project_root / "experiments"
    else:
        root = Path(experiments_root)

    pointer_file = root / "latest_champion.json"
    if not pointer_file.exists():
        return None

    with open(pointer_file, "r", encoding="utf-8") as f:
        pointer = json.load(f)

    champ_id = pointer.get("latest_champion_id")
    if not champ_id:
        return None

    return load_experiment_run(root / champ_id)

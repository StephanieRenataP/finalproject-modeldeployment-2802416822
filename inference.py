from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import training_pipeline  


# Prediction Result (value object)
# ==============================================================================
@dataclass(frozen=True)
class PredictionResult:

    predicted_class: str
    class_probabilities: Dict[str, float]
    raw_argmax_class: str
    threshold_applied: bool = False

    @property
    def confidence(self) -> float:
        return self.class_probabilities[self.predicted_class]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "raw_argmax_class": self.raw_argmax_class,
            "threshold_applied": self.threshold_applied,
            "confidence": round(self.confidence, 4),
            "class_probabilities": {k: round(v, 4) for k, v in self.class_probabilities.items()},
        }


# Input Validator
# ==============================================================================
class InputValidator:

    def __init__(self, required_columns: List[str]):
        self._required_columns = required_columns

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self._required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Kolom input berikut wajib ada namun tidak ditemukan: {missing}. "
                f"Kolom yang dibutuhkan: {self._required_columns}"
            )
        if df.empty:
            raise ValueError("Data input kosong (0 baris).")


# Main Service Inferencing
# ==============================================================================
class CreditScoreInferenceService:

    def __init__(self, artifact: Dict[str, Any]):
        self._pipeline = artifact["pipeline"]
        self._raw_feature_columns: List[str] = artifact["raw_feature_columns"]
        self._target_classes: List[str] = artifact["target_classes"]
        # Kompatibel dengan artefak model v1/v2 (belum punya threshold tuning)
        self._minority_class: Optional[str] = artifact.get("minority_class")
        self._good_class_threshold: Optional[float] = artifact.get("good_class_threshold")
        self._validator = InputValidator(self._raw_feature_columns)

    # -- Factory Method (constructor alternatif) --
    @classmethod
    def from_pickle(cls, model_path: Union[str, Path]) -> "CreditScoreInferenceService":
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"File model tidak ditemukan: {model_path.resolve()}")
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        return cls(artifact)

    # -- Public Property (read-only, encapsulation) --
    @property
    def required_columns(self) -> List[str]:
        return list(self._raw_feature_columns)

    @property
    def target_classes(self) -> List[str]:
        return list(self._target_classes)

    @property
    def minority_class(self) -> Optional[str]:
        return self._minority_class

    @property
    def good_class_threshold(self) -> Optional[float]:
        return self._good_class_threshold

    @property
    def has_threshold_tuning(self) -> bool:
        return self._minority_class is not None and self._good_class_threshold is not None

    # -- API --
    def _prepare_frame(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Lengkapi kolom yang tidak dikirim (mis. ID/SSN/Name yang bersifat identitas dan tidak wajib diisi pengguna web) dengan NaN, karena
        # kolom-kolom tsb pada akhirnya dibuang oleh CreditDataCleaner dan tidak memengaruhi prediksi.
        for col in self._raw_feature_columns:
            if col not in df.columns:
                df[col] = np.nan

        return df[self._raw_feature_columns]

    def predict_batch(
        self,
        data: Union[Dict[str, Any], pd.DataFrame],
        apply_good_threshold: bool = False,
    ) -> List[PredictionResult]:

        df = self._prepare_frame(data)
        self._validator.validate(df)

        predictions = self._pipeline.predict(df)
        probabilities = self._pipeline.predict_proba(df)
        classifier_classes = list(self._pipeline.classes_)

        use_threshold = apply_good_threshold and self.has_threshold_tuning
        minority_idx = classifier_classes.index(self._minority_class) if use_threshold else None

        results: List[PredictionResult] = []
        for pred, proba_row in zip(predictions, probabilities):
            proba_dict = {cls: float(p) for cls, p in zip(classifier_classes, proba_row)}
            final_class = str(pred)
            threshold_applied = False

            if use_threshold and proba_row[minority_idx] >= self._good_class_threshold:
                final_class = self._minority_class
                threshold_applied = final_class != str(pred)

            results.append(PredictionResult(
                predicted_class=final_class,
                class_probabilities=proba_dict,
                raw_argmax_class=str(pred),
                threshold_applied=threshold_applied,
            ))
        return results

    def predict_single(
        self, data: Dict[str, Any], apply_good_threshold: bool = False
    ) -> PredictionResult:
        return self.predict_batch(data, apply_good_threshold=apply_good_threshold)[0]


# Fast CLI for Manual Checking
# ==============================================================================
def _demo() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Uji cepat inference dari CLI.")
    parser.add_argument("--model-path", default="credit_score_model.pkl")
    parser.add_argument("--csv-path", required=True, help="Path CSV data mentah (format sama seperti data_D.csv)")
    args = parser.parse_args()

    service = CreditScoreInferenceService.from_pickle(args.model_path)
    df = pd.read_csv(args.csv_path)
    results = service.predict_batch(df)
    for i, r in enumerate(results):
        print(f"Baris {i}: {r.to_dict()}")


if __name__ == "__main__":
    _demo()

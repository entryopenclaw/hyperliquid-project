from __future__ import annotations

from dataclasses import dataclass

from .models import FeatureVector, ModelArtifact, SignalPrediction
from .utils import sigmoid


@dataclass(slots=True)
class WeightedFeatureModel:
    version: str
    weights: dict[str, float]
    intercept: float = 0.0

    def predict(self, features: FeatureVector) -> SignalPrediction:
        score = self.intercept
        for key, value in features.values.items():
            score += self.weights.get(key, 0.0) * value
        expected_return_bps = score * 10.0
        adverse_move_bps = max(0.5, features.values.get("realized_vol_20", 0.0) * 10_000.0)
        confidence = sigmoid(abs(score) * 2.0)
        return SignalPrediction(
            timestamp=features.timestamp,
            symbol=features.symbol,
            model_version=self.version,
            expected_return_bps=expected_return_bps,
            adverse_move_bps=adverse_move_bps,
            confidence=confidence,
            score=score,
            feature_values=features.values,
        )

    @classmethod
    def from_artifact(cls, artifact: ModelArtifact) -> "WeightedFeatureModel":
        return cls(version=artifact.version, weights=artifact.weights, intercept=artifact.intercept)

    @classmethod
    def heuristic(cls) -> "WeightedFeatureModel":
        return cls(
            version="heuristic-v1",
            intercept=0.0,
            weights={
                "depth_imbalance": 0.45,
                "trade_imbalance": 0.35,
                "momentum_5": 8.0,
                "momentum_20": 10.0,
                "trend_alignment": 8.0,
                "microprice_deviation": 12.0,
                "spread_bps": -0.03,
                "realized_vol_20": -3.0,
                "premium": -1.0,
            },
        )


class SignalEngine:
    def __init__(self, model: WeightedFeatureModel | None = None):
        self.model = model or WeightedFeatureModel.heuristic()

    def load_model(self, artifact: ModelArtifact | None) -> None:
        self.model = WeightedFeatureModel.from_artifact(artifact) if artifact else WeightedFeatureModel.heuristic()

    def predict(self, features: FeatureVector) -> SignalPrediction:
        return self.model.predict(features)

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator


def score_features(model: BaseEstimator, features: pd.DataFrame) -> pd.DataFrame:
    scores = model.decision_function(features)
    labels = model.predict(features)
    return pd.DataFrame(
        {
            "anomaly_score": -scores,
            "is_anomaly": labels == -1,
        },
        index=features.index,
    )

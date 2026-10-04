"""
CreditBridge - Temporal Dataset Contract & Leakage Defense
Path: src/temporal_contract.py

Enforces explicit observation and prediction windows:
- Observation Window (T_obs_start to T_obs_end): Only events occurring in this window
  are permitted to enter feature engineering.
- Prediction Horizon (T_pred_start to T_pred_end): Forward window in which default
  events are observed.
- Leakage Defense: Automated checks that reject any records or transactions stamped
  after the observation cutoff T_obs_end.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Tuple, Union

import pandas as pd


class TemporalLeakageError(ValueError):
    """Raised when data from the future prediction window leaks into the feature observation window."""
    pass


@dataclass(frozen=True)
class TemporalWindowContract:
    """Defines exact temporal boundaries for observation and prediction."""
    observation_months: int = 12
    prediction_months: int = 3
    cutoff_date: Optional[date] = None

    def validate_transaction_dates(
        self,
        transaction_dates: pd.Series,
        observation_end: Union[str, date, datetime],
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates that no transaction occurs after the observation cutoff date.
        """
        cutoff = pd.to_datetime(observation_end)
        tx_dates = pd.to_datetime(transaction_dates)

        future_tx = tx_dates > cutoff
        if future_tx.any():
            max_future = tx_dates[future_tx].max()
            count_future = future_tx.sum()
            msg = (
                f"TEMPORAL LEAKAGE DETECTED: {count_future} transactions occur after "
                f"observation cutoff {cutoff.strftime('%Y-%m-%d')}. "
                f"Max future transaction date: {max_future.strftime('%Y-%m-%d')}."
            )
            return False, msg
        return True, None


def assert_zero_temporal_leakage(
    transactions_df: pd.DataFrame,
    cutoff_date: Union[str, date, datetime],
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Enforces that transactions strictly precede or coincide with cutoff_date.
    If any transaction is post-cutoff, raises TemporalLeakageError with machine-readable detail.
    Otherwise returns filtered valid observation transactions.
    """
    if date_column not in transactions_df.columns:
        raise ValueError(f"Date column '{date_column}' not found in transactions DataFrame")

    tx_dates = pd.to_datetime(transactions_df[date_column])
    cutoff = pd.to_datetime(cutoff_date)

    future_mask = tx_dates > cutoff
    if future_mask.any():
        num_leaked = int(future_mask.sum())
        max_leaked = str(tx_dates[future_mask].max().date())
        raise TemporalLeakageError(
            f"TEMPORAL_LEAKAGE_REJECTED: Found {num_leaked} post-cutoff transactions "
            f"beyond {cutoff.date()} (latest leaked: {max_leaked}). "
            f"Feature calculation must never observe future events."
        )

    return transactions_df.copy()


def split_temporal_dataset(
    df: pd.DataFrame,
    split_column: str = "cohort_split",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into deterministic temporal partitions: train, validation, out-of-time (OOT).
    """
    if split_column not in df.columns:
        raise ValueError(f"Split column '{split_column}' not found in DataFrame.")

    from typing import cast
    train_df = cast(pd.DataFrame, df[df[split_column] == "train"].copy())
    val_df = cast(pd.DataFrame, df[df[split_column] == "val"].copy())
    oot_df = cast(pd.DataFrame, df[df[split_column] == "oot"].copy())

    if len(train_df) == 0 or len(val_df) == 0 or len(oot_df) == 0:
        raise ValueError(
            f"Invalid temporal split sizes: train={len(train_df)}, val={len(val_df)}, oot={len(oot_df)}. "
            f"All temporal splits must have non-zero rows."
        )

    return train_df, val_df, oot_df

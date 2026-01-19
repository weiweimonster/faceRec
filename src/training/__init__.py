"""
Training module for XGBoost hyperparameter search.

Loads features from SQL features_json column and performs
grid search over feature combinations and hyperparameters.
"""

from .trainer import XGBoostTrainer, Leaderboard

__all__ = ["XGBoostTrainer", "Leaderboard"]

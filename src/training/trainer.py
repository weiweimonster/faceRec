"""
XGBoost hyperparameter search trainer.

Loads features from SQL features_json column, auto-detects feature names,
and performs grid search over feature combinations and hyperparameters.
Saves top N models to src/rank/models/.
"""

import itertools
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from src.util.logger import logger


@dataclass
class ModelEntry:
    """A single entry in the leaderboard."""
    score: float
    features: Tuple[str, ...]
    params: Dict[str, Any]
    model: Optional[xgb.XGBRanker] = None


class Leaderboard:
    """Tracks top N models during hyperparameter search."""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.entries: List[ModelEntry] = []

    def update(
        self,
        score: float,
        features: Tuple[str, ...],
        params: Dict[str, Any],
        model: Optional[xgb.XGBRanker] = None
    ) -> Optional[int]:
        """
        Update leaderboard with a new result.

        Returns:
            Rank (1-indexed) if model made it to leaderboard, None otherwise.
        """
        # Check if it qualifies
        if len(self.entries) < self.capacity:
            qualifies = True
        elif score > self.entries[-1].score:
            qualifies = True
        else:
            qualifies = False

        if not qualifies:
            return None

        # Insert entry
        entry = ModelEntry(
            score=score,
            features=features,
            params=params.copy(),
            model=model
        )
        self.entries.append(entry)

        # Sort descending by score (higher NDCG is better)
        self.entries.sort(key=lambda x: x.score, reverse=True)

        # Trim to capacity
        if len(self.entries) > self.capacity:
            self.entries = self.entries[:self.capacity]

        # Return rank if still in leaderboard
        try:
            return self.entries.index(entry) + 1
        except ValueError:
            return None

    def get_top_models(self) -> List[ModelEntry]:
        """Get all entries in the leaderboard."""
        return self.entries.copy()

    def print_board(self):
        """Print formatted leaderboard to console."""
        print("\n" + "=" * 100)
        print(f"TOP {self.capacity} MODELS LEADERBOARD")
        print("=" * 100)
        print(f"{'Rank':<5} | {'Score':<8} | {'Features':<50} | {'Params'}")
        print("-" * 100)

        for i, entry in enumerate(self.entries):
            feat_str = ", ".join(entry.features)
            if len(feat_str) > 48:
                feat_str = feat_str[:45] + "..."

            param_str = ", ".join(f"{k}={v}" for k, v in entry.params.items())

            print(f"#{i+1:<4} | {entry.score:.5f}  | {feat_str:<50} | {param_str}")

        print("=" * 100 + "\n")


class XGBoostTrainer:
    """
    XGBoost hyperparameter search trainer.

    Loads training data from SQL features_json column, auto-detects
    available features, and performs grid search.
    """

    # Default hyperparameter grid
    DEFAULT_PARAM_GRID = {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
        "n_estimators": [50, 100, 200],
        "subsample": [0.8, 1.0]
    }

    def __init__(
        self,
        db_path: str = ".db/sqlite/photos.db",
        model_output_dir: str = "src/rank/models",
        param_grid: Optional[Dict[str, List[Any]]] = None,
        min_features: int = 2,
        max_features: Optional[int] = None,
        leaderboard_capacity: int = 10,
        test_size: float = 0.2,
        random_state: int = 42,
        early_stopping_rounds: int = 10
    ):
        """
        Initialize the trainer.

        Args:
            db_path: Path to SQLite database.
            model_output_dir: Directory to save top models.
            param_grid: Hyperparameter grid for search. Uses defaults if None.
            min_features: Minimum number of features per combination.
            max_features: Maximum features per combination. None = all features.
            leaderboard_capacity: Number of top models to track.
            test_size: Fraction of data for validation.
            random_state: Random seed for reproducibility.
            early_stopping_rounds: XGBoost early stopping patience.
        """
        self.db_path = db_path
        self.model_output_dir = model_output_dir
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID
        self.min_features = min_features
        self.max_features = max_features
        self.leaderboard_capacity = leaderboard_capacity
        self.test_size = test_size
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds

        # Will be populated after loading data
        self.df: Optional[pd.DataFrame] = None
        self.all_features: List[str] = []
        self.leaderboard = Leaderboard(capacity=leaderboard_capacity)

    def load_data(self) -> pd.DataFrame:
        """
        Load training data from SQL features_json column.

        Returns:
            DataFrame with features, labels, and session grouping.
        """
        logger.info(f"Loading training data from {self.db_path}")

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)

        # Fetch interactions with features_json
        query = """
            SELECT i.session_id, i.label, i.features_json
            FROM search_interactions i
            JOIN search_history h ON i.session_id = h.session_id
            WHERE i.features_json IS NOT NULL
            ORDER BY h.timestamp ASC
        """

        rows = conn.execute(query).fetchall()
        conn.close()

        logger.info(f"Found {len(rows)} interactions with features")

        if not rows:
            raise ValueError("No training data found in database")

        # Parse JSON and build DataFrame
        data = []
        for session_id, label, features_json in rows:
            try:
                features = json.loads(features_json)
                features["session_id"] = session_id
                features["label"] = label
                data.append(features)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse features JSON: {e}")
                continue

        df = pd.DataFrame(data)

        if df.empty:
            raise ValueError("No valid training data after parsing")

        # Auto-detect feature columns (exclude meta columns)
        meta_cols = {"session_id", "label", "qid"}
        self.all_features = [c for c in df.columns if c not in meta_cols]
        logger.info(f"Auto-detected {len(self.all_features)} features: {self.all_features}")

        # Create query groups (QID) for ranking
        df["qid"] = df.groupby("session_id").ngroup()

        # Sort by QID (critical for XGBoost ranking)
        df = df.sort_values("qid").reset_index(drop=True)

        self.df = df
        logger.info(f"Loaded {len(df)} training rows across {df['qid'].nunique()} sessions")

        return df

    def _generate_feature_combinations(self) -> List[Tuple[str, ...]]:
        """Generate all feature combinations to try."""
        max_feat = self.max_features or len(self.all_features)
        max_feat = min(max_feat, len(self.all_features))

        combinations = []
        for r in range(self.min_features, max_feat + 1):
            combinations.extend(itertools.combinations(self.all_features, r))

        return combinations

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all hyperparameter combinations to try."""
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _train_single(
        self,
        features: Tuple[str, ...],
        params: Dict[str, Any]
    ) -> Tuple[float, Optional[xgb.XGBRanker]]:
        """
        Train a single model configuration.

        Returns:
            Tuple of (score, model). Score is 0.0 and model is None on failure.
        """
        if self.df is None:
            return 0.0, None

        # Prepare data
        X = self.df[list(features)]
        y = self.df["label"]
        qids = self.df["qid"]

        # Split by groups (sessions)
        gss = GroupShuffleSplit(
            test_size=self.test_size,
            n_splits=1,
            random_state=self.random_state
        )

        try:
            train_idx, val_idx = next(gss.split(X, y, groups=qids))
        except Exception as e:
            logger.debug(f"Split failed: {e}")
            return 0.0, None

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        qid_train = qids.iloc[train_idx]
        qid_val = qids.iloc[val_idx]

        # Create and train model
        model = xgb.XGBRanker(
            tree_method="hist",
            objective="rank:pairwise",
            eval_metric="ndcg@5",
            early_stopping_rounds=self.early_stopping_rounds,
            **params
        )

        try:
            model.fit(
                X_train, y_train,
                group=qid_train.value_counts().sort_index().values,
                eval_set=[(X_val, y_val)],
                eval_group=[qid_val.value_counts().sort_index().values],
                verbose=False
            )
        except Exception as e:
            logger.debug(f"Training failed: {e}")
            return 0.0, None

        # Get best score
        if hasattr(model, 'best_score'):
            return model.best_score, model

        return 0.0, None

    def run(self, save_models: bool = True, verbose: bool = True) -> Leaderboard:
        """
        Run the full hyperparameter search.

        Args:
            save_models: Whether to save top models to disk.
            verbose: Whether to print progress updates.

        Returns:
            Leaderboard with top models.
        """
        # Load data if not already loaded
        if self.df is None:
            self.load_data()

        # Generate combinations
        feature_combos = self._generate_feature_combinations()
        param_combos = self._generate_param_combinations()
        total_runs = len(feature_combos) * len(param_combos)

        if verbose:
            print(f"\nStarting Hyperparameter Search")
            print(f"  Feature combinations: {len(feature_combos)}")
            print(f"  Param combinations: {len(param_combos)}")
            print(f"  Total configurations: {total_runs}")
            print("=" * 80)

        start_time = time.time()
        processed = 0

        for feats in feature_combos:
            for params in param_combos:
                processed += 1

                score, model = self._train_single(feats, params)

                # Update leaderboard
                rank = self.leaderboard.update(score, feats, params, model)

                if verbose and rank is not None:
                    print(f"NEW TOP {rank}: {score:.5f} | Iter {processed}/{total_runs}")
                    if rank <= 3:
                        print(f"   Features: {feats}")
                        print(f"   Params: {params}")

                # Progress heartbeat
                if verbose and processed % 500 == 0:
                    elapsed = time.time() - start_time
                    remaining = (total_runs - processed) * (elapsed / processed)
                    print(f"   ... {processed}/{total_runs} done. "
                          f"Est remaining: {timedelta(seconds=int(remaining))}")

        # Print final leaderboard
        if verbose:
            self.leaderboard.print_board()
            total_time = time.time() - start_time
            print(f"Total time: {timedelta(seconds=int(total_time))}")

        # Save top models
        if save_models:
            self.save_top_models()

        return self.leaderboard

    def save_top_models(self):
        """Save top models to disk."""
        # Create output directory
        output_dir = Path(self.model_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving top {len(self.leaderboard.entries)} models to {output_dir}")

        for i, entry in enumerate(self.leaderboard.entries):
            if entry.model is None:
                logger.warning(f"Model {i+1} has no trained model, skipping")
                continue

            # Build filename with rank and score
            filename = f"model_rank{i+1}_score{entry.score:.4f}.json"
            filepath = output_dir / filename

            # Get the underlying Booster and save
            booster = entry.model.get_booster()

            # Set feature names on the booster so XGBoostRanker can read them
            booster.feature_names = list(entry.features)

            booster.save_model(str(filepath))

            # Also save metadata
            meta_path = output_dir / f"model_rank{i+1}_meta.json"
            meta = {
                "rank": i + 1,
                "score": entry.score,
                "features": list(entry.features),
                "params": entry.params
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"Saved model {i+1}: {filepath}")

        print(f"\nSaved {len(self.leaderboard.entries)} models to {output_dir}")


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost hyperparameter search")
    parser.add_argument(
        "--db-path",
        default=".db/sqlite/photos.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--output-dir",
        default="src/rank/models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=2,
        help="Minimum features per combination"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum features per combination (default: all)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top models to save"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save models to disk"
    )

    args = parser.parse_args()

    trainer = XGBoostTrainer(
        db_path=args.db_path,
        model_output_dir=args.output_dir,
        min_features=args.min_features,
        max_features=args.max_features,
        leaderboard_capacity=args.top_k
    )

    trainer.run(save_models=not args.no_save)


if __name__ == "__main__":
    main()

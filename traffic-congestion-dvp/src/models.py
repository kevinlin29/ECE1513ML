"""
models.py — Model definitions for traffic speed prediction.

Provides factory functions that return configured model instances,
including a custom HistoricalAverageModel baseline, tree-based models
(RF, XGBoost, LightGBM), and a PyTorch neural network with learned
entity embeddings for location and direction.

Scope: Predict avg_speed on streets near U of T campus using City of
Toronto midblock speed bin data (2020-2025).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import torch
import torch.nn as nn


class HistoricalAverageModel(BaseEstimator, RegressorMixin):
    """Baseline model that predicts mean speed by (location_name, hour_of_day, day_of_week).

    Follows the scikit-learn estimator API so it can be used interchangeably
    with other regressors in training and evaluation pipelines.

    The model expects the input matrix ``X`` to contain columns named
    ``location_name``, ``hour_of_day``, and ``day_of_week``.  During
    prediction, any unseen group key falls back to the global training mean.
    """

    def __init__(self):
        self.averages_ = None
        self.global_mean_ = None
        self._group_cols = ["location_name", "hour_of_day", "day_of_week"]

    def fit(self, X, y):
        """Compute mean target value for each (location_name, hour_of_day, day_of_week) group.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.  Must contain columns ``location_name``,
            ``hour_of_day``, and ``day_of_week`` (or positional columns
            0, 1, 2 when an ndarray is passed).
        y : array-like
            Target values (avg_speed).

        Returns
        -------
        self
        """
        df = self._to_dataframe(X).copy()
        df["_target"] = np.asarray(y)

        self.global_mean_ = df["_target"].mean()
        self.averages_ = (
            df.groupby(self._group_cols)["_target"]
            .mean()
            .to_dict()
        )
        return self

    def predict(self, X):
        """Predict mean speed for each row based on its group key.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix with the same schema as in ``fit``.

        Returns
        -------
        np.ndarray
            Predicted avg_speed values.
        """
        df = self._to_dataframe(X)
        predictions = df.apply(
            lambda row: self.averages_.get(
                tuple(row[col] for col in self._group_cols),
                self.global_mean_,
            ),
            axis=1,
        )
        return predictions.values

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_dataframe(self, X):
        """Convert ``X`` to a DataFrame with the expected column names."""
        if isinstance(X, pd.DataFrame):
            return X
        # Assume the first three columns correspond to the group cols.
        df = pd.DataFrame(X)
        if set(self._group_cols).issubset(df.columns):
            return df
        # Map positional columns when X is a plain ndarray.
        rename = {i: col for i, col in enumerate(self._group_cols)}
        return df.rename(columns=rename)


# ----------------------------------------------------------------------
# Factory functions
# ----------------------------------------------------------------------

def get_baseline_model():
    """Return a HistoricalAverageModel instance.

    Returns
    -------
    HistoricalAverageModel
    """
    return HistoricalAverageModel()


def get_linear_regression():
    """Return a scikit-learn LinearRegression instance.

    Returns
    -------
    LinearRegression
    """
    return LinearRegression()


def get_random_forest(n_estimators=100):
    """Return a scikit-learn RandomForestRegressor.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.

    Returns
    -------
    RandomForestRegressor
    """
    return RandomForestRegressor(n_estimators=n_estimators, random_state=42)


def get_xgboost(n_estimators=100, learning_rate=0.1, device="cuda"):
    """Return an XGBRegressor.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.1
        Step size shrinkage.
    device : str, default="cuda"
        Device to train on ("cuda" for GPU, "cpu" for CPU).

    Returns
    -------
    XGBRegressor
    """
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42,
        device=device,
    )


def get_lightgbm_dart(n_estimators=500, max_depth=6, learning_rate=0.05):
    """Return a LightGBM regressor with DART boosting.

    DART (Dropouts meet Multiple Additive Regression Trees) uses dropout
    to prevent over-specialization of later trees, which often improves
    generalization on temporal splits.

    Parameters
    ----------
    n_estimators : int, default=500
    max_depth : int, default=6
    learning_rate : float, default=0.05

    Returns
    -------
    lgb.LGBMRegressor
    """
    return lgb.LGBMRegressor(
        boosting_type="dart",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        drop_rate=0.1,
        random_state=42,
        verbose=-1,
    )


# ----------------------------------------------------------------------
# PyTorch neural network with entity embeddings
# ----------------------------------------------------------------------

class _TrafficSpeedNet(nn.Module):
    """MLP with learned embeddings for categorical features (location, direction)."""

    def __init__(self, n_locations, n_directions, n_continuous,
                 loc_embed_dim=16, dir_embed_dim=4,
                 hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        self.loc_embed = nn.Embedding(n_locations, loc_embed_dim)
        self.dir_embed = nn.Embedding(n_directions, dir_embed_dim)

        input_dim = loc_embed_dim + dir_embed_dim + n_continuous
        layers = []
        for h in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, loc_ids, dir_ids, continuous):
        loc_emb = self.loc_embed(loc_ids)
        dir_emb = self.dir_embed(dir_ids)
        x = torch.cat([loc_emb, dir_emb, continuous], dim=1)
        return self.mlp(x).squeeze(-1)


class EmbeddingNeuralNet(BaseEstimator, RegressorMixin):
    """Sklearn-compatible PyTorch neural network with entity embeddings.

    Learns dense vector representations for location and direction,
    allowing the model to generalize to unseen locations based on
    learned spatial patterns rather than opaque integer IDs.

    Parameters
    ----------
    n_locations : int
        Number of unique locations (embedding table size).
    n_directions : int
        Number of unique directions (embedding table size).
    loc_embed_dim : int
        Dimensionality of location embeddings.
    dir_embed_dim : int
        Dimensionality of direction embeddings.
    hidden_dims : tuple of int
        Hidden layer sizes.
    dropout : float
        Dropout rate between layers.
    lr : float
        Learning rate for Adam optimizer.
    batch_size : int
        Mini-batch size for training.
    epochs : int
        Maximum number of training epochs.
    patience : int
        Early stopping patience (epochs without val improvement).
    device : str
        PyTorch device ("cuda" or "cpu").
    """

    # These column names are expected in the input DataFrame
    LOC_COL = "location_encoded"
    DIR_COL = "direction_encoded"

    def __init__(self, n_locations=200, n_directions=10,
                 loc_embed_dim=16, dir_embed_dim=4,
                 hidden_dims=(256, 128, 64), dropout=0.3,
                 lr=1e-3, batch_size=1024, epochs=100,
                 patience=10, device="cuda"):
        self.n_locations = n_locations
        self.n_directions = n_directions
        self.loc_embed_dim = loc_embed_dim
        self.dir_embed_dim = dir_embed_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device

    def _split_features(self, X):
        """Split X into (loc_ids, dir_ids, continuous) tensors."""
        if isinstance(X, pd.DataFrame):
            loc = torch.LongTensor(X[self.LOC_COL].values)
            dir_ = torch.LongTensor(X[self.DIR_COL].values)
            cont_cols = [c for c in X.columns
                         if c not in (self.LOC_COL, self.DIR_COL)]
            cont = torch.FloatTensor(X[cont_cols].values)
        else:
            X = np.asarray(X, dtype=np.float32)
            loc = torch.LongTensor(X[:, -2].astype(int))
            dir_ = torch.LongTensor(X[:, -1].astype(int))
            cont = torch.FloatTensor(X[:, :-2])
        return loc, dir_, cont

    def fit(self, X, y):
        from torch.utils.data import DataLoader, TensorDataset

        loc, dir_, cont = self._split_features(X)
        y_arr = np.asarray(y, dtype=np.float32)

        # Replace NaN with column median in continuous features
        cont_np = cont.numpy()
        for j in range(cont_np.shape[1]):
            mask = np.isnan(cont_np[:, j])
            if mask.any():
                cont_np[mask, j] = np.nanmedian(cont_np[:, j])
        cont = torch.FloatTensor(cont_np)

        # Normalize continuous features (robust: use median/IQR)
        self.cont_median_ = cont.median(dim=0).values
        q75 = cont.quantile(0.75, dim=0)
        q25 = cont.quantile(0.25, dim=0)
        self.cont_iqr_ = (q75 - q25).clamp(min=1e-3)
        cont = (cont - self.cont_median_) / self.cont_iqr_

        # Normalize target
        self.y_mean_ = float(np.nanmean(y_arr))
        self.y_std_ = max(float(np.nanstd(y_arr)), 1e-3)
        y_t = torch.FloatTensor((y_arr - self.y_mean_) / self.y_std_)

        n_cont = cont.shape[1]

        # 90/10 train/val split (chronological)
        n = len(y_t)
        n_val = max(1, int(n * 0.1))
        n_tr = n - n_val

        train_ds = TensorDataset(loc[:n_tr], dir_[:n_tr], cont[:n_tr], y_t[:n_tr])
        val_ds = TensorDataset(loc[n_tr:], dir_[n_tr:], cont[n_tr:], y_t[n_tr:])
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size * 2)

        self.model_ = _TrafficSpeedNet(
            self.n_locations, self.n_directions, n_cont,
            self.loc_embed_dim, self.dir_embed_dim,
            self.hidden_dims, self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        criterion = nn.HuberLoss(delta=1.0)

        best_val_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model_.train()
            for bl, bd, bc, by in train_dl:
                pred = self.model_(bl.to(self.device), bd.to(self.device),
                                   bc.to(self.device))
                loss = criterion(pred, by.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()

            self.model_.eval()
            vloss = []
            with torch.no_grad():
                for bl, bd, bc, by in val_dl:
                    pred = self.model_(bl.to(self.device), bd.to(self.device),
                                       bc.to(self.device))
                    vloss.append(criterion(pred, by.to(self.device)).item())
            vl = np.mean(vloss)
            scheduler.step(vl)

            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.cpu().clone()
                              for k, v in self.model_.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict(self, X):
        loc, dir_, cont = self._split_features(X)

        # Apply same NaN handling and normalization
        cont_np = cont.numpy()
        for j in range(cont_np.shape[1]):
            mask = np.isnan(cont_np[:, j])
            if mask.any():
                cont_np[mask, j] = float(self.cont_median_[j])
        cont = torch.FloatTensor(cont_np)
        cont = (cont - self.cont_median_) / self.cont_iqr_

        self.model_.eval()
        preds = []
        bs = self.batch_size * 2
        with torch.no_grad():
            for i in range(0, len(loc), bs):
                j = min(i + bs, len(loc))
                p = self.model_(loc[i:j].to(self.device),
                                dir_[i:j].to(self.device),
                                cont[i:j].to(self.device))
                preds.append(p.cpu().numpy())
        # Denormalize
        raw = np.concatenate(preds)
        return raw * self.y_std_ + self.y_mean_


# ----------------------------------------------------------------------
# Sequence models: LSTM and Transformer
# ----------------------------------------------------------------------

class _TrafficLSTM(nn.Module):
    """LSTM for time-series speed prediction from feature sequences."""

    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])  # last layer hidden state
        return out.squeeze(-1)


class _TrafficTransformer(nn.Module):
    """Transformer encoder for time-series speed prediction."""

    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2,
                 dropout=0.1, max_seq_len=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])  # last position
        return out.squeeze(-1)


class SequenceModelRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for LSTM and Transformer sequence models.

    Expects X to be a 3D numpy array of shape (n_samples, seq_len, n_features),
    as produced by ``preprocessing.create_sequences``.

    Parameters
    ----------
    model_type : str
        ``"lstm"`` or ``"transformer"``.
    hidden_dim : int
        Hidden size for LSTM / d_model for Transformer.
    n_layers : int
        Number of recurrent / encoder layers.
    n_heads : int
        Number of attention heads (Transformer only).
    dropout : float
        Dropout rate.
    lr : float
        Learning rate for Adam.
    batch_size : int
        Mini-batch size.
    epochs : int
        Max training epochs.
    patience : int
        Early stopping patience.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(self, model_type="lstm", hidden_dim=128, n_layers=2,
                 n_heads=4, dropout=0.2, lr=1e-3, batch_size=1024,
                 epochs=100, patience=10, device="cuda"):
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device

    def fit(self, X, y):
        from torch.utils.data import DataLoader, TensorDataset

        X_np = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        # Replace NaN with per-feature median
        for j in range(X_np.shape[-1]):
            col = X_np[:, :, j] if X_np.ndim == 3 else X_np[:, j]
            mask = np.isnan(col)
            if mask.any():
                col[mask] = np.nanmedian(col)

        X_t = torch.FloatTensor(X_np)
        input_dim = X_t.shape[2]

        # Robust normalization (median/IQR) per channel
        flat = X_t.reshape(-1, input_dim)
        self.feat_median_ = flat.median(dim=0).values
        q75 = flat.quantile(0.75, dim=0)
        q25 = flat.quantile(0.25, dim=0)
        self.feat_iqr_ = (q75 - q25).clamp(min=1e-3)
        X_t = (X_t - self.feat_median_) / self.feat_iqr_

        # Normalize target
        self.y_mean_ = float(np.nanmean(y_arr))
        self.y_std_ = max(float(np.nanstd(y_arr)), 1e-3)
        y_t = torch.FloatTensor((y_arr - self.y_mean_) / self.y_std_)

        # 90/10 chronological split
        n = len(y_t)
        n_val = max(1, int(n * 0.1))
        n_tr = n - n_val

        train_dl = DataLoader(TensorDataset(X_t[:n_tr], y_t[:n_tr]),
                              batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(TensorDataset(X_t[n_tr:], y_t[n_tr:]),
                            batch_size=self.batch_size * 2)

        if self.model_type == "lstm":
            self.model_ = _TrafficLSTM(
                input_dim, self.hidden_dim, self.n_layers, self.dropout,
            ).to(self.device)
        else:
            self.model_ = _TrafficTransformer(
                input_dim, d_model=self.hidden_dim, n_heads=self.n_heads,
                n_layers=self.n_layers, dropout=self.dropout,
            ).to(self.device)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        criterion = nn.HuberLoss(delta=1.0)

        best_val_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model_.train()
            for bx, by in train_dl:
                pred = self.model_(bx.to(self.device))
                loss = criterion(pred, by.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()

            self.model_.eval()
            vloss = []
            with torch.no_grad():
                for bx, by in val_dl:
                    pred = self.model_(bx.to(self.device))
                    vloss.append(criterion(pred, by.to(self.device)).item())
            vl = np.mean(vloss)
            scheduler.step(vl)

            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.cpu().clone()
                              for k, v in self.model_.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict(self, X):
        X_np = np.asarray(X, dtype=np.float32)
        for j in range(X_np.shape[-1]):
            col = X_np[:, :, j] if X_np.ndim == 3 else X_np[:, j]
            mask = np.isnan(col)
            if mask.any():
                col[mask] = float(self.feat_median_[j])

        X_t = torch.FloatTensor(X_np)
        X_t = (X_t - self.feat_median_) / self.feat_iqr_

        self.model_.eval()
        preds = []
        bs = self.batch_size * 2
        with torch.no_grad():
            for i in range(0, len(X_t), bs):
                p = self.model_(X_t[i:i + bs].to(self.device))
                preds.append(p.cpu().numpy())
        raw = np.concatenate(preds)
        return raw * self.y_std_ + self.y_mean_

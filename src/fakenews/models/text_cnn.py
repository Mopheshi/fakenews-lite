"""Kim-style convolutional network over frozen GloVe embeddings."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from fakenews.config import TextCNNConfig
from fakenews.models.base import BaseClassifier

logger = logging.getLogger(__name__)


class TextCNNModel(BaseClassifier):
    """Parallel convolution branches with global max pooling."""

    name = "TextCNN"

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        *,
        embedding_dim: int = 100,
        embedding_matrix: np.ndarray | None = None,
        config: TextCNNConfig | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.config = config or TextCNNConfig()
        self.model: Model | None = None
        self.history = None
        self.train_seconds: float | None = None
        self.best_epoch: int | None = None

    def build(self) -> Model:
        inputs = Input(shape=(self.max_len,), name="tokens")

        if self.embedding_matrix is not None:
            embedding = Embedding(
                self.vocab_size,
                self.embedding_dim,
                weights=[self.embedding_matrix],
                trainable=not self.config.freeze_embeddings,
                name="glove_embedding",
            )(inputs)
            logger.info(
                "Embedding: pre-trained GloVe (%s)",
                "frozen" if self.config.freeze_embeddings else "fine-tuned",
            )
        else:
            embedding = Embedding(self.vocab_size, self.embedding_dim, name="embedding")(inputs)
            logger.info("Embedding: randomly initialised and trainable")

        branches = []
        for filter_size in self.config.filter_sizes:
            conv = Conv1D(
                filters=self.config.num_filters,
                kernel_size=filter_size,
                activation="relu",
                name=f"conv_{filter_size}gram",
            )(embedding)
            branches.append(GlobalMaxPooling1D(name=f"pool_{filter_size}gram")(conv))

        merged = Concatenate(name="concat")(branches) if len(branches) > 1 else branches[0]
        dense = Dense(self.config.dense_units, activation="relu", name="dense")(merged)
        dropped = Dropout(self.config.dropout_rate, name="dropout")(dense)
        outputs = Dense(1, activation="sigmoid", name="p_fake")(dropped)

        self.model = Model(inputs=inputs, outputs=outputs, name="text_cnn")
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return self.model

    def fit(self, X_train, y_train, X_val, y_val, *, verbose: int = 1) -> TextCNNModel:
        if self.model is None:
            self.build()

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=verbose,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=verbose,
        )

        started = time.perf_counter()
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose,
        )
        self.train_seconds = time.perf_counter() - started

        val_losses = self.history.history["val_loss"]
        self.best_epoch = int(np.argmin(val_losses)) + 1
        logger.info(
            "TextCNN trained in %.1f s; best epoch %d of %d",
            self.train_seconds,
            self.best_epoch,
            len(val_losses),
        )
        return self

    def predict_proba(self, X, *, batch_size: int = 256) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been built")
        return self.model.predict(X, batch_size=batch_size, verbose=0).ravel()

    def save(self, directory: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been built")
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "text_cnn.keras")

    @classmethod
    def load(cls, directory: Path, *, max_len: int, vocab_size: int) -> TextCNNModel:
        instance = cls(vocab_size=vocab_size, max_len=max_len)
        instance.model = load_model(Path(directory) / "text_cnn.keras")
        return instance

    @property
    def parameter_counts(self) -> dict[str, int]:
        """Trainable, frozen and total weights.

        The embedding matrix dominates the total but is frozen GloVe, learned by
        someone else and never updated here, so it is reported separately rather
        than folded into a single headline figure.
        """
        if self.model is None:
            raise RuntimeError("Model has not been built")

        trainable = int(sum(np.prod(w.shape) for w in self.model.trainable_weights))
        non_trainable = int(sum(np.prod(w.shape) for w in self.model.non_trainable_weights))
        return {
            "trainable": trainable,
            "frozen": non_trainable,
            "total": trainable + non_trainable,
        }

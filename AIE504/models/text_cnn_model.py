from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class TextCNNModel:
    """TextCNN model with GloVe embeddings for fake news detection"""

    def __init__(self, vocab_size, embedding_dim, max_len, embedding_matrix=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.model = None

    def build_model(self, filter_sizes=[3, 4, 5], num_filters=128, dropout_rate=0.5):
        print("\nBUILDING TextCNN MODEL...")
        print("-" * 80)

        # Input layer
        input_layer = Input(shape=(self.max_len,))

        # Embedding layer
        if self.embedding_matrix is not None:
            embedding = Embedding(self.vocab_size, self.embedding_dim, weights=[self.embedding_matrix],
                                  input_length=self.max_len, trainable=False  # Freeze pre-trained embeddings
                                  )(input_layer)
            print("  Using pre-trained GloVe embeddings (frozen)")
        else:
            embedding = Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len)(input_layer)
            print("  Using trainable embeddings")

        # Multiple convolutional layers with different kernel sizes
        conv_blocks = []
        for filter_size in filter_sizes:
            conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(embedding)
            pool = GlobalMaxPooling1D()(conv)
            conv_blocks.append(pool)

        # Concatenate all features
        concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        # Dense layers
        dense = Dense(128, activation='relu')(concat)
        dropout = Dropout(dropout_rate)(dense)

        # Output layer
        output = Dense(1, activation='sigmoid')(dropout)

        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        print(f"\n  Model architecture:")
        print(f"  - Embedding: {self.vocab_size} x {self.embedding_dim}")
        print(f"  - Conv1D filters: {filter_sizes} (each with {num_filters} filters)")
        print(f"  - Dense layer: 128 units")
        print(f"  - Output: sigmoid activation")

        return self.model

    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=128):
        """Train the TextCNN model"""
        print("\n[7] TRAINING TextCNN MODEL...")
        print("-" * 80)

        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)

        # Train model
        history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size,
                                 callbacks=[early_stop, reduce_lr], verbose=1)

        print("\n  Training complete")

        return history

    def predict(self, x):
        predictions = self.model.predict(x, verbose=0)
        return (predictions > 0.5).astype(int).flatten()

    def predict_probabilities(self, x):
        return self.model.predict(x, verbose=0).flatten()

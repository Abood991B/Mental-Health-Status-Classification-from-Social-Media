from __future__ import annotations


def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for BiLSTM, CNN, and GRU models. "
            "Install tensorflow or run the notebook in Google Colab."
        ) from exc
    return tf


def build_bilstm_model(vocab_size: int, embedding_dim: int, num_classes: int, embedding_matrix=None, max_len: int = 200):
    tf = require_tensorflow()
    layers = tf.keras.layers
    embedding = layers.Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix] if embedding_matrix is not None else None,
        trainable=False if embedding_matrix is not None else True,
    )
    model = tf.keras.Sequential(
        [
            embedding,
            layers.Bidirectional(layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(64, dropout=0.3)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_model(vocab_size: int, embedding_dim: int, num_classes: int, embedding_matrix=None, max_len: int = 200):
    tf = require_tensorflow()
    layers = tf.keras.layers
    embedding = layers.Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix] if embedding_matrix is not None else None,
        trainable=False if embedding_matrix is not None else True,
    )
    model = tf.keras.Sequential(
        [
            embedding,
            layers.Conv1D(filters=128, kernel_size=5, activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_gru_model(vocab_size: int, embedding_dim: int, num_classes: int, embedding_matrix=None, max_len: int = 200):
    tf = require_tensorflow()
    layers = tf.keras.layers
    embedding = layers.Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix] if embedding_matrix is not None else None,
        trainable=False if embedding_matrix is not None else True,
    )
    model = tf.keras.Sequential(
        [
            embedding,
            layers.GRU(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True),
            layers.GRU(64, dropout=0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

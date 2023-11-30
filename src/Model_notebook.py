import itertools
import re
import time

import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras.layers import (
    LSTM,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Input,
    SpatialDropout1D,
)
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.env_variable_settings import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    EPOCHS,
    GLOVE_EMB,
    LR,
    MAX_SEQUENCE_LENGTH,
    TRAIN_SIZE,
)

matplotlib.use("TkAgg")
nltk.download("stopwords")
# Print Tensorflow version
print("Tensorflow Version", tf.__version__)


def label_decoder(label):
    lab_to_sentiment = {0: "Negative", 4: "Positive"}
    return lab_to_sentiment[label]


def load_data_and_preprocess():
    # Load the data
    df = pd.read_csv(
        "/tmp/dataset/training.1600000.processed.noemoticon.csv",
        encoding="latin",
        header=None,
    )
    df.columns = ["sentiment", "id", "date", "query", "user_id", "text"]
    df = df.drop(["id", "date", "query", "user_id"], axis=1)
    df.sentiment = df.sentiment.apply(lambda x: label_decoder(x))
    return df


def preprocess(text, stem=False):
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    text_cleaning_re = r"@\S+|https?://\S+|http?://\S+|[^A-Za-z0-9]+"
    text = re.sub(text_cleaning_re, " ", str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


def perform_data_cleaning(df):
    df.text = df.text.apply(lambda x: preprocess(x))
    return df


def split_train_test(df):
    train_data, test_data = train_test_split(
        df, test_size=1 - TRAIN_SIZE, random_state=7
    )  # Splits Dataset into Training and Testing set
    print("Train Data size:", len(train_data))
    print("Test Data size", len(test_data))
    return train_data, test_data


def perform_tokenizer(train_data, test_data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data.text)

    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary Size :", vocab_size)

    x_train = pad_sequences(
        tokenizer.texts_to_sequences(train_data.text), maxlen=MAX_SEQUENCE_LENGTH
    )
    x_test = pad_sequences(
        tokenizer.texts_to_sequences(test_data.text), maxlen=MAX_SEQUENCE_LENGTH
    )

    print("Training X Shape:", x_train.shape)
    print("Testing X Shape:", x_test.shape)

    return word_index, vocab_size, x_train, x_test


def perform_label_encoder(train_data, test_data):
    encoder = LabelEncoder()
    encoder.fit(train_data.sentiment.to_list())

    y_train = encoder.transform(train_data.sentiment.to_list())
    y_test = encoder.transform(test_data.sentiment.to_list())

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return y_train, y_test


def create_embedding_layer(vocab_size, word_index):
    embeddings_index = {}

    f = open(GLOVE_EMB)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()

    print("Found %s word vectors." % len(embeddings_index))

    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = tf.keras.layers.Embedding(
        vocab_size,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False,
    )

    return embedding_layer


def build_lstm_model(embedding_layer):
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedding_sequences = embedding_layer(sequence_input)
    x = SpatialDropout1D(0.2)(embedding_sequences)
    x = Conv1D(64, 5, activation="relu")(x)
    x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(sequence_input, outputs)
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    reduceLRonPlateau = ReduceLROnPlateau(
        factor=0.1, min_lr=0.01, monitor="val_loss", verbose=1
    )
    return model, reduceLRonPlateau


def start_training(model, reduceLRonPlateau, x_train, y_train, x_test, y_test):
    print("Training on GPU...") if tf.test.is_gpu_available() else print(
        "Training on CPU..."
    )

    start_time = time.time()  # Start measuring the time

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=[reduceLRonPlateau],
    )

    end_time = time.time()  # Stop measuring the time
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    print("Model building time:", elapsed_time, "seconds")

    return history


def print_accuracy(history):
    s, (at, al) = plt.subplots(2, 1)
    at.plot(history.history["accuracy"], c="b")
    at.plot(history.history["val_accuracy"], c="r")
    at.set_title("model accuracy")
    at.set_ylabel("accuracy")
    at.set_xlabel("epoch")
    at.legend(["LSTM_train", "LSTM_val"], loc="upper left")

    al.plot(history.history["loss"], c="m")
    al.plot(history.history["val_loss"], c="c")
    al.set_title("model loss")
    al.set_ylabel("loss")
    al.set_xlabel("epoch")
    al.legend(["train", "val"], loc="upper left")
    plt.tight_layout()
    plt.show()


def decode_sentiment(score):
    return "Positive" if score > 0.5 else "Negative"


def predict(model, x_test):
    scores = model.predict(x_test, verbose=1, batch_size=10000)
    y_pred_1d = [decode_sentiment(score) for score in scores]
    return y_pred_1d


def plot_confusion_matrix(
    cm,
    classes,
    test_data,
    y_pred_1d,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=13)
    plt.yticks(tick_marks, classes, fontsize=13)

    fmt = ".2f"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label", fontsize=17)
    plt.xlabel("Predicted label", fontsize=17)

    plt.figure(figsize=(6, 6))
    plt.show()


if __name__ == "__main__":
    print("Loading and Preprocessing Data...")
    df = load_data_and_preprocess()
    df = perform_data_cleaning(df)

    print("Splitting Data into Train and Test sets...")
    train_data, test_data = split_train_test(df)

    print("Tokenizing and Padding Data...")
    word_index, vocab_size, x_train, x_test = perform_tokenizer(train_data, test_data)
    y_train, y_test = perform_label_encoder(train_data, test_data)

    print("Creating Embedding Layer...")
    embedding_layer = create_embedding_layer(vocab_size, word_index)

    print("Building LSTM Model...")
    model, reduceLRonPlateau = build_lstm_model(embedding_layer)

    print("Training LSTM Model...")
    history = start_training(model, reduceLRonPlateau, x_train, y_train, x_test, y_test)
    print("Accuracy graph")
    print_accuracy(history)
    y_pred_1d = predict(model, x_test)
    cnf_matrix = confusion_matrix(test_data.sentiment.to_list(), y_pred_1d)
    plot_confusion_matrix(
        cm=cnf_matrix,
        classes=test_data.sentiment.unique(),
        test_data=test_data,
        y_pred_1d=y_pred_1d,
        title="Confusion matrix",
    )
    print(classification_report(list(test_data.sentiment), y_pred_1d))

import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from rnn.data import train_data, test_data

if not os.path.exists('images'):
    os.makedirs('images')


def prepare_data(data):
    sentences = list(data.keys())
    labels = [1 if sentiment else 0 for sentiment in data.values()]
    return sentences, labels


def main():
    train_sentences, train_labels = prepare_data(train_data)
    test_sentences, test_labels = prepare_data(test_data)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sentences + test_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)

    max_length = max(max(len(x) for x in train_sequences), max(len(x) for x in test_sequences))
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

    vocab_size = len(tokenizer.word_index) + 1

    print(f"Vocabulary size: {vocab_size}")
    print(f"Training examples: {len(train_sentences)}")
    print(f"Testing examples: {len(test_sentences)}")

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=16, input_length=max_length),
        SimpleRNN(32, return_sequences=False),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    print("\nTraining Library RNN model...")
    history = model.fit(
        train_padded, np.array(train_labels),
        epochs=100,
        validation_split=0.2,
        verbose=1
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('images/library_rnn_training.png')

    train_predictions = (model.predict(train_padded) > 0.5).astype(int).flatten()
    train_accuracy = accuracy_score(train_labels, train_predictions)
    train_precision = precision_score(train_labels, train_predictions)
    train_recall = recall_score(train_labels, train_predictions)
    train_f1 = f1_score(train_labels, train_predictions)
    train_cm = confusion_matrix(train_labels, train_predictions)

    test_predictions = (model.predict(test_padded) > 0.5).astype(int).flatten()
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_precision = precision_score(test_labels, test_predictions)
    test_recall = recall_score(test_labels, test_predictions)
    test_f1 = f1_score(test_labels, test_predictions)
    test_cm = confusion_matrix(test_labels, test_predictions)

    print("\nLibrary RNN Model Performance:")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Train F1-Score: {train_f1:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")

    print("\nConfusion Matrix (Training):")
    print(f"True Negative: {train_cm[0][0]}, False Positive: {train_cm[0][1]}")
    print(f"False Negative: {train_cm[1][0]}, True Positive: {train_cm[1][1]}")

    print("\nConfusion Matrix (Testing):")
    print(f"True Negative: {test_cm[0][0]}, False Positive: {test_cm[0][1]}")
    print(f"False Negative: {test_cm[1][0]}, True Positive: {test_cm[1][1]}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(train_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Training)')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = train_cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, train_cm[i, j],
                     horizontalalignment="center",
                     color="white" if train_cm[i, j] > thresh else "black")

    plt.subplot(1, 2, 2)
    plt.imshow(test_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Testing)')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = test_cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, test_cm[i, j],
                     horizontalalignment="center",
                     color="white" if test_cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig('images/library_rnn_confusion_matrix.png')

    with open('metrics.txt', 'a') as f:
        f.write(f"Library RNN Train Accuracy: {train_accuracy:.6f}\n")
        f.write(f"Library RNN Test Accuracy: {test_accuracy:.6f}\n")
        f.write(f"Library RNN Train F1-Score: {train_f1:.6f}\n")
        f.write(f"Library RNN Test F1-Score: {test_f1:.6f}\n")
        f.write(f"Library RNN Train Precision: {train_precision:.6f}\n")
        f.write(f"Library RNN Test Precision: {test_precision:.6f}\n")
        f.write(f"Library RNN Train Recall: {train_recall:.6f}\n")
        f.write(f"Library RNN Test Recall: {test_recall:.6f}\n")

    print("\nModel Comparison:")
    with open('metrics.txt', 'r') as f:
        print(f.read())


if __name__ == "__main__":
    main()

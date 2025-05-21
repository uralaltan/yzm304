import numpy as np
import matplotlib.pyplot as plt
import os
from rnn.data import train_data, test_data

if not os.path.exists('images'):
    os.makedirs('images')


def preprocess_data(data, existing_vocab=None, existing_word_to_idx=None):
    if existing_vocab is None or existing_word_to_idx is None:
        vocab_set = set()
        for sentence in data.keys():
            for word in sentence.split():
                vocab_set.add(word)
        vocab = list(sorted(vocab_set))
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        is_training = True
    else:
        vocab = existing_vocab
        word_to_idx = existing_word_to_idx
        is_training = False

    vocab_size = len(vocab)
    X = []
    y = []
    skipped_sentences = 0
    for sentence, sentiment in data.items():
        words = sentence.split()
        known_word_indices = [word_to_idx[word] for word in words if word in word_to_idx]

        if not known_word_indices:
            skipped_sentences += 1
            continue

        sequence = np.zeros((len(known_word_indices), vocab_size))
        for t, index in enumerate(known_word_indices):
            sequence[t, index] = 1

        X.append(sequence)
        y.append(1 if sentiment else 0)

    if skipped_sentences > 0 and not is_training:
        print(f"Skipped {skipped_sentences} sentences containing only OOV words during non-training preprocessing.")

    if is_training:
        return X, y, vocab, word_to_idx
    else:
        return X, y


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        scale_Wxh = np.sqrt(1.0 / input_size)
        scale_Whh = np.sqrt(1.0 / hidden_size)
        scale_Why = np.sqrt(1.0 / hidden_size)

        self.Wxh = np.random.randn(hidden_size, input_size) * scale_Wxh
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale_Whh
        self.Why = np.random.randn(output_size, hidden_size) * scale_Why

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        self.parameters = {'Wxh': self.Wxh, 'Whh': self.Whh, 'Why': self.Why, 'bh': self.bh, 'by': self.by}

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))
        self.x_list, self.h_list = [], []
        self.h_list.append(h.copy())

        if inputs.shape[0] == 0:
            return np.array([[0.5]])

        for x_t in inputs:
            x = x_t.reshape(-1, 1)
            self.x_list.append(x)
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            self.h_list.append(h)

        y_logit = np.dot(self.Why, h) + self.by
        y = sigmoid(y_logit)

        return y

    def backward(self, dy, learning_rate=0.1):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        h_last = self.h_list[-1]
        dWhy += np.dot(dy, h_last.T)
        dby += dy

        dh_next = np.dot(self.Why.T, dy)

        for t in reversed(range(len(self.x_list))):
            h_t = self.h_list[t + 1]
            h_prev = self.h_list[t]
            x_t = self.x_list[t]

            dtanh_input = dh_next * (1 - h_t * h_t)

            dbh += dtanh_input
            dWxh += np.dot(dtanh_input, x_t.T)
            dWhh += np.dot(dtanh_input, h_prev.T)

            dh_next = np.dot(self.Whh.T, dtanh_input)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

    def train(self, X, y, learning_rate=0.01, epochs=100):
        losses = []
        accuracies = []
        num_examples = len(X)

        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0

            indices = np.random.permutation(num_examples)
            X_shuffled = [X[i] for i in indices]
            y_shuffled = [y[i] for i in indices]

            valid_examples_processed_count = 0
            for i in range(num_examples):
                current_x = X_shuffled[i]
                current_y_target = y_shuffled[i]

                if current_x.shape[0] == 0:
                    continue

                valid_examples_processed_count += 1
                y_pred = self.forward(current_x)
                y_pred_scalar = y_pred.item()

                prediction = 1 if y_pred_scalar > 0.5 else 0
                if prediction == current_y_target:
                    correct_predictions += 1

                epsilon = 1e-9
                if current_y_target == 1:
                    loss = -np.log(y_pred_scalar + epsilon)
                else:
                    loss = -np.log(1 - y_pred_scalar + epsilon)
                total_loss += loss

                dy = y_pred - current_y_target

                self.backward(dy, learning_rate)

            if valid_examples_processed_count > 0:
                avg_loss = total_loss / valid_examples_processed_count
                accuracy = correct_predictions / valid_examples_processed_count
            else:
                avg_loss = 0
                accuracy = 0

            losses.append(avg_loss)
            accuracies.append(accuracy)

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

        return losses, accuracies

    def predict(self, X):
        predictions = []
        for inputs in X:
            if inputs.shape[0] == 0:
                predictions.append(0)
                continue

            y_pred = self.forward(inputs)
            prediction = 1 if y_pred.item() > 0.5 else 0
            predictions.append(prediction)
        return predictions


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0,
            'confusion_matrix': {'true_positive': 0, 'false_positive': 0, 'true_negative': 0, 'false_negative': 0}
        }

    if len(y_true) != len(y_pred):
        print(f"Error in calculate_metrics: y_true ({len(y_true)}) and y_pred ({len(y_pred)}) have different lengths.")
        return {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0,
            'confusion_matrix': {'true_positive': 0, 'false_positive': 0, 'true_negative': 0, 'false_negative': 0}
        }

    accuracy = sum(y_true[i] == y_pred[i] for i in range(len(y_true))) / len(y_true)

    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    confusion_matrix = {
        'true_positive': tp,
        'false_positive': fp,
        'true_negative': tn,
        'false_negative': fn
    }

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix
    }


def main():
    print("Processing training data...")
    X_train, y_train, vocab, word_to_idx = preprocess_data(train_data)

    print("\nProcessing testing data...")
    X_test, y_test = preprocess_data(test_data, existing_vocab=vocab, existing_word_to_idx=word_to_idx)

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Training examples: {len(X_train)}")
    print(f"Testing examples (after OOV filtering): {len(X_test)}")

    if not X_test:
        print("Error: Test set is empty after removing sentences with only OOV words. Cannot proceed with evaluation.")
        return

    input_size = len(vocab)
    hidden_size = 32
    output_size = 1

    rnn = RNN(input_size, hidden_size, output_size)

    print("\nTraining custom RNN model...")
    losses, accuracies = rnn.train(X_train, y_train, learning_rate=0.01, epochs=100)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/custom_rnn_training.png')
    plt.close()

    print("\nEvaluating model...")
    train_predictions = rnn.predict(X_train)
    test_predictions = rnn.predict(X_test)

    train_metrics = calculate_metrics(y_train, train_predictions)
    test_metrics = calculate_metrics(y_test, test_predictions)

    print("\nCustom RNN Model Performance:")
    print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Train F1-Score: {train_metrics['f1_score']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"Train Precision: {train_metrics['precision']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Train Recall: {train_metrics['recall']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")

    cm_train = train_metrics['confusion_matrix']
    print("\nConfusion Matrix (Training):")
    print(f"  Predicted:")
    print(f"  Neg Pos")
    print(f"T Neg {cm_train['true_negative']:<3} {cm_train['false_positive']:<3}")
    print(f"r Pos {cm_train['false_negative']:<3} {cm_train['true_positive']:<3}")

    cm_test = test_metrics['confusion_matrix']
    print("\nConfusion Matrix (Testing):")
    print(f"  Predicted:")
    print(f"  Neg Pos")
    print(f"T Neg {cm_test['true_negative']:<3} {cm_test['false_positive']:<3}")
    print(f"r Pos {cm_test['false_negative']:<3} {cm_test['true_positive']:<3}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    cm_train_array = np.array([[cm_train['true_negative'], cm_train['false_positive']],
                               [cm_train['false_negative'], cm_train['true_positive']]])
    plt.imshow(cm_train_array, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Training)')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'], rotation=90, va='center')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    thresh = cm_train_array.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm_train_array[i, j],
                     horizontalalignment="center",
                     color="white" if cm_train_array[i, j] > thresh else "black")

    plt.subplot(1, 2, 2)
    cm_test_array = np.array([[cm_test['true_negative'], cm_test['false_positive']],
                              [cm_test['false_negative'], cm_test['true_positive']]])
    plt.imshow(cm_test_array, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Testing)')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'], rotation=90, va='center')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    thresh = cm_test_array.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm_test_array[i, j],
                     horizontalalignment="center",
                     color="white" if cm_test_array[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig('images/custom_rnn_confusion_matrix.png')
    plt.close()

    print("\nWriting metrics to metrics.txt...")
    with open('metrics.txt', 'w') as f:
        f.write(f"Custom RNN Train Accuracy: {train_metrics['accuracy']:.6f}\n")
        f.write(f"Custom RNN Test Accuracy: {test_metrics['accuracy']:.6f}\n")
        f.write(f"Custom RNN Train F1-Score: {train_metrics['f1_score']:.6f}\n")
        f.write(f"Custom RNN Test F1-Score: {test_metrics['f1_score']:.6f}\n")
        f.write(f"Custom RNN Train Precision: {train_metrics['precision']:.6f}\n")
        f.write(f"Custom RNN Test Precision: {test_metrics['precision']:.6f}\n")
        f.write(f"Custom RNN Train Recall: {train_metrics['recall']:.6f}\n")
        f.write(f"Custom RNN Test Recall: {test_metrics['recall']:.6f}\n\n")
        f.write(f"Confusion Matrix (Training):\n")
        f.write(f"TN={cm_train['true_negative']}, FP={cm_train['false_positive']}\n")
        f.write(f"FN={cm_train['false_negative']}, TP={cm_train['true_positive']}\n\n")
        f.write(f"Confusion Matrix (Testing):\n")
        f.write(f"TN={cm_test['true_negative']}, FP={cm_test['false_positive']}\n")
        f.write(f"FN={cm_test['false_negative']}, TP={cm_test['true_positive']}\n")

    print("\nScript finished.")


if __name__ == "__main__":
    main()

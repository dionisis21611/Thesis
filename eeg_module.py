from sklearn.model_selection import KFold, ShuffleSplit
from keras.regularizers import l1_l2
from keras.layers import GRU, Bidirectional, MaxPooling1D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, LSTM, TimeDistributed, GlobalAveragePooling1D, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class EEGExperiment:
    def __init__(self, csv_file):
        self.eeg_data, self.labels = self.read_eeg_data(csv_file)
        self.X, self.y = self.preprocess_eeg_data(self.eeg_data, self.labels)

    @staticmethod
    def read_eeg_data(csv_file):
        df = pd.read_csv(csv_file)
        eeg_data = df.drop(columns=['label'])
        labels = df['label']
        return eeg_data, labels

    @staticmethod
    def preprocess_eeg_data(X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_3D = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)

        return X_3D, y_categorical

    def compute_statistics(self):
        mean = self.eeg_data.mean()
        std_dev = self.eeg_data.std()
        median = self.eeg_data.median()
        minimum = self.eeg_data.min()
        maximum = self.eeg_data.max()

        statistics = pd.DataFrame({
            'Mean': mean,
            'Standard Deviation': std_dev,
            'Median': median,
            'Minimum': minimum,
            'Maximum': maximum
        })

        return statistics

    def train_model(self, epochs=50, batch_size=32, model_type='cnn_lstm'):
        # Automatically detect the number of input channels
        input_channels = 1 if len(self.X.shape) == 2 else self.X.shape[2]

        # Automatically detect the number of classes
        num_classes = self.y.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)

        if model_type == 'cnn_lstm':
            self.model = Sequential([
                Conv1D(64, kernel_size=3, activation='relu',
                       input_shape=(X_train.shape[1], input_channels)),
                Conv1D(32, kernel_size=3, activation='relu'),
                LSTM(32, return_sequences=True),
                TimeDistributed(Dense(32, activation='relu')),
                GlobalAveragePooling1D(),
                Dense(num_classes, activation='softmax')
            ])

    def train_model2(self, epochs=100, batch_size=32, model_type='cnn_lstm', num_folds=5):
        # Automatically detect the number of input channels
        input_channels = 1 if len(self.X.shape) == 2 else self.X.shape[2]

        # Automatically detect the number of classes
        num_classes = self.y.shape[1]

        if num_folds == 1:
            cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        else:
            cv = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        # Iterate through the folds
        for fold, (train_index, test_index) in enumerate(cv.split(self.X)):
            print(f"Processing fold {fold + 1}/{num_folds}...")
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            if model_type == 'cnn_lstm':
                self.model = Sequential([
                    Conv1D(64, kernel_size=3, activation='relu',
                           input_shape=(X_train.shape[1], input_channels)),
                    Conv1D(32, kernel_size=3, activation='relu'),
                    LSTM(32, return_sequences=True),
                    TimeDistributed(Dense(32, activation='relu',
                                    kernel_regularizer=l1_l2(l1=0.01, l2=0.01))),
                    GlobalAveragePooling1D(),
                    Dense(num_classes, activation='softmax')
                ])
            elif model_type == 'cnn_gru':
                self.model = Sequential([
                    Conv1D(64, kernel_size=3, activation='relu',
                           input_shape=(X_train.shape[1], input_channels)),
                    Conv1D(32, kernel_size=3, activation='relu'),
                    GRU(32, return_sequences=True),
                    TimeDistributed(Dense(32, activation='relu',
                                    kernel_regularizer=l1_l2(l1=0.01, l2=0.01))),
                    GlobalAveragePooling1D(),
                    Dense(num_classes, activation='softmax')
                ])
            elif model_type == 'cnn_bi_lstm':
                self.model = Sequential([
                    Conv1D(64, kernel_size=3, activation='relu',
                           input_shape=(X_train.shape[1], input_channels)),
                    Conv1D(32, kernel_size=3, activation='relu'),
                    Bidirectional(LSTM(32, return_sequences=True)),
                    TimeDistributed(Dense(32, activation='relu',
                                    kernel_regularizer=l1_l2(l1=0.01, l2=0.01))),
                    GlobalAveragePooling1D(),
                    Dense(num_classes, activation='softmax')
                ])
            elif model_type == 'cnn_lstm_2':
                self.model = Sequential()
                # Input Layer
                self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(
                    X_train.shape[1], input_channels)))
                self.model.add(MaxPooling1D(pool_size=2))
                self.model.add(Dropout(0.2))
                # Hidden Layer
                self.model.add(
                    Conv1D(filters=64, kernel_size=3, activation='relu'))
                self.model.add(MaxPooling1D(pool_size=2))
                self.model.add(Dropout(0.2))
                # LSTM Layer
                self.model.add(LSTM(100, return_sequences=False))
                # Output Layer
                self.model.add(Dense(num_classes, activation='softmax'))

            self.model.compile(
                optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            self.history = self.model.fit(
                X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
            print(
                f"Fold {fold + 1} Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # Clip gradients with magnitude larger than 1.0
        # Gradient Clipping: To prevent exploding gradients,
        # you can apply gradient clipping to your optimizer. This limits the maximum gradient value.
        # optimizer = Adam(clipvalue=1.0)
        # self.model.compile(optimizer='adam',
        #                    loss='categorical_crossentropy', metrics=['accuracy'])

        # self.history = self.model.fit(X_train, y_train, epochs=epochs,
        #                               batch_size=batch_size, validation_data=(X_test, y_test))

    def plot_history(self):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'],
                 label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def evaluate_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        precision = precision_score(
            y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(
            y_test_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")


def example():

    # eeg_experiment = EEGExperiment("emotions.csv")
    eeg_experiment = EEGExperiment("gameemo_preprocessed.csv")
    eeg_experiment.train_model2(
        model_type='cnn_lstm_2', num_folds=1, epochs=50)
    eeg_experiment.plot_history()
    eeg_experiment.evaluate_model()

    statistics = eeg_experiment.compute_statistics()
    print("Statistical characteristics of the EEG signals:")
    print(statistics)

    eeg_experiment.plot_history()


if __name__ == '__main__':
    example()

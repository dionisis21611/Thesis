import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, LSTM, TimeDistributed, GlobalAveragePooling1D
from keras.utils import to_categorical

data = pd.read_csv(
    "emotions.csv")

# Load  dataset and extract features (X) and labels (y)
X = data.drop(["label"], axis=1)
y = data.loc[:, 'label'].values

# Preprocessing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the data to be 3D (samples, timesteps, features) for the 1D CNN
X_3D = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X_3D, y, test_size=0.3, random_state=42)

# Encoding the labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

# Building the model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu',
           input_shape=(X_train.shape[1], 1)),
    Conv1D(32, kernel_size=3, activation='relu'),
    LSTM(32, return_sequences=True),
    TimeDistributed(Dense(32, activation='relu')),
    GlobalAveragePooling1D(),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train_categorical, epochs=50,
                    batch_size=32, validation_data=(X_test, y_test_categorical))

# Plotting the training history


def plot_history(history):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


plot_history(history)

# Evaluating the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

#  Test Accuracy: 0.964062511920929

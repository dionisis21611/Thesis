import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_eeg_data(csv_file):
    df = pd.read_csv(csv_file)
    eeg_data = df.drop(columns=['label'])
    labels = df['label']
    return eeg_data, labels


def compute_statistics(eeg_data):
    mean = eeg_data.mean()
    std_dev = eeg_data.std()
    median = eeg_data.median()
    minimum = eeg_data.min()
    maximum = eeg_data.max()

    statistics = pd.DataFrame({
        'Mean': mean,
        'Standard Deviation': std_dev,
        'Median': median,
        'Minimum': minimum,
        'Maximum': maximum
    })

    return statistics


if __name__ == '__main__':
    csv_file = 'emotions.csv'
    eeg_data, labels = read_eeg_data(csv_file)
    statistics = compute_statistics(eeg_data)

    print('Statistical characteristics of the EEG signals:')
    print(statistics)

import pandas as pd
from sklearn.utils import shuffle

# List of file names and their corresponding labels
files = [
    ('GAMEEMO/S01G1AllChannels.csv', 'G1'),
    ('GAMEEMO/S01G2AllChannels.csv', 'G2'),
    ('GAMEEMO/S01G3AllChannels.csv', 'G3'),
    ('GAMEEMO/S01G4AllChannels.csv', 'G4')
]

data_frames = []

# Iterate over the files and their labels
for file, label in files:
    data = pd.read_csv(file, header=0)

    if 'Unnamed: 14' in data.columns:
        data = data.drop(columns=['Unnamed: 14'])

    # Add a new column with the label
    data['label'] = label

    # Append the data to the data_frames list
    data_frames.append(data)

combined_data = pd.concat(data_frames)

combined_data = shuffle(combined_data)

combined_data.to_csv('gameemo_preprocessed.csv', index=False)

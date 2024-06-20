import os
import numpy as np
import pandas as pd
import torch

def load_and_preprocess_data(train_path, valid_path, remove_values, base, valid_prob=0.2, test_prob=0.1):
    """
    Loads and preprocesses data from CSV files and then splits it into training, validation, and test sets.

    Args:
        train_path (str): Path to the training CSV file.
        valid_path (str): Path to the validation CSV file.
        remove_values (list): List of values to remove from the dataset.
        base (str): The base path of the data for batch filtering.
        valid_prob (float): Probability for validation set. Default is 0.2.
        test_prob (float): Probability for test set. Default is 0.1.

    Returns:
        train (pd.DataFrame): Training set.
        valid (pd.DataFrame): Validation set.
        test (pd.DataFrame): Test set.
    """
    # Load training data
    df_train = pd.read_csv(train_path)
    df_train["split"] = "train"

    # Load validation data
    df_valid = pd.read_csv(valid_path)
    df_valid["split"] = "valid"

    # Concatenate dataframes
    data = pd.concat([df_train, df_valid], axis=0, ignore_index=True)

    # Print the shape of the concatenated dataframe
    print(f"Combined dataframe shape: {data.shape}")

    # Remove rows with unknown speaker age or gender
    original_shape = data.shape[0]
    data = data[data["SpeakerAge"] != "Unknown"]
    data = data[data["SpeakerGender"] != "Unknown"]

    # Remove rows with unwanted speaker dialects
    data = data[~data["SpeakerDialect"].isin(remove_values)]

    # Print information about removed rows
    removed_rows = original_shape - data.shape[0]
    print(f"Removed {removed_rows} rows with unknown age/gender and unwanted dialects.")

    # Filter the DataFrame to select data in the specified drive (batch_1)
    df_with_existing_files = data[data['FileName'].apply(lambda x: os.path.exists(os.path.join(base, x.split('/')[-1])))]

    # Calculate the probabilities for each split
    train_prob = 1 - valid_prob - test_prob

    # Generate random choices for 'split'
    df_with_existing_files['split'] = np.random.choice(['train', 'valid', 'test'],
                                                      size=len(df_with_existing_files),
                                                      p=[train_prob, valid_prob, test_prob])

    data = df_with_existing_files
    # Filter rows where FileName contains 'batch_1'
    data = data[data['FileName'].str.contains('batch_1')]
    data = data.dropna()

    # Separate data into training, validation, and test sets
    train = data[data['split']=='train']
    valid = data[data['split']=='valid']
    test = data[data['split']=='test']

    return train, valid, test, data

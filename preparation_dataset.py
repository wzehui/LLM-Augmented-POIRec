import json
import numpy as np
import pandas as pd
import os.path as osp
from datetime import datetime
from filtering import load_csv
from preprocessing import save_csv
from main.data.session_dataset import SessionDataset, TemporalSplit

def remap_id(df, column, save_path, seed):
    """
    Maps a single column to integer values and saves the mapping as a JSON file.

    Args:
        df (pd.DataFrame): DataFrame containing the column to be mapped.
        column (str): Name of the column to map.
        save_path (str): Path to save the JSON mapping file.
        seed (int): Random seed for repeatability.

    Returns:
        pd.DataFrame: DataFrame with the column mapped to integers.
    """
    # Set random seed for repeatability
    np.random.seed(seed)

    # Create unique mapping for the column
    unique_values = df[column].unique()
    mapping = {val: i for i, val in enumerate(unique_values, start=1)}

    # Map the column to integers
    df[column] = df[column].map(mapping)

    # Save mapping to JSON file
    mapping_path = osp.join(save_path, f"{column.lower()}_to_int.json")
    with open(mapping_path, "w") as json_file:
        json.dump(mapping, json_file, indent=4)

    print(f"Mapping for column '{column}' saved to {mapping_path}")
    return df

if __name__ == "__main__":
    seed = 2025
    WORKING_DIR = './yelp'
    load_path = osp.join(WORKING_DIR, 'csv')
    save_path = osp.join(WORKING_DIR, 'dataset')
    checkin_df = load_csv(load_path, 'checkin.csv')

    checkin_df = checkin_df.drop(columns=['review_id'])
    column_mapping = {
        'user_id': 'SessionId',
        'business_id': 'ItemId',
        'stars': 'Reward',
        'date': 'Time'
    }
    checkin_df = checkin_df.rename(columns=column_mapping)

    checkin_df = remap_id(checkin_df, 'SessionId', save_path, seed=seed)
    checkin_df = remap_id(checkin_df, 'ItemId', save_path, seed=seed)

    checkin_df['Time'] = pd.to_datetime(checkin_df['Time'], errors='coerce',
                                        format="%Y-%m-%d %H:%M:%S")
    checkin_df['SessionId'] = checkin_df['SessionId'].astype(int)
    checkin_df['ItemId'] = checkin_df['ItemId'].astype(str)
    checkin_df['Reward'] = checkin_df['Reward'].astype(float)

    save_csv(checkin_df, save_path, 'checkin_filtered_m.csv')

    DATASET_CONFIG = {
        "filepath_or_bytes": osp.join(save_path, 'checkin_filtered_m.csv'),
        # "filepath_or_bytes": osp.join(save_path, 'checkin_filtered_m_split.csv'),
        "sample_size": None,
        "sample_random_state": None,
        "n_withheld": 1,
        "evolving": False,
    }
    SPLIT_CONFIG = {
        # "test_frac": 0.2,
        "num_folds": 3,
        # "test_cutoff": datetime(2022, 1, 1, 0, 0),
        "test_cutoff": datetime(2019, 12, 25, 0, 0),
        "filter_non_trained_test_items": True,
    }

    dataset = SessionDataset(**DATASET_CONFIG)
    dataset.load_and_split(TemporalSplit(**SPLIT_CONFIG))
    dataset.to_pickle(osp.join(save_path, 'dataset'))
    # dataset.to_pickle(osp.join(save_path, 'dataset_W5S3'))

    dataset.train_data.to_csv('./yelp/dataset/dataset_training.csv',
                              index=False)

    DATASET_CONFIG = {
        "filepath_or_bytes": osp.join(save_path, 'dataset_training.csv'),
        # "filepath_or_bytes": csv_buffer.getvalue(),
        "sample_size": None,
        "sample_random_state": None,
        "n_withheld": 1,
        "evolving": False,
    }

    SPLIT_CONFIG = {
        "num_folds": 3,
        "test_cutoff": datetime(2019, 12, 15, 0, 0),
        "filter_non_trained_test_items": True,
    }
    dataset_train = SessionDataset(**DATASET_CONFIG)
    dataset_train.load_and_split(TemporalSplit(**SPLIT_CONFIG))
    dataset_train.to_pickle(osp.join(save_path, 'dataset_training'))
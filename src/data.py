import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import joblib
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from src.config import CFG

# Set environment variable
os.environ["KERAS_BACKEND"] = "jax"  # you can also use tensorflow or torch

# Define a function to process a single spectrogram_id
def process_spec(spec_id, split="train"):
    BASE_PATH = "data"
    SPEC_DIR = "/tmp/dataset/hms-hbac"
    spec_path = f"{BASE_PATH}/{split}_spectrograms/{spec_id}.parquet"
    spec = pd.read_parquet(spec_path)
    spec = spec.fillna(0).values[:, 1:].T  # fill NaN values with 0, transpose for (Time, Freq) -> (Freq, Time)
    spec = spec.astype("float32")
    np.save(f"{SPEC_DIR}/{split}_spectrograms/{spec_id}.npy", spec)

def load_data():
    BASE_PATH = "data"
    SPEC_DIR = "/tmp/dataset/hms-hbac"
    os.makedirs(SPEC_DIR + '/train_spectrograms', exist_ok=True)
    os.makedirs(SPEC_DIR + '/test_spectrograms', exist_ok=True)

    # Train + Valid
    df = pd.read_csv(f'{BASE_PATH}/train.csv')
    df['eeg_path'] = f'{BASE_PATH}/train_eegs/' + df['eeg_id'].astype(str) + '.parquet'
    df['spec_path'] = f'{BASE_PATH}/train_spectrograms/' + df['spectrogram_id'].astype(str) + '.parquet'
    df['spec2_path'] = f'{SPEC_DIR}/train_spectrograms/' + df['spectrogram_id'].astype(str) + '.npy'
    df['class_name'] = df.expert_consensus.copy()
    df['class_label'] = df.expert_consensus.map(CFG.name2label)
    display(df.head(2))

    # Test
    test_df = pd.read_csv(f'{BASE_PATH}/test.csv')
    test_df['eeg_path'] = f'{BASE_PATH}/test_eegs/' + test_df['eeg_id'].astype(str) + '.parquet'
    test_df['spec_path'] = f'{BASE_PATH}/test_spectrograms/' + test_df['spectrogram_id'].astype(str) + '.parquet'
    test_df['spec2_path'] = f'{SPEC_DIR}/test_spectrograms/' + test_df['spectrogram_id'].astype(str) + '.npy'
    display(test_df.head(2))

    return df, test_df

def preprocess_data(df, test_df):
    SPEC_DIR = "/tmp/dataset/hms-hbac"

    # Process training spectrograms
    spec_ids = df["spectrogram_id"].unique()
    _ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_spec)(spec_id, "train")
        for spec_id in tqdm(spec_ids, total=len(spec_ids))
    )

    # Process testing spectrograms
    test_spec_ids = test_df["spectrogram_id"].unique()
    _ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_spec)(spec_id, "test")
        for spec_id in tqdm(test_spec_ids, total=len(test_spec_ids))
    ) 
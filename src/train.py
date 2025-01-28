import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
from src.config import CFG
from src.data import load_data, preprocess_data
from src.utils import build_dataset, plot_dataset_samples
from src.model import build_model
from src.augmentations import build_augmenter
import keras

def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 5e-5, 6e-6 * batch_size, 1e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        elif mode == 'exp':
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step':
            lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs = epochs - lr_ramp_ep - lr_sus_ep + 3
            decay_epoch_index = epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('lr')
        plt.title('LR Scheduler')
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback

def main():
    # Load and preprocess data
    df, test_df = load_data()
    preprocess_data(df, test_df)
    
    # Stratified Group K-Fold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
    df["fold"] = -1
    df.reset_index(drop=True, inplace=True)
    for fold, (train_idx, valid_idx) in enumerate(
        sgkf.split(df, y=df["class_label"], groups=df["patient_id"])
    ):
        df.loc[valid_idx, "fold"] = fold
    df.groupby(["fold", "class_name"])[["eeg_id"]].count().T

    # Sample from full data
    sample_df = df.groupby("spectrogram_id").head(1).reset_index(drop=True)
    train_df = sample_df[sample_df.fold != CFG.fold]
    valid_df = sample_df[sample_df.fold == CFG.fold]
    print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")

    # Build Datasets
    train_paths = train_df.spec2_path.values
    train_offsets = train_df.spectrogram_label_offset_seconds.values.astype(int)
    train_labels = train_df.class_label.values
    train_ds = build_dataset(train_paths, train_offsets, train_labels, batch_size=CFG.batch_size,
                             repeat=True, shuffle=True, augment=True, cache=True)

    valid_paths = valid_df.spec2_path.values
    valid_offsets = valid_df.spectrogram_label_offset_seconds.values.astype(int)
    valid_labels = valid_df.class_label.values
    valid_ds = build_dataset(valid_paths, valid_offsets, valid_labels, batch_size=CFG.batch_size,
                             repeat=False, shuffle=False, augment=False, cache=True)

    # Dataset Check
    plot_dataset_samples(train_ds)
    
    # Build Model
    model = build_model()
    model.summary()

    # LR Schedule
    lr_cb = get_lr_callback(batch_size=CFG.batch_size, mode=CFG.lr_mode, plot=True)

    # Model Checkpoint
    ckpt_cb = keras.callbacks.ModelCheckpoint("best_model.keras",
                                             monitor='val_loss',
                                             save_best_only=True,
                                             save_weights_only=False,
                                             mode='min')

    # Train Model
    history = model.fit(
        train_ds, 
        epochs=CFG.epochs,
        callbacks=[lr_cb, ckpt_cb], 
        steps_per_epoch=len(train_df)//CFG.batch_size,
        validation_data=valid_ds, 
        verbose=CFG.verbose
    )

    # Save Training History if needed
    # joblib.dump(history.history, 'training_history.pkl')

if __name__ == "__main__":
    main() 
class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    preset = "mobilenet_v3_large_imagenet"  # Name of pretrained classifier
    image_size = [400, 300]  # Input image size
    epochs = 13  # Training epochs
    batch_size = 64  # Batch size
    lr_mode = "cos"  # LR scheduler mode from one of "cos", "step", "exp"
    drop_remainder = True  # Drop incomplete batches
    num_classes = 6  # Number of classes in the dataset
    fold = 0  # Which fold to set as validation data
    class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    label2name = dict(enumerate(class_names))
    name2label = {v: k for k, v in label2name.items()}

    # Reproducibility:
    import keras
    keras.utils.set_random_seed(seed) 
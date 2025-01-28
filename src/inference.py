import tensorflow as tf
from src.config import CFG
from src.data import load_data
from src.utils import build_dataset
from src.model import build_model
import matplotlib.pyplot as plt

def main():
    # Load best model
    model = build_model()
    model.load_weights("best_model.keras")

    # Load test data
    _, test_df = load_data()
    test_paths = test_df.spec2_path.values
    test_ds = build_dataset(test_paths, batch_size=min(CFG.batch_size, len(test_df)),
                             repeat=False, shuffle=False, cache=False, augment=False)

    # Inference
    preds = model.predict(test_ds)

    # Further processing of preds as needed
    # ...

if __name__ == "__main__":
    main() 
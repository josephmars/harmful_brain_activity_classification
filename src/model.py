import keras
import keras_cv
from src.config import CFG

def build_model():
    # Build Classifier
    model = keras_cv.models.ImageClassifier.from_preset(
        CFG.preset, num_classes=CFG.num_classes
    )

    # Compile the model  
    LOSS = keras.losses.KLDivergence()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss=LOSS)
    
    return model 
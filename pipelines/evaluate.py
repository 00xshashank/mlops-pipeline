from typing import Tuple, Dict, List
from pathlib import Path

import numpy as np
import joblib
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize

from zenml import step, pipeline

@step
def load_cnn_feature_extractor(model_path: str) -> tf.keras.Model:
    model = tf.keras.models.load_model(model_path)
    return model

@step
def load_ensemble(model_path: str):
    return joblib.load(model_path)

@step
def load_test_data(
    test_dir: str,
    image_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:

    test_dir = Path(test_dir)
    class_names = sorted(
        [p.name for p in test_dir.iterdir() if p.is_dir()]
    )

    images, labels = [], []

    for label_idx, class_name in enumerate(class_names):
        for img_path in (test_dir / class_name).glob("*"):
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=image_size
            )
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img / 255.0

            images.append(img)
            labels.append(label_idx)

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        class_names,
    )

@step
def extract_features(
    cnn: tf.keras.Model,
    images: np.ndarray,
) -> np.ndarray:
    features = cnn.predict(images, verbose=0)
    return features

@step
def evaluate_model(
    ensemble,
    features: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:

    y_pred = ensemble.predict(features)
    y_proba = ensemble.predict_proba(features)

    accuracy = accuracy_score(y_true, y_pred)

    y_true_bin = label_binarize(
        y_true, classes=list(range(len(class_names)))
    )

    roc_auc = roc_auc_score(
        y_true_bin,
        y_proba,
        multi_class="ovr",
        average="macro",
    )

    print("\nClassification Report\n")
    print(classification_report(
        y_true, y_pred, target_names=class_names
    ))

    return {
        "accuracy": accuracy,
        "roc_auc_macro": roc_auc,
    }

@pipeline
def evaluate_skin_disease_pipeline(
    cnn_model_path: str,
    ensemble_model_path: str,
    test_data_dir: str,
):
    cnn = load_cnn_feature_extractor(cnn_model_path)
    ensemble = load_ensemble(ensemble_model_path)

    images, labels, class_names = load_test_data(test_data_dir)

    features = extract_features(cnn, images)

    evaluate_model(
        ensemble=ensemble,
        features=features,
        y_true=labels,
        class_names=class_names,
    )

if __name__ == "__main__":
    evaluate_skin_disease_pipeline(
        cnn_model_path="cnn_feature_extractor_model2.h5",
        ensemble_model_path="ensemble_model.pkl",
        test_data_dir="data/test",
    )
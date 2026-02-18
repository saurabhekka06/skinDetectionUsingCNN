import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
MODEL_PATH = "skin_model_pro_max_v2.keras"
VAL_DIR = "dataset/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
OUTPUT_DIR = "model_reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

y_true = val_generator.classes
predictions = model.predict(val_generator).ravel()
fpr, tpr, thresholds = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")

roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
plt.savefig(roc_path)
plt.close()

print("ROC Curve saved:", roc_path)
print("AUC Score:", roc_auc)
prob_true, prob_pred = calibration_curve(
    y_true,
    predictions,
    n_bins=10
)

brier = brier_score_loss(y_true, predictions)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("True Probability")
plt.title(f"Calibration Curve (Brier Score = {brier:.4f})")

calib_path = os.path.join(OUTPUT_DIR, "calibration_curve.png")
plt.savefig(calib_path)
plt.close()

print("Calibration Curve saved:", calib_path)
print("Brier Score:", brier)

print("\nModel Visualization Completed.")

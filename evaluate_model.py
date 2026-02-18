import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
MODEL_PATH = "skin_model_pro_max_v2.keras"
VAL_DIR = "dataset/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
model = load_model(MODEL_PATH)
print("Model loaded successfully")
val_datagen = ImageDataGenerator(rescale=1.0/255)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("Class indices:", val_generator.class_indices)
predictions = model.predict(val_generator)
true_classes = val_generator.classes
fpr, tpr, thresholds = roc_curve(true_classes, predictions)
roc_auc = auc(fpr, tpr)
J = tpr - fpr
best_index = np.argmax(J)
best_threshold = thresholds[best_index]

print(f"\nBest Threshold Found: {best_threshold:.4f}")
y_pred = (predictions >= best_threshold).astype(int)
cm = confusion_matrix(true_classes, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\n Classification Report:")
print(classification_report(
    true_classes,
    y_pred,
    target_names=["Melanoma", "Normal"],
    zero_division=0
))

print("\n AUC Score:", roc_auc)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

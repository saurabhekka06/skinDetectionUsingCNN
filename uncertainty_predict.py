import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
model = tf.keras.models.load_model("skin_model.keras", compile=False)
print("Model loaded")

IMG_SIZE = 224

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def mc_dropout_prediction(img_array, n_samples=30):
    predictions = []

    for _ in range(n_samples):
        preds = model(img_array, training=True).numpy()
        predictions.append(preds[0][0])

    predictions = np.array(predictions)

    mean_prediction = predictions.mean()
    std_prediction = predictions.std()

    return mean_prediction, std_prediction

def interpret(mean, std, threshold=0.27):

    if std > 0.15:
        confidence = "LOW (Send to doctor)"
    elif std > 0.08:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH "

    if mean < threshold:
        diagnosis = "Melanoma"
    else:
        diagnosis = "Normal"

    return diagnosis, confidence
img_path = r"D:\skinDetectionUsingCNN\dataset\val\melanoma\ISIC_0034329.jpg"
img_array = preprocess(img_path)

mean, std = mc_dropout_prediction(img_array)

diagnosis, confidence = interpret(mean, std)

print("\n Monte Carlo Dropout Result")
print(f"Mean Prediction: {mean:.4f}")
print(f"Uncertainty (Std Dev): {std:.4f}")
print(f"Diagnosis: {diagnosis}")
print(f"Confidence Level: {confidence}")

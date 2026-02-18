import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
MODEL_PATH = "skin_model_pro_max_v2.keras"
VAL_DIR = "dataset/val"
OUTPUT_DIR = "gradcam_batch_outputs"
IMG_SIZE = 224
MC_SAMPLES = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")
last_conv_layer = None

for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

if last_conv_layer is None:
    raise ValueError("No Conv2D layer found in model.")

print("Using Conv Layer:", last_conv_layer.name)
grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[last_conv_layer.output, model.output]
)
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)
def mc_dropout_prediction(img_tensor, n_samples=MC_SAMPLES):

    preds = []

    for _ in range(n_samples):
        prediction = model(img_tensor, training=True)
        preds.append(prediction.numpy()[0][0])

    return np.mean(preds), np.std(preds)
def make_gradcam_heatmap(img_tensor):

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_tensor, training=False)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()
def get_risk_category(prob):
    if prob >= 0.6:
        return "HIGH RISK"
    elif prob >= 0.3:
        return "MODERATE RISK"
    else:
        return "LOW RISK"

def get_confidence(std):
    if std < 0.05:
        return "VERY HIGH"
    elif std < 0.10:
        return "HIGH"
    else:
        return "LOW"
def process_image(img_path):

    img_tensor = preprocess(img_path)

    mean_pred, uncertainty = mc_dropout_prediction(img_tensor)
    heatmap = make_gradcam_heatmap(img_tensor)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    risk_score = mean_pred * 100
    risk_category = get_risk_category(mean_pred)
    confidence = get_confidence(uncertainty)

    cv2.putText(superimposed_img,
                f"Risk: {risk_score:.1f}%",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.putText(superimposed_img,
                risk_category,
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2)

    cv2.putText(superimposed_img,
                f"Uncertainty: {uncertainty:.3f}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    cv2.putText(superimposed_img,
                f"Confidence: {confidence}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    output_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(output_path, superimposed_img)
print("\nStarting Batch Grad-CAM Processing...\n")

count = 0

for root, dirs, files in os.walk(VAL_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):

            img_path = os.path.join(root, file)
            process_image(img_path)

            count += 1
            if count % 10 == 0:
                print(f"Processed {count} images...")

print("\nBatch Grad-CAM Completed.")
print("Results saved in:", OUTPUT_DIR)

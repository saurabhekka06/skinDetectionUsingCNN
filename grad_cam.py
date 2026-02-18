import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
MODEL_PATH = "skin_model_industry.keras"
IMG_SIZE = 224
TEST_IMAGE = "dataset/val/melanoma/ISIC_0034329.jpg"
OUTPUT_DIR = "gradcam_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")
base_model = model.get_layer("efficientnetb0")
last_conv_layer = base_model.get_layer("top_conv")
print("Using Conv Layer:", last_conv_layer.name)
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array)
def generate_gradcam(img_tensor):

    with tf.GradientTape() as tape:
        conv_outputs = base_model(img_tensor, training=False)
        x = model.layers[2](conv_outputs)   # GlobalAveragePooling
        x = model.layers[3](x, training=False)  # Dropout
        predictions = model.layers[4](x)    # Dense

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()
def run_gradcam(img_path):

    img_tensor = preprocess(img_path)
    heatmap = generate_gradcam(img_tensor)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    output_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(output_path, superimposed_img)

    print("Grad-CAM saved at:", output_path)
run_gradcam(TEST_IMAGE)

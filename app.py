import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
MODEL_PATH = "skin_model_pro_max_v2.keras"
IMG_SIZE = 224
MC_SAMPLES = 20
UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded!")
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)
def mc_dropout_prediction(img_tensor):
    preds = []
    for _ in range(MC_SAMPLES):
        prediction = model(img_tensor, training=True)
        preds.append(prediction.numpy()[0][0])
    return np.mean(preds), np.std(preds)
def generate_gradcam(img_tensor, img_path):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    gradcam_path = os.path.join(GRADCAM_FOLDER, os.path.basename(img_path))
    cv2.imwrite(gradcam_path, superimposed_img)
    return gradcam_path
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
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            img_tensor = preprocess(filepath)
            mean_pred, uncertainty = mc_dropout_prediction(img_tensor)
            risk_score = mean_pred * 100
            risk_category = get_risk_category(mean_pred)
            confidence = get_confidence(uncertainty)
            gradcam_path = generate_gradcam(img_tensor, filepath)
            return render_template(
                "result.html",
                image_path=filepath,
                gradcam_path=gradcam_path,
                risk_score=f"{risk_score:.2f}",
                risk_category=risk_category,
                uncertainty=f"{uncertainty:.4f}",
                confidence=confidence
            )

    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)

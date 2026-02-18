import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
MODEL_PATH = "skin_model_pro_max_v2.keras"

IMG_SIZE = 224
TEST_IMAGE = "dataset/val/melanoma/ISIC_0034329.jpg"  # <-- change if needed
MC_SAMPLES = 30  # Monte Carlo forward passes
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
def mc_dropout_prediction(img_array, n_samples=30):
    predictions = []

    for _ in range(n_samples):
        prediction = model(img_array, training=True)
        predictions.append(prediction.numpy()[0][0])

    mean_prediction = np.mean(predictions)
    uncertainty = np.std(predictions)

    return mean_prediction, uncertainty
img_array = preprocess(TEST_IMAGE)

mean_prediction, uncertainty = mc_dropout_prediction(
    img_array, MC_SAMPLES
)

probability = mean_prediction
risk_score = probability * 100
print("\nAI Assessment")
print(f"Melanoma Probability: {probability:.4f}")
print(f"Risk Score: {risk_score:.2f}%")
print(f"Uncertainty (Std Dev): {uncertainty:.4f}")
if uncertainty < 0.05:
    confidence = "VERY HIGH"
elif uncertainty < 0.10:
    confidence = "HIGH"
elif uncertainty < 0.20:
    confidence = "MODERATE"
else:
    confidence = "LOW"

print(f"Confidence Level: {confidence}")
print("\nFinal Recommendation:")

if risk_score >= 60:
    if uncertainty < 0.10:
        print("Immediate dermatologist referral recommended.")
    else:
        print(" High risk but model uncertainty elevated.")
        print(" Urgent specialist review required.")

elif 30 <= risk_score < 60:
    if uncertainty < 0.10:
        print("Clinical review advised.")
    else:
        print("Borderline & uncertain. Retake image or consult specialist.")

else:
    if uncertainty < 0.10:
        print("Low risk. Routine monitoring suggested.")
    else:
        print("Low risk but uncertain prediction. Retake image.")



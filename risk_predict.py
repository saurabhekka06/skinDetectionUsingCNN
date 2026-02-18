import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
MODEL_PATH = "skin_model_industry.keras"
IMG_SIZE = 224
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded for Risk Scoring")
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
def interpret_risk(prob):

    if prob < 0.20:
        return " LOW RISK", "Routine monitoring recommended."

    elif prob < 0.45:
        return " MODERATE RISK", "Clinical follow-up suggested."

    elif prob < 0.80:
        return " HIGH RISK", "Dermatologist evaluation required."

    else:
        return "CRITICAL RISK", "Urgent medical attention recommended."
def predict_risk(img_path):

    img_array = preprocess(img_path)

    prediction = model.predict(img_array)[0][0]
    melanoma_prob = 1 - prediction

    risk_score = melanoma_prob * 100

    level, advice = interpret_risk(melanoma_prob)

    print("\n AI Risk Assessment")
    print(f"Melanoma Probability: {melanoma_prob:.4f}")
    print(f"Risk Score: {risk_score:.2f}%")
    print(f"Risk Level: {level}")
    print(f"Recommendation: {advice}")
    
TEST_IMAGE = "dataset/val/melanoma/ISIC_0034329.jpg"

if os.path.exists(TEST_IMAGE):
    predict_risk(TEST_IMAGE)
else:
    print("Test image not found.")

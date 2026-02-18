import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Multiply,
    Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 15

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_PATH = "skin_model_pro_max_v2.keras"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class indices:", train_generator.class_indices)
counts = train_generator.classes
melanoma_count = np.sum(counts == 0)
normal_count = np.sum(counts == 1)

class_weights = {
    0: normal_count / melanoma_count,  # boost melanoma
    1: 1.0
}

print("Class Weights:", class_weights)
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
se = GlobalAveragePooling2D()(x)
se = Reshape((1, 1, 1280))(se)
se = Dense(80, activation="relu")(se)
se = Dense(1280, activation="sigmoid")(se)
x = Multiply()([x, se])

x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()
callbacks = [
    ReduceLROnPlateau(
        monitor="val_auc",
        factor=0.3,
        patience=2,
        verbose=1,
        mode="max"
    ),
    EarlyStopping(
        monitor="val_auc",
        patience=5,
        restore_best_weights=True,
        mode="max"
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=1
    )
]
print("\n PHASE 1: Training classifier head...\n")

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=callbacks
)
print("\nPHASE 2: Fine-tuning backbone...\n")

base_model.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # lower LR for fine-tuning
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=callbacks
)

print("\n PRO MAX V2 Training Completed.")

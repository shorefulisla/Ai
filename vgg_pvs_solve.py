
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.datasets.cifar10 import load_data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# Load Dataset
(x_train, y_train), (x_test, y_test) = load_data()

# Select Airplane, Automobile and Truck
train_idx = np.isin(y_train.flatten(), [0, 1, 9])
test_idx = np.isin(y_test.flatten(), [0, 1, 9])

x_train, y_train = x_train[train_idx], y_train[train_idx]
x_test, y_test = x_test[test_idx], y_test[test_idx]

y_train = np.where(y_train == 9, 2, y_train)
y_test = np.where(y_test == 9, 2, y_test)

# Split Training and Validation Data
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

# Preprocess Images
x_train = preprocess_input(x_train.astype("float32"))
x_test = preprocess_input(x_test.astype("float32"))
x_val = preprocess_input(x_val.astype("float32"))


# Load VGG16
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3)
)

# Entire VGG16 Fine-Tuning
base_model.trainable = True

# Build Model
model = Sequential([
    Input(shape=(32, 32, 3)),
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Reduce Learning Rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.0000001
)

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.000001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    jit_compile=False
)

model.summary()

# Train Model
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Entire VGG16 Test Accuracy:", test_accuracy)


# Plot Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Entire VGG16: Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# Partial VGG16 Fine-Tuning
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3)
)

base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

# Build Model
model = Sequential([
    Input(shape=(32, 32, 3)),
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.000001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    jit_compile=False
)

model.summary()

# Train Model
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Partial VGG16 Test Accuracy:", test_accuracy)


# Plot Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Partial VGG16: Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
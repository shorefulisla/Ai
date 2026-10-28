
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adagrad


# Load Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Select Odd Digits
train_idx = np.isin(y_train, [1, 3, 5, 7, 9])
test_idx = np.isin(y_test, [1, 3, 5, 7, 9])

X_train, y_train = X_train[train_idx], y_train[train_idx]
X_test, y_test = X_test[test_idx], y_test[test_idx]

y_train = (y_train - 1) // 2
y_test = (y_test - 1) // 2

# Split Training and Validation Data
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    stratify=y_train,
    random_state=42
)

# Preprocess Images
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0

# Reshape Images for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)

# Build CNN Model
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

# Model Checkpoint
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Compile Model
model.compile(
    optimizer=Adagrad(learning_rate=0.003),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train First 10 Epochs
history1 = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)

# Freeze First 3 Convolutional Layers
conv_count = 0

for layer in model.layers:
    if isinstance(layer, Conv2D):
        conv_count += 1
        if conv_count <= 3:
            layer.trainable = False

# Compile Model
model.compile(
    optimizer=Adagrad(learning_rate=0.003),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Next 20 Epochs
history2 = model.fit(
    X_train,
    y_train,
    initial_epoch=10,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)

# Evaluate Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)




# ADDITIONAL PART 



# Evaluate Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Combine Training History
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
accuracy = history1.history['accuracy'] + history2.history['accuracy']
val_accuracy = history1.history['val_accuracy'] + history2.history['val_accuracy']

# Plot Loss
plt.figure(figsize=(8, 5))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure(figsize=(8, 5))
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

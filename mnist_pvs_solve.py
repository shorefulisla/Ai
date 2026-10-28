
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# Load Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Split Training and Validation Data
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
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

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Build CNN Model
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(1, 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile Model
model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train Model with Original Dataset
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val)
)

# Evaluate Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Original Model - Test Loss:", test_loss)
print("Original Model - Test Accuracy:", test_accuracy)

# Train Second Model with Augmented Dataset
model_aug = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(1, 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile Model
model_aug.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_aug.summary()

history_aug = model_aug.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=10,
    validation_data=(X_val, y_val)
)

# Augment Test Dataset
test_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

X_test_aug = next(
    test_datagen.flow(X_test, y_test, batch_size=len(X_test), shuffle=False)
)[0]

# Evaluate Original Model on Augmented Test Data
loss1, acc1 = model.evaluate(X_test_aug, y_test)

print("Original Model - Augmented Test Accuracy:", acc1)

# Evaluate Augmented Model
loss2, acc2 = model_aug.evaluate(X_test, y_test)

print("Augmented Model - Original Test Accuracy:", acc2)

loss3, acc3 = model_aug.evaluate(X_test_aug, y_test)

print("Augmented Model - Augmented Test Accuracy:", acc3)

# Plot Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Original Training Loss')
plt.plot(history.history['val_loss'], label='Original Validation Loss')
plt.plot(history_aug.history['loss'], label='Augmented Training Loss')
plt.plot(history_aug.history['val_loss'], label='Augmented Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Original Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Original Validation Accuracy')
plt.plot(history_aug.history['accuracy'], label='Augmented Training Accuracy')
plt.plot(history_aug.history['val_accuracy'], label='Augmented Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

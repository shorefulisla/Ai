import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Generate Data
np.random.seed(42)
tf.random.set_seed(42)

def equation(x):
    return 5 * x**4 - 10 * x**3 - x + 5

x = np.linspace(-20, 20, 1000).reshape(-1, 1)
y = equation(x)

# 2. Split Data: 80% Train, 10% Validation, 10% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.20, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# 3. Normalize Data to [-1, +1]
x_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train)
y_val = y_scaler.transform(y_val)
y_test = y_scaler.transform(y_test)

# 4. Build DNN Model
model = Sequential([
    Input(shape=(1,)),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='linear')
])

# 5. Display Model and Parameters
model.summary()

# 6. Compile Model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 7. Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# 8. Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# 9. Evaluate Model on Test Data
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)            

print("Test MSE (normalized):", test_loss)
print("Test MAE (normalized):", test_mae)

# 10. Predict Test Data
y_pred_scaled = model.predict(X_test, verbose=0)

y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test)
X_test_original = x_scaler.inverse_transform(X_test)

mse = np.mean((y_true - y_pred) ** 2)
mae = np.mean(np.abs(y_true - y_pred))

print("Test MSE:", mse)
print("Test MAE:", mae)

# 11. Display True vs Predicted Values
order = np.argsort(X_test_original[:, 0])

plt.plot(X_test_original[order], y_true[order], label='True Values')
plt.plot(X_test_original[order], y_pred[order], label='Predicted Values')

plt.xlabel('x')
plt.ylabel('y')
plt.title('True vs Predicted Values')
plt.legend()
plt.grid()
plt.show()



plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()
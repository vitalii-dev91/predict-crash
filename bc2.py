import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

# ------------------------
# Parameters
# ------------------------
SEQ_LEN = 6  # number of past values used to predict next
NUM_CLASSES = 3  # R, B, M

sequences = [
    [2.37, 1.47, 2.88, 1.33, 6.66, 3.25, 1.24, 8.27, 4.46, 7.93, 1.65, 2.09, 2.00, 9.24, 33.53, 10.88, 1.33, 2.10, 1.14, 1.20, 3.8, 3.14, 1.44, 5.28, 1.85, 1.10, 20.26, 1.31, 1.21, 3.33, 1.71, 1.62, 1.10, 1.69, 2.31, 1.02, 1.58, 10.2, 1.00, 1.12, 15.52, 2.05, 6.62, 5.61, 3.18, 1.61, 7.89, 53.14, 1.7, 1.64, 1.28, 5.46, 2.78, 1.87, 1.37, 2.64, 1.57, 1.57, 1.65, 4.19, 1.9, 1.05, 3.04, 1.0, 1.0, 2.16, 4.04, 2.24, 11.26, 1.12, 3.69, 3.55, 1.79, 4.13, 17.99, 5.54, 1.46, 2.7, 14.57, 50.21],
    [1.31, 19.81, 8.42, 8.89, 3.16, 1.55, 2.27, 2.5, 1.06, 23.69, 1.1, 1.08, 1.15, 3.03, 1.43, 2.66, 2.43, 1.22, 6.93, 1.7, 2.3, 2.39, 2.02, 1.78, 1.28, 3.15, 1.1, 8.09, 4.86, 1.78, 2.66, 2.62, 1.2, 12.86, 2.45, 1.05, 1.47, 1, 1.03, 6.84, 2.28, 1.84, 1.56, 1, 2.18, 5.36, 49.98, 2.63, 3.18, 24.18, 3.05, 18.03, 1.52, 1.77, 8.27, 1.43, 1.19, 5.88, 3.74, 1.74],
    [4.89, 1.79, 1.02, 39.88, 5.64, 1.46, 3.98, 4.67, 3.87, 1.84, 4.78, 1.41, 3.08, 9.98, 1.63, 1.52, 5.3, 1.91, 1.04, 1.72, 1.89, 1.85, 1.63, 1.11, 1.67, 3.22, 3.49, 2.1, 3.11, 3.44, 26.16, 4.94, 1.2, 1.73, 1.98, 1, 1.77, 1.11, 1.9, 2.32, 1.33, 38.25, 1.24, 2.54, 14.63, 1.02, 4.16, 1.82, 1.09, 1.04, 1.82, 6.62, 1.36, 2.51, 4.34, 1.54, 2.57, 1.89, 7.01, 2.12, 148.78, 1.64, 1.1, 1.3, 1.04, 3.31, 25.87, 1.23, 124.16, 1.33, 1.26, 1.64, 1.42, 1.31, 2.77, 23.67, 3.14, 4.9, 3.67, 4.85, 4.13, 8.64, 1.17, 9.52, 6.25, 3.52, 2.94, 2.82, 1.05, 31.31, 1.28, 2.66, 2.44, 1.4, 1.02, 4.28, 2.01, 1.1, 7.99, 1.51],
    [1.42, 1.31, 2.77, 23.67, 3.14, 4.9, 3.67, 4.85, 4.13, 8.64, 1.17, 9.52, 6.25, 3.52, 2.94, 2.82, 1.05, 31.31, 1.28, 2.66, 2.44, 1.4, 1.02, 4.28, 2.01, 1.1, 7.99, 1.51, 7.37, 1.16, 3.57, 1.03, 1.83, 6.63, 1.89, 2.89, 1.03, 1.12, 1.14, 1.32, 2.08, 7.23, 3.69, 6, 1.44, 4.26, 1.73, 11.7, 1.05, 1.16, 2.12, 4.11, 2.14, 3.32, 1.35, 1.43, 1.32, 1.75, 5.7, 1.16, 2.81, 2.29, 2.08, 2.03, 1.76, 2.15, 8.15, 1.39],
    [1.25, 1.3, 4.2, 1.18, 1.18, 5.15, 2.54, 1.15, 5.22, 1.38, 1.37, 5.29, 5.19, 89.79, 1.27, 5.13, 1, 2, 2.23, 1.69, 1.36, 1.54, 2.96, 4.46, 1.03, 1.28, 1.19, 2.32, 1.84, 1.26, 1.79, 3.87, 1.78, 2.36, 1.6, 1.51, 1.35, 12.65, 1.28, 1.09, 5.97, 2.82, 3.26, 1.38, 1.72, 1.21, 2.57, 29.24, 3.48, 7.39, 3.39],
    [5.97, 2.82, 3.26, 1.38, 1.72, 1.21, 2.57, 29.24, 3.48, 7.39, 3.39, 44.54, 1.87, 1.58, 3.25, 3.8, 1.18, 6.74, 2.9, 1.89, 7.75, 2.58, 3.06, 2.21, 6.45, 1.59, 9.67, 16.07, 1.7, 8.52, 1.27, 2.1, 1.29, 1.1, 11.38, 2, 3.92, 2.05, 3.14, 5.88, 3.1, 4.81, 5.12, 25.37, 11.01, 6.07, 1.67, 1.33, 1.2, 1.95],
    [1.38, 1.12, 1.86, 1.56, 1.63, 2.82, 1.61, 1, 1.85, 1.26, 1.45, 1.41, 1.25, 5.07, 1.12, 1.35, 1.44, 1, 11.19, 2.23, 1.02, 2.73, 1.17, 1.81, 10.8, 1.26, 4.09, 3.45, 8.28, 5.15, 1.24, 1.09, 3.35, 66.87, 1.64, 3.58, 1.07, 1.28, 1.02, 1.5, 1.17, 9.36, 1.77, 3.12, 9.96, 1.33, 1.16, 75.83, 1.52, 5.65],
    [12.85, 1.2, 2.37, 1.53, 1.6, 2.62, 1.07, 2.89, 2.23, 1.11, 25.66, 7.58, 1.59, 2.16, 22.19, 1.54, 2.95, 2.19, 9.57, 1.13, 1.75, 70.35, 1.96, 1.12, 1, 1.91, 1.02, 5.18, 4.84, 1.18, 1.2, 1.89, 6.37, 4.94, 1.47, 4.09, 2.66, 1.7, 1.59, 1.07, 262.82, 7.8, 2.43, 9.63, 17.44, 2.16, 5.35, 1.31, 1.06, 2.13, 5.86],
    [1.75, 70.35, 1.96, 1.12, 1, 1.91, 1.02, 5.18, 4.84, 1.18, 1.2, 1.89, 6.37, 4.94, 1.47, 4.09, 2.66, 1.7, 1.59, 1.07, 262.82, 7.8, 2.43, 9.63, 17.44, 2.16, 5.35, 1.31, 1.06, 2.13, 5.86, 1.53, 1.26, 4.26, 28.34, 1.03, 4.68, 2.17, 1.05, 5.89, 1.88, 3.67, 3.53, 1.02, 8.18, 1.2, 3.2, 1.15, 1.43, 1.39],
    [1.15, 1.43, 1.39, 6.27, 1.41, 1.98, 2.55, 1.12, 1.3, 1, 4.18, 1.65, 1.28, 1.11, 1.23, 1, 1.06, 1.44, 1.07, 4.15, 1.04, 1.06, 1.09, 60.25, 1.63, 1.06, 5.89, 2.52, 1.9, 1.38, 3.97, 1.38, 4.71, 4.84, 1.9, 1.54, 2.16, 2.07, 1.98, 1.04, 1.49, 1.43, 1.21, 55.13, 2.89, 5.61, 6.27, 1.32, 1.14, 2.32],
    [1.86, 7.3, 1.9, 2.04, 1.98, 1, 3.42, 4.01, 2.16, 1.42, 1.21, 1.94, 11.57, 37.14, 1.14, 1.59, 1.13, 5, 3.74, 1.12, 1.22, 1, 1.92, 26.66, 50.66, 1.25, 1.85, 8.63, 1.94, 2.24, 46.95, 1.73, 1.21, 1.04, 1.05, 1.82, 4.08, 1.27, 1.78, 1.58, 5.43, 1.05, 116.91, 2.29, 4.2, 1.03, 7.6, 2.04, 2.68, 1.97],
    [1.12, 2.67, 1.52, 1.07, 1.39, 1.39, 44.87, 2.73, 2.26, 1.88, 6.77, 1.59, 3.43, 1.78, 3.96, 1.37, 1.47, 1.25, 4.12, 2.58, 2.68, 2.07, 2.99, 13.93, 1.12, 100.92, 1.21, 3.35, 1.26, 1.77, 1, 1.13, 29.83, 2.21, 1.15, 2.12, 3.77, 1.72, 2.58, 2.7, 1.27, 3.33, 1.15, 1.28, 1.33, 1.31, 10.15, 27.21, 10.06, 8.01, 27.13, 1.34, 1.66, 4.29],
    [1.69, 1.69, 19.26, 4.86, 9.3, 15.12, 46.43, 7.18, 2.72, 1, 1.06, 8.7, 6.89, 1.32, 3.84, 1.37, 2.39, 1.46, 3.36, 4.63, 1.73, 1, 3.73, 6.89, 1.65, 5.56, 3.53, 2.31, 1.33, 1.04, 8.51, 1.83, 1.01, 8.08, 1.13, 3.86, 1.1, 5.59, 11.49, 3.31, 1.25, 1.53, 1.07, 4.61, 8.65, 8.81, 2.02, 4.02, 1.36, 3.86, 4.92, 1.13, 4.6, 1.23, 8.35, 1.74, 19.72],
    [6.39, 4.94, 1.68, 11.85, 7.97, 6.59, 4.8, 1.4, 1.57, 2.24, 3.88, 1.96, 8.62, 4.58, 1.31, 1.54, 2.06, 1.54, 1.64, 2.23, 1.11, 2.68, 1.1, 2.82, 1, 3.34, 1.07, 8.13, 1.78, 3.44, 1.84, 1.02, 1, 2.3, 1.58, 1.04, 1.61, 2.7, 5.52, 1, 1.08, 1.34, 39.32, 5.15, 277.34, 2.34, 1.53, 1.14, 1.41, 1.47, 1.03, 1.51, 1.91, 2.18, 1.37, 12.01, 2.23, 4.02, 4.86, 2.29, 1.05, 1.17, 1.93, 1.19, 2.81, 2.34, 1.37, 2.33, 2.13, 1.68, 1.33, 3.17, 6.51],
    [1.13, 1.74, 271.1, 12.5, 1.97, 1.3, 3.29, 2.68, 1.09, 10.22, 1.44, 2.28, 1.45, 4.26, 5.33, 1.93, 1.36, 1.55, 3.33, 1.27, 2.35, 3.18, 15.02, 5.4, 1.39, 2.15, 2.85, 1.55, 1.88, 6.31, 7, 1.13, 2.06, 4.99, 1, 2.01, 1.15, 2.35, 2.25, 8.59, 1.08, 1.19, 3.12, 1.29, 1.27, 1.52, 3.13, 1.38, 1.5, 2.39, 1.05],
    [5.42, 1.46, 1.64, 36.25, 1.31, 2.01, 3.81, 28.6, 7.73, 26.71, 3.46, 1.7, 3.32, 1.23, 2.77, 1.17, 1.49, 1.13, 1.09, 2.52, 5.43, 1.37, 2.45, 2.53, 1.83, 1.43, 1.2, 1.87, 1.26, 1.6, 2.03, 4.65, 1.6, 1.42, 1.39, 40.71, 1.31, 5.93, 1.95, 2.91, 3.53, 2.18, 6.46, 3.41, 2.11, 1.44, 1.64, 1.14, 1.34, 29.57, 1.59, 1.26, 5.08, 1.35, 2.53, 2.15],
    [1.25, 2.84, 5.61, 1.1, 4.82, 1.58, 4.05, 1.16, 17.16, 1.7, 9.84, 2.36, 1.16, 1.26, 2.11, 2.73, 8.75, 20.06, 2.81, 4.04, 2.29, 1.16, 3.11, 1, 5.94, 208.68, 6.18, 3.24, 1.41, 1.63, 2.96, 4.19, 4.87, 10.68, 1.01, 1.09, 1.43, 2.42, 4.49, 2.42, 1.97, 1.27, 1.63, 1.25, 8.04, 1.83, 6.9, 3.29, 13.76, 1.85, 1.42, 1.27, 1.1, 8.54, 14.72, 17.87, 22.36, 1.68, 1.17, 1.29, 1.43, 2.69],
    [1.27, 1.63, 1.25, 8.04, 1.83, 6.9, 3.29, 13.76, 1.85, 1.42, 1.27, 1.1, 8.54, 14.72, 17.87, 22.36, 1.68, 1.17, 1.29, 1.43, 1.9, 1.19, 2.69, 1.9, 1.29, 1.08, 1.1, 25.38, 1.33, 1.36, 9.49, 1.54, 1.51, 135.58, 1.52, 7.31, 8.88, 5.98, 1.25, 4.51, 3.26, 3.03, 1.66, 1.36, 1.66, 3.72, 2.74, 2.94, 9.73, 15.69, 48.44, 1.32, 1.78, 3.8, 2.6, 3.86, 1.3, 9.99, 1.13, 1.9, 1.14, 1.83, 17.33, 2.24, 31.57, 1.5, 1.18, 1.23, 234.13, 9.85, 1.53, 1.72, 2.5, 2.34, 1.6, 14.76, 1.31, 5.18, 1, 6.79, 1.77, 1.05, 1.17, 1.12, 5.24, 2.8, 1, 5.16, 1.32, 2.88, 3.95],
    [4.55, 5.04, 2.53, 4.55, 2.79, 2.68, 1.72, 1, 1, 1.24, 2.64, 1.3, 40.09, 13.58, 1.09, 1.14, 2.02, 14.83, 3.17, 1.45, 1, 55.87, 1.28, 1.94, 1.12, 1.63, 5.43, 1.11, 3.76, 1.39, 4.59, 1, 1.05, 1.26, 1.92, 8.93, 1.87, 1.19, 1.72, 2.84, 4.9, 2.81, 5.16, 34.36, 1.06, 3.23, 4.02, 1.1, 1, 16.87, 7.27, 1.33, 1.33, 4.63, 2.43, 2.29, 2.56, 1.09, 1.17, 5.82, 3.78, 1.24, 17.1, 1.05, 1.16, 2.22, 2.61, 3.61, 1.18, 1.45, 1.31, 2.55, 1.41, 1.01, 4.66, 2.05, 1.07]
]

# ------------------------
# Step 2: Define transform and classification
# ------------------------
def transform(x):
    return x

def classify(x):
    y = transform(x)
    if y < 2:
        return 0  # R
    elif y < 10:
        return 1  # B
    else:
        return 2  # M

label_map = {0: 'R', 1: 'B', 2: 'M'}

# ------------------------
# Step 3: Create dataset using sliding window
# ------------------------
X = []
y = []

for seq in sequences:
    for i in range(SEQ_LEN, len(seq)):
        window = seq[i-SEQ_LEN:i]
        label = classify(seq[i])
        X.append(window)
        y.append(label)

X = np.array(X)
y = to_categorical(y, num_classes=NUM_CLASSES)

# Normalize X
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, SEQ_LEN)).reshape(-1, SEQ_LEN, 1)

# ------------------------
# Step 4: Split the data
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------------
# Step 5: Build LSTM model
# ------------------------
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(SEQ_LEN, 1)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ------------------------
# Step 6: Train the model
# ------------------------
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ------------------------
# Step 7: Evaluate
# ------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# ------------------------
# Step 8: Predict the next class for a new sequence
# ------------------------

# Example new input sequence (length must match SEQ_LEN)
new_sequence = np.array([])


while True:
    try:
        num = float(input(f"Enter number {new_sequence.size + 1}: "))
        new_sequence = np.append(new_sequence, num)
        print(new_sequence.size)
        if new_sequence.size == SEQ_LEN:

            # Normalize using the same scaler
            new_sequence_scaled = scaler.transform(new_sequence.reshape(1, -1)).reshape(1, SEQ_LEN, 1)

            # Predict class
            prediction = model.predict(new_sequence_scaled)
            predicted_class = np.argmax(prediction)
            predicted_label = label_map[predicted_class]

            print(f"Input sequence: {new_sequence}")
            print(f"Predicted next class: {predicted_label} (probabilities: {prediction[0]})")
            
            new_sequence = new_sequence[1:]
    except ValueError:
        print("Please enter a valid number.")



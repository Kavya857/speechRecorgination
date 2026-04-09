import os
import librosa
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ✅ Dataset path
dataset_path = "./RAVDESS"

# ✅ Emotion mapping
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised"
}

# ✅ Feature extraction
def extract_features(file):
    audio, sr = librosa.load(file, duration=2)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1)
    return mfcc

# ✅ Data containers
X = []
y = []

print("Building dataset...\n")

# ✅ Load dataset
for actor in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor)

    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                file_path = os.path.join(actor_path, file)

                try:
                    # 🎯 Features
                    features = extract_features(file_path)

                    # 🎯 Label
                    parts = file.split("-")
                    emotion_code = parts[2]
                    emotion = emotion_map[emotion_code]

                    X.append(features)
                    y.append(emotion)

                except Exception as e:
                    print(f"{file} → Error")

# ✅ Convert to numpy
X = np.array(X)
y = np.array(y)

print("\nDataset Ready!")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================
# 🔥 MACHINE LEARNING PART
# ============================

# ✅ Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ✅ Build model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(y_encoded)), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Train model
print("\nTraining model...\n")
model.fit(X_train, y_train, epochs=20, batch_size=32)

# ✅ Evaluate model
loss, acc = model.evaluate(X_test, y_test)

print("\nModel Accuracy:", acc)

# ============================
# 🔥 TEST PREDICTION
# ============================

# Predict one sample
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

predicted_label = encoder.inverse_transform([np.argmax(prediction)])

print("\nSample Prediction:", predicted_label[0])
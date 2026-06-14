import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter
import json
import time
import datetime

# -------------------------------
# USER MODE
# -------------------------------
num_users = int(input("Enter number of patients (1 or 2): "))

if num_users == 1:
    user1 = input("Enter patient name: ")
elif num_users == 2:
    user1 = input("Enter Patient 1 name: ")
    user2 = input("Enter Patient 2 name: ")
else:
    num_users = 1
    user1 = "Patient"

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model("EmotionSenseNet_model.h5")

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Healthcare mapping
emotion_map = {
    "Happy": "Stable Positive State",
    "Surprise": "Stable Positive State",
    "Neutral": "Calm Baseline State",
    "Sad": "Emotional Distress",
    "Angry": "Emotional Distress",
    "Fear": "Emotional Distress",
    "Disgust": "Emotional Distress"
}

# Buffers for subtle detection
prob_buffer_user1 = deque(maxlen=10)
prob_buffer_user2 = deque(maxlen=10)

# History buffers
history_user1 = deque(maxlen=50)
history_user2 = deque(maxlen=50)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

# Start time
start_time = time.time()

# -------------------------------
# REAL-TIME LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.2, 5, minSize=(100, 100)
    )

    # Limit faces
    max_faces = 1 if num_users == 1 else 2
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[:max_faces]

    for i, (x, y, w, h) in enumerate(faces):

        face = gray[y:y+h, x:x+w]

        try:
            roi = cv2.resize(face, (96, 96))
        except:
            continue

        roi = roi / 255.0
        roi = np.stack((roi,)*3, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi, verbose=0)[0]

        # 🔥 Subtle enhancement
        prediction = prediction ** 1.2
        prediction[3] *= 1.3  # boost smile

        # 🔥 Temporal averaging
        if i == 0:
            prob_buffer_user1.append(prediction)
            avg_pred = np.mean(prob_buffer_user1, axis=0)
        elif i == 1 and num_users == 2:
            prob_buffer_user2.append(prediction)
            avg_pred = np.mean(prob_buffer_user2, axis=0)
        else:
            avg_pred = prediction

        confidence = np.max(avg_pred)

        if confidence < 0.3:
            emotion = "Neutral"
        else:
            if avg_pred[3] > 0.25:
                emotion = "Happy"
            else:
                emotion = emotion_labels[np.argmax(avg_pred)]

        # Map to healthcare terms
        emotion = emotion_map[emotion]

        # Store history
        if i == 0:
            history_user1.append(emotion)
        elif i == 1 and num_users == 2:
            history_user2.append(emotion)

    cv2.imshow("AI Mental Health Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()

# -------------------------------
# TIME INFO
# -------------------------------
end_time = time.time()
duration = round(end_time - start_time, 2)

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -------------------------------
# ANALYSIS FUNCTION
# -------------------------------
def analyze(history, username):

    if len(history) == 0:
        history.append("Calm Baseline State")

    counts = Counter(history)
    total = len(history)

    dominant = max(counts, key=counts.get)

    distress = counts.get('Emotional Distress', 0)
    score = distress / total

    # Stability calculation
    changes = sum(
        1 for i in range(1, total)
        if history[i] != history[i-1]
    )

    stability = 1 - (changes / total)

    # -------------------------------
    # RISK CLASSIFICATION
    # -------------------------------
    if score < 0.3:
        status = "Stable Condition"
    elif score < 0.6:
        status = "Moderate Emotional Risk"
    else:
        status = "High Emotional Risk"

    # -------------------------------
    # 🔥 CONTEXT-AWARE INTERPRETATION
    # -------------------------------
    if status == "Stable Condition":
        interpretation = "Emotionally Stable"
        observation = "Consistent emotional pattern observed with minimal variation."
        recommendation = "No intervention required. Maintain current routine."

    elif status == "Moderate Emotional Risk":
        interpretation = "Possible Emotional Imbalance"
        observation = "Noticeable emotional fluctuations and mild stress indicators."
        recommendation = "Consider relaxation techniques or short breaks."

    else:
        interpretation = "Elevated Stress Condition"
        observation = "Frequent signs of emotional distress and instability detected."
        recommendation = "Further monitoring recommended. Consider professional support."

    # Additional insight
    if stability > 0.7:
        insight = "Stable emotional pattern observed over time."
    else:
        insight = "Frequent emotional changes detected."

    # -------------------------------
    # SAVE RESULTS
    # -------------------------------
    result = {
        "user": username,
        "time": current_time,
        "duration_seconds": duration,
        "emotion_counts": dict(counts),
        "dominant_emotion": dominant,
        "mental_health_score": round(score, 2),
        "stability_score": round(stability, 2),
        "status": status,
        "interpretation": interpretation,
        "observation": observation,
        "recommendation": recommendation,
        "insight": insight,
        "emotion_sequence": list(history)
    }

    with open(f"{username}_results.json", "w") as f:
        json.dump(result, f)

    print(f"✅ Report saved for {username}")


# -------------------------------
# RUN ANALYSIS
# -------------------------------
analyze(history_user1, user1)

if num_users == 2:
    analyze(history_user2, user2)
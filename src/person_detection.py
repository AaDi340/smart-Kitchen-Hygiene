import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

def main():
    # --- 1. SETUP PATHS AND CONSTANTS ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Path for Keras model is still the same
    model_path = os.path.join(project_root, "models", "keras_model.h5")
    labels_path = os.path.join(project_root, "models", "labels.txt")
    
    # --- THIS IS THE CHANGED LINE ---
    # Path for the Haar Cascade now looks in the SAME folder as the script
    cascade_path = r"D:\Aditya\smart-Kitchen-Hygiene\models\haarcascade_fullbody.xml"


    CONFIDENCE_THRESHOLD = 0.75
    COLOR_POSITIVE = (0, 255, 0)      # Green
    COLOR_NEGATIVE = (0, 0, 255)    # Red
    COLOR_UNCERTAIN = (0, 255, 255) # Yellow

    # --- 2. LOAD BOTH MODELS ---
    print("[INFO] Loading models...")
    try:
        # Load your hygiene classification model
        hygiene_model = load_model(model_path, compile=False)
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        
        # Load the Haar Cascade person detector
        person_cascade = cv2.CascadeClassifier(cascade_path)
        if person_cascade.empty():
            raise IOError("Haar Cascade XML file not found or is corrupted.")

    except Exception as e:
        print(f"[ERROR] Could not load models. Details: {e}")
        return

    print("[INFO] Models loaded successfully.")

    # --- 3. INITIALIZE WEBCAM ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return
    print("[INFO] Starting webcam... Press 'q' to quit.")

    # --- 4. MAIN VIDEO LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- STEP A: DETECT PERSONS USING HAAR CASCADE ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        persons = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

        if len(persons) > 0:
            # --- STEP B: CLASSIFY HYGIENE USING YOUR KERAS MODEL ---
            image_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
            image_normalized = (image_array / 127.5) - 1
            
            prediction = hygiene_model.predict(image_normalized)
            index = np.argmax(prediction)
            class_name = labels[index]
            confidence_score = prediction[0][index]

            display_text = ""
            display_color = (0,0,0)

            if confidence_score < CONFIDENCE_THRESHOLD:
                display_text = f"Unhygienic ({confidence_score*100:.0f}%)"
                display_color = COLOR_UNCERTAIN
            else:
                label = class_name[2:]
                display_text = f"{label}: {confidence_score*100:.0f}%"
                positive_keywords = ["hygienic", "compliant", "clean", "good", "safe", "pass"]
                if any(keyword in label.lower() for keyword in positive_keywords):
                    display_color = COLOR_POSITIVE
                else:
                    display_color = COLOR_NEGATIVE
            
            # --- STEP C: DRAW VISUALS ON THE FRAME ---
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, h), display_color, 15)
            cv2.putText(frame, display_text, (25, 60), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255,255,255), 5, cv2.LINE_AA)
            cv2.putText(frame, display_text, (25, 60), cv2.FONT_HERSHEY_DUPLEX, 1.8, display_color, 2, cv2.LINE_AA)

            for (x, y, w, h) in persons:
                cv2.rectangle(frame, (x, y), (x+w, y+h), display_color, 3)

        cv2.imshow("Smart Kitchen Hygiene Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 5. CLEANUP ---
    print("[INFO] Closing application...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
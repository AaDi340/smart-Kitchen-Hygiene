import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

def main():
    # --- 1. SETUP PATHS ---
    # This makes sure the paths work no matter where you run the script from.
    # It finds the directory the script is in and joins the filenames to it.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "keras_model.h5")
    labels_path = os.path.join(script_dir, "labels.txt")

    # --- 2. LOAD MODEL AND LABELS ---
    print("Loading model...")
    try:
        model = load_model(model_path, compile=False)
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
    except FileNotFoundError as e:
        print(f"Error: Could not find model or labels file.")
        print(f"Please make sure '{os.path.basename(model_path)}' and '{os.path.basename(labels_path)}' are in the same folder as the script.")
        print(f"Details: {e}")
        return # Exit the function if files are not found

    print("Model and labels loaded successfully.")

    # --- 3. START CAMERA ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # --- 4. MAIN VIDEO LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Exiting... Could not read frame from camera.")
            break

        # --- PREPARE IMAGE FOR PREDICTION ---
        # Resize to 224x224
        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        # Reshape and change data type
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        # Normalize the image
        img = (img / 127.5) - 1

        # --- PREDICTION ---
        prediction = model.predict(img)
        index = np.argmax(prediction)
        class_name = labels[index]
        confidence_score = prediction[0][index]

        # --- DISPLAY RESULTS ON SCREEN ---
        # Set text color: Green if "clean", otherwise Red
        color = (0, 255, 0) if "clean" in class_name.lower() else (0, 0, 255)
        # Format the text string
        text = f"{class_name}: {confidence_score*100:.2f}%"
        # Put the text on the frame
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        # Show the final frame in a window
        cv2.imshow("Smart Kitchen Hygiene", frame)

        # --- QUIT CONDITION ---
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 5. CLEANUP ---
    print("Closing application...")
    cap.release()
    cv2.destroyAllWindows()


# This standard line runs the main() function when you execute the script
if __name__ == "__main__":
    main()
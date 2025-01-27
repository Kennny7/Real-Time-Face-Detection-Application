from typing import Tuple, Optional
import cv2
import numpy as np
import dlib
import logging
from tensorflow.keras.models import load_model

# --------------------------- Setup Logging ---------------------------
logging.basicConfig(
    filename="maskify.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------- Utility Functions ---------------------------
def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error("Failed to open the camera.")
        raise RuntimeError("Unable to access the camera.")
    logging.info("Camera successfully initialized.")
    return cap

def process_frame(frame: np.ndarray, detector: dlib.fhog_object_detector, mask_model: Optional[any]) -> Tuple[np.ndarray, int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_count = 0

    for face in faces:
        x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
        face_count += 1

        # Extract the face ROI with boundary checks
        face_roi = frame[max(0, y):min(y1, frame.shape[0]), max(0, x):min(x1, frame.shape[1])]

        # Predict mask status if a mask model is provided
        if mask_model:
            try:
                resized_face = cv2.resize(face_roi, (224, 224))  # Match model input size
                mask_prediction = mask_model.predict(np.expand_dims(resized_face / 255.0, axis=0))
                label = "Mask" if mask_prediction[0][0] > 0.5 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            except Exception as e:
                logging.error(f"Error during mask prediction: {e}")
                label = "Error"
                color = (255, 255, 0)
        else:
            label = "Unknown"
            color = (255, 255, 0)

        # Draw a rectangle and label around the face
        cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        logging.info(f"Detected face {face_count} at coordinates: {x}, {y}, {x1}, {y1} - Label: {label}")

    return frame, face_count

def release_resources(cap: cv2.VideoCapture) -> None:
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Resources released successfully.")

# --------------------------- Main Application ---------------------------
def main() -> None:
    try:
        cap = initialize_camera()
        detector = dlib.get_frontal_face_detector()

        # Load the mask detection model
        #mask_model = load_model('mask_detector.h5')
        mask_model = load_model("mask_detector.keras")


        logging.info("Starting Maskify application.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)
            processed_frame, face_count = process_frame(frame, detector, mask_model)

            # Display the frame
            cv2.putText(
                processed_frame, f"Faces: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            cv2.imshow("Maskify - Mask Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("User exited the application.")
                break

    except Exception as e:
        logging.critical(f"Critical error in Maskify: {e}")
        print(f"Critical Error: {e}")

    finally:
        release_resources(cap)

if __name__ == "__main__":
    main()

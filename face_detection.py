# Import required libraries
import cv2
import numpy as np
import dlib
import logging

# --------------------------- Setup Logging ---------------------------
logging.basicConfig(filename='face_detection.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------- Camera Initialization ---------------------------
try:
    # Connect to your computer's default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logging.error("Failed to open the camera.")
        print("Error: Unable to access the camera.")
        exit(1)

    logging.info("Camera successfully initialized.")

    # Initialize the face detector using dlib's get_frontal_face_detector method
    detector = dlib.get_frontal_face_detector()

    # --------------------------- Main Loop to Capture Frames ---------------------------
    while True:
        try:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                logging.warning("Failed to capture frame.")
                break

            # Flip the frame horizontally for a mirror view
            frame = cv2.flip(frame, 1)

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = detector(gray)

            # Iterator to count detected faces
            i = 0
            for face in faces:
                # Get the coordinates of the detected face
                x, y = face.left(), face.top()
                x1, y1 = face.right(), face.bottom()

                # Draw a rectangle around detected faces
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                # Increment the iterator for each detected face
                i += 1

                # Display the face number on the screen
                cv2.putText(frame, f'Face num {i}', (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                logging.info(f'Detected face {i} at coordinates: {x}, {y}, {x1}, {y1}')

            # Show the resulting frame
            cv2.imshow('Face Detection', frame)

            # Press 'q' to quit the application
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("User pressed 'q' to exit the application.")
                break

        except Exception as e:
            logging.error(f"Error during frame capture and processing: {e}")

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Application closed successfully.")

except Exception as e:
    logging.critical(f"Failed to start the application: {e}")
    print(f"Critical Error: {e}")

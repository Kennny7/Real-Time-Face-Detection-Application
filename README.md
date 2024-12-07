# Real-Time Face Detection Application

## Project Overview

Welcome to the **Real-Time Face Detection Application**! This project leverages OpenCV and dlib to capture video input from your camera, detect faces in real-time, and log detailed information about each detected face. Designed with a focus on performance and reliability, it offers seamless integration with computer cameras, efficient face detection algorithms, and robust logging to help debug any issues. Whether you're exploring face detection or just want to experiment with OpenCV and dlib, this project serves as a solid foundation for your computer vision endeavors.

### Key Features

- **Real-Time Face Detection:** Detects multiple faces simultaneously with high accuracy.
- **Robust Logging:** Logs detected face coordinates and any errors encountered in `face_detection.log`.
- **Error Handling:** Ensures that any issues during frame capture and processing are logged for troubleshooting.
- **Scalability:** Easily extendable for more advanced features like facial recognition, emotion detection, and tracking.

## Project Structure

```
Real Time Face Detection Application/
├── face_detection.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── face_detection.log
└── .vscode/
└── .ipynb_checkpoints/
```

- `face_detection.py`: The main script for real-time camera input and face detection.
- `requirements.txt`: Contains all the necessary dependencies.
- `.gitignore`: Lists files and directories that should be ignored by Git (e.g., log files, notebooks).
- `face_detection.log`: Log file storing all detection and application status messages.

### How to Run the Project

1. **Clone the Repository**  
   Clone the project repository to your local machine.
   ```bash
   git clone https://github.com/Kennny7/Real-Time-Face-Detection-Application.git
   cd Real-Time-Face-Detection-Application

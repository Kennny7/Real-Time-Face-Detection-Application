# Maskify - Real-Time Mask Detection

Maskify is a real-time face mask detection application using a webcam. The application identifies faces in video frames and determines whether the detected faces are wearing masks, providing a visual overlay to indicate the mask status. This tool can be particularly useful in public safety and health scenarios.

## Features
- Detects faces in real-time using a webcam.
- Supports integration with a pre-trained mask detection model.
- Annotates frames with the mask status of detected faces (e.g., "Mask" or "No Mask").
- Real-time face count display.
- Easy-to-understand logs for debugging and analysis.

## Technologies Used
- **Python**: Programming language for building the application.
- **TensorFlow**: For building and deploying machine learning models, including deep learning algorithms.
- **OpenCV**: Library for real-time computer vision tasks.
- **dlib**: Library for facial detection.
- **NumPy**: For efficient array manipulations.
- **Logging**: For application event tracking.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/maskify.git
   ```
2. Navigate to the project directory:
   ```bash
   cd maskify
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The requirements.txt file should include OpenCV, dlib, and NumPy.)*

4. Run the application:
   ```bash
   python maskify_project.py
   ```

5. Press 'q' to quit the application.

## Project Structure
- **maskify_project.py**: Main application script.
- **maskify.log**: Log file generated during application runtime.

## Usage Notes
- Replace the `mask_model` placeholder with your pre-trained mask detection model for full functionality.
- Ensure your webcam is functional and properly configured.

## Contribution
Feel free to fork the repository and make contributions. Create a pull request for any feature improvements or bug fixes.

## License
This project is licensed under the [MIT License](LICENSE).

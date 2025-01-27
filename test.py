import h5py

try:
    with h5py.File('mask_detector.model', 'r') as f:
        print("File is in HDF5 format.")
except:
    print("File is not in HDF5 format.")


import tensorflow as tf
print(tf.__version__)


from tensorflow.keras.models import load_model, save_model

# Load the original model
model = load_model("mask_detector.h5")

# Save it in the new Keras `.keras` format
save_model(model, "mask_detector.keras")

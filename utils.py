import numpy as np
from tensorflow.keras.preprocessing import image

# Example class names - replace with your actual class names
CLASS_NAMES = ['class1', 'class2', 'class3']

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess the image for model prediction.
    Args:
        img_path (str): Path to the image file.
        target_size (tuple): Target size for the image (width, height).
    Returns:
        np.array: Preprocessed image ready for prediction.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0,1]
    return img_array

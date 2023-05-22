import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image as img_prep
import numpy as np

from PIL import Image
from io import BytesIO


def preprocess_image(image):
    # load the image from the InMemoryUploadedFile
    img = Image.open(BytesIO(image.read()))
    img = img.resize((28, 28))  # resize the image to 28x28
    img = img.convert('L')  # convert image to grayscale
    img_tensor = img_prep.img_to_array(img)  # convert image to a tensor
    img_tensor = np.expand_dims(img_tensor, axis=0)  # the model expects a batch of images as input, so we add an extra dimension
    img_tensor /= 255.  # normalize pixel values to [0, 1]
    return img_tensor


def load_model():
    dir_path = os.path.dirname(os.path.realpath(__file__))  # gets the path of the current file
    model_path = os.path.join(dir_path, 'models', 'my_model.h5')  # constructs the path to the model
    return tf.keras.models.load_model(model_path)


def predict(image):
    model = load_model()
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)

    # Створюємо словник міток
    class_labels = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9',
    }

    # Отримуємо індекс найбільшого елемента
    predicted_class = np.argmax(prediction).item()  # <-- використовуємо .item() тут

    # Повертаємо відповідну мітку
    return class_labels[predicted_class]


"""
    # return prediction.tolist()    
    # code:
    return {"prediction": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]} 
    Where the numbers represent the probability of identifying digits from 0 to 9    
"""


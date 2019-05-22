from keras.preprocessing.image import load_img, img_to_array
from constants.constants_local import EMOTIONS_LIST, TRAINED_MODEL_PATH, IMAGE_SIZE
import numpy as np
import pickle


def load_model(model_file):
    with open(model_file, "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model


def predict_emotion(face_images, model_file=TRAINED_MODEL_PATH):
    loaded_model = load_model(model_file)
    predictions = loaded_model.predict(face_images)
    return EMOTIONS_LIST[np.argmax(predictions)]


images = []
image = load_img('data_files/face.jpg', target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode='grayscale')
image = img_to_array(image)
image = np.resize(image, (1, 48, 48, 1))
images.append(image)
result = predict_emotion(face_images=images)
print(result)

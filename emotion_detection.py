import os
import time
import numpy as np
import cv2
from keras.models import load_model


class Emotion(object):
    """
    API for Emotion classification.

    use by:
        1. create Emotion instance and model initialization.
        1. call .load() to load the model.
        3. call .predict() to get the probability values of 8 emotion:
         neutral, happiness, surprise, sadness,anger, disgust, fear, contempt

    # Example:
    #     >>> image = cv2.imread(resource_path("path/to/img/***.img"))
    #     >>> emotion = Emotion()
    #     >>> emotion.load(config)
    #     >>> probability = emotion.predict(image)
    """

    def __init__(self):
        self.model = None
        self.load()

    def load(self):
        t0 = time.time()
        emotion_model_path = "models/0.5.1.h5"
        if os.path.exists(emotion_model_path):
            self.model = load_model(emotion_model_path)
            self.model.predict(np.zeros((1, 64, 64, 1)))
        else:
            raise IOError("{} file does not exist!".format(emotion_model_path))
        print("Finished load emotion models spend time is {}s".format(round((time.time() - t0), 2)))

    @staticmethod
    def preprocess(images, img_height=64, img_width=64):
        """
        :param image:np.array; bgr image (h, w, 3) or gray img (h, w)

        :param img_height: int; target height for resizing
        :param img_width: int; target width for resizing
        :return: np.array; face regions bgr image (img_height, img_width, 3) or gray img (img_height, img_width)
        """
        face = None
        # if len(images.shape) == 3 and images.shape[2] == 3:
        #     images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        #     face = cv2.resize(images, (img_height, img_width))
        #     face = np.reshape(face, (1, img_height, img_width, 1))
        # elif len(images.shape) == 4 and images.shape[3] == 3:
        #     frames = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images])
        #     face = np.array([cv2.resize(frame, (img_height, img_width)) for frame in frames])
        #     face = np.reshape(face, (-1, img_height, img_width, 1))
        if type(images) is list:
            frames = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
            face = np.array([cv2.resize(frame, (img_height, img_width)) for frame in frames])
            face = np.reshape(face, (-1, img_height, img_width, 1))
        return face

    def predict(self, image):
        """
        :param image:np.array; rgb image (h, w, 3) or gray img (h, w)
        :param face_rect: list; location of face regions [left,top,right,bottom]
        :return:np.array;the probability values of 8 emotion
        """
        image = self.preprocess(image)
        probability = self.model.predict(image)
        return probability


if __name__ == '__main__':
    emo = Emotion()

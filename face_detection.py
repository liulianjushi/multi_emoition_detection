import cv2
import os
import time
import numpy as np


class FaceDetection(object):
    """

    """

    def __init__(self):
        self.net = None
        self.load()

    def load(self):
        """
        :param config: dict; load form config file
        :return:
        """
        t0 = time.time()
        prototxt_path = "models/deploy.prototxt.txt"
        caffemodel_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
        if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)  # load face model
        else:
            raise IOError("{} or {} file does not exist!".format(prototxt_path, caffemodel_path))
        print("Finished load face models spend time is {}s".format(round((time.time() - t0), 2)))

    @staticmethod
    def get_face(image, face_rects):
        """
        :param image:np.array; bgr image (h, w, 3) or gray img (h, w)
        :param face_rect: list; location of face regions [left,top,right,bottom]
        :return: np.array; bgr image or gray img
        """
        frame_faces = [image[int(face_rect[1]):int(face_rect[3]), int(face_rect[0]):int(face_rect[2]), :]
                       for _, face_rect in enumerate(face_rects)]
        # faces = []
        # for face_rect in face_rects:
        #     if face_rect is not None:
        #         face = image[face_rect[1]:face_rect[3], face_rect[0]:face_rect[2]]
        #         faces.append(face)
        #     else:
        #         faces.append(None)
        return frame_faces

    def predict(self, image, confidence=0.5, single_rect=True):
        """
        :param image: np.array; face regions
        :param confidence: float; the confidence threshold
        :param single_rect: only detection one region of face, the default is True
        :return:
        """
        face_rect = None
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        if detections.shape[2] != 0:
            faces = detections[0, 0, :, :]
            faces = faces * np.array([1, 1, 1, w, h, w, h])
            faces = faces[np.where(
                (faces[:, 2] > confidence) & (faces[:, 3] > 0) & (faces[:, 4] > 0) & (faces[:, 5] < w) & (
                        faces[:, 6] < h))]
            if len(faces) == 0:
                return face_rect
            face_rect = faces[:, 3:7].astype(int)
            if single_rect:
                face_rect = np.expand_dims(face_rect[0], axis=0)

            return face_rect


if __name__ == '__main__':
    face = FaceDetection()

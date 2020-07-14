import cv2
import numpy as np
import pickle
import os

class FaceDetector:
    def __init__(self, face_detection_path):
        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.sep.join([face_detection_path, "deploy.prototxt"])
        modelPath = os.path.sep.join([face_detection_path,
                "res10_300x300_ssd_iter_140000.caffemodel"])
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

    def detect(self, frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)):
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()
        rects = []
        for i, detection in enumerate(detections):
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # get the width, height
            boxwidth = endX-startX
            boxheight = endY-startY
            # rects.append((startX, startY, endX, endY))
            rects.append((startX, startY, boxwidth, boxheight))
        return rects


class FaceRecognizer:
    def __init__(self, classifier_path, embedding_model_path, le_path):
        self.classifier_path = classifier_path
        # load our serialized face embedding model from disk and set the
        # preferable target to MYRIAD
        print("[INFO] loading face recognizer...")
        self.embedder = cv2.dnn.readNetFromTorch(embedding_model_path)
        self.embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

        # load the actual face recognition model along with the label encoder
        self.recognizer = pickle.loads(open(classifier_path, "rb").read())
        self.le = pickle.loads(open(le_path, "rb").read())

    def predict(self, face):
        (fH, fW) = face.shape[:2]
        # ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
            return 'toosmall', 0.0

        # construct a blob for the face ROI, then pass the blob
        # through our face embedding model to obtain the 128-d
        # quantification of the face
        faceBlob = cv2.dnn.blobFromImage(cv2.resize(face,
                (96, 96)), 1.0 / 255, (96, 96), (0, 0, 0),
                swapRB=True, crop=False)
        self.embedder.setInput(faceBlob)
        vec = self.embedder.forward()

        # perform classification to recognize the face
        preds = self.recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = self.le.classes_[j]
        return name, proba

    def setConfidenceThreshold(self, confidence):
        self.confidence = confidence

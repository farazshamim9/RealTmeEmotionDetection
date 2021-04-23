import os
import cv2
import numpy as np
import requests

#from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

class EmotionAnalyzer(object):
    def __init__(self):
        # load model
        #self.model = model_from_json(open("fer.json", "r").read())
        # load weights
        #self.model.load_weights('fer.h5')
        self.model = load_model('model.h5')

        self.face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        self.cap = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows
    
    def analyseEmotion(self):

        while True:
            ret, test_img = self.cap.read()  # captures frame and returns boolean value and captured image
            if not ret:
              continue
            gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

            faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
               cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
               roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
               roi_gray = cv2.resize(roi_gray, (48, 48))
               img_pixels = image.img_to_array(roi_gray)
               img_pixels = np.expand_dims(img_pixels, axis=0)
               img_pixels /= 255

               predictions = self.model.predict(img_pixels)

              # find max indexed array
               max_index = np.argmax(predictions[0])

               emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
               predicted_emotion = emotions[max_index]

               cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            resized_img = cv2.resize(test_img, (1000, 700))
            cv2.imshow('Facial emotion analysis ', resized_img)

            if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
                    break








        """success, image = self.video.read()
        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()"""

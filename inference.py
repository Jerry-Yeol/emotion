import cv2
import numpy as np
from keras import layers, models, applications, losses, optimizers, callbacks
from keras import backend as K
from network import *
from constant import *

haar_face = cv2.CascadeClassifier(HAAR_FACE_DIR)
emonet = models.load_model(CHECKPOINTS_DIR)

print("Read Cam!")
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_face.detectMultiScale(gray, 1.3, 5)

    # 한 사람 얼굴만 
    for idx, (x,y,w,h) in enumerate(faces):
        if idx > 0: continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
    
    if len(faces) !=0:
        input_tensor = np.reshape(cv2.resize(roi_gray,
         (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE)),
         (1, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 1))
        infer = emonet.predict(input_tensor)
        for index, emotion in enumerate(infer[0]):
            cv2.putText(frame, '%s : %.2f'%(NUM2EMO_DICT[index], emotion*100), (10, index * 20 + 20), \
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            cv2.rectangle(frame, (130, index * 20 + 10), \
                (130 + int(emotion * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
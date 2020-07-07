HAAR_EYE_DIR = './haar/haarcascade_eye.xml'
HAAR_FACE_DIR = './haar/haarcascade_frontalface_default.xml'

DATA_DIR = './data/fer2013/data.npz'

CHECKPOINTS_DIR = './checkpoints/model_2.h5'

TRAIN_IMAGE_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 50

NUM2EMO_DICT = {
    0:"Angry", 1:"Disgust", 
    2:"Fear", 3:"Happy", 
    4:"Sad", 5:"Surprise", 6:"Neutral"}

EMO2NUM_DICT = {
    "Angry":0, "Disgust":1, 
    "Fear":2, "Happy":3, 
    "Sad":4, "Surprise":5, "Neutral":6}
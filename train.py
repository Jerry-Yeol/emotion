import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, losses, optimizers, callbacks
from tensorflow.keras import backend as K
from network import *
from constant import *
print("Package Loaded!")

# For Efficiency
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

tmp = np.load(DATA_DIR)
tmp.allow_pickle = True
tr_img, val_img, te_img = tmp['imgs']
tr_img, val_img, te_img = tr_img/255., val_img/255., te_img/255.

tr_lab, val_lab, te_lab = tmp['labs']
print("Data Loaded!")


checkpoints = callbacks.ModelCheckpoint('./checkpoints/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',save_best_only=True, save_weights_only=True)
earlystop = callbacks.EarlyStopping(patience=10, verbose=1)
lronplateau = callbacks.ReduceLROnPlateau(patienct=5, verbose=1)
csvlogger = callbacks.CSVLogger(f"./checkpoints/result.csv")

emonet = EmoNet()

model_json = emonet.to_json()
with open(f"./checkpoints/model.json", "w") as json_file:
    json_file.write(model_json)

emonet.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])

print("Start Training!")
emonet.fit(
    x=tr_img, y=tr_lab, batch_size=BATCH_SIZE, epochs=EPOCHS,\
    validation_data=(val_img, val_lab), \
    callbacks=[checkpoints, earlystop, lronplateau, csvlogger]
    )
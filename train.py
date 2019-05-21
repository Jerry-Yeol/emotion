import numpy as np
from keras import layers, models, applications, losses, optimizers, callbacks
from keras import backend as K
from network import *
from constant import *
print("Package Loaded!")

tmp = np.load(DATA_DIR)
tr_img, val_img, te_img = tmp['imgs']
tr_img, val_img, te_img = tr_img/255., val_img/255., te_img/255.

tr_lab, val_lab, te_lab = tmp['labs']
print("Data Loaded!")


checkpoints = callbacks.ModelCheckpoint('./checkpoints/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',save_best_only=False)

emonet = EmoNet()

emonet.compile(optimizer=optimizers.adam(), loss='categorical_crossentropy', metrics=['acc'])

print("Start Training!")
emonet.fit(x=tr_img, y=tr_lab, batch_size=BATCH_SIZE,
 validation_data=(val_img, val_lab), callbacks=[checkpoints])
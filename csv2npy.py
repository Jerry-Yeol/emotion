import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_input", help="Path of csv", type=str)
parser.add_argument("--npy_output", help="Path of Output", type=str)
args = parser.parse_args()

import pandas as pd
import numpy as np
from constant import *
from keras.utils import to_categorical, Progbar
print("Package Loaded!")

df = pd.read_csv(args.csv_input)

tr_img = []
tr_lab = []

val_img = []
val_lab = []

te_img = []
te_lab = []
print("Start Making Data!")
progbar = Progbar(len(df))
for i, df_iter in enumerate(df.iterrows()):
    tmp_lab = df_iter[1][0]
    tmp_img = [int(i) for i in df_iter[1][1].split(' ')]
    if df_iter[1][2] == 'Training':
        tr_img.append(tmp_img)
        tr_lab.append(tmp_lab)
        
    elif df_iter[1][2] == 'PublicTest':
        val_img.append(tmp_img)
        val_lab.append(tmp_lab)
    else: 
        te_img.append(tmp_img)
        te_lab.append(tmp_lab)
    progbar.update(i+1)

tr_img = np.reshape(np.array(tr_img), [-1, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 1])
tr_lab = to_categorical(np.array(tr_lab), 7)

val_img = np.reshape(np.array(val_img), [-1, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 1])
val_lab = to_categorical(np.array(val_lab), 7)

te_img = np.reshape(np.array(te_img), [-1, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 1])
te_lab = to_categorical(np.array(te_lab), 7)

np.savez(args.npy_output + 'data', imgs=(tr_img, val_img, te_img), labs=(tr_lab, val_lab, te_lab))

print("Processing Done!")
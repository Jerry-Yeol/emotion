# %%
import os
import shutil
import cv2 as cv
import numpy as np
import pandas as pd

# %%
jaffe_root = './data/jaffe'
img_path = os.path.join(jaffe_root, "images")
csv_path = os.path.join(jaffe_root, "data.csv")

# %%
csv = pd.read_csv(csv_path)
csv.head(5)

# %%
labels = sorted(list(set(csv['facial_expression'].tolist())))
labels
# %%
for label in labels:
    os.makedirs(os.path.join(img_path, label), exist_ok=True)
    label_img_list = csv[csv['facial_expression']==label]
    for i, rows in label_img_list.iterrows():
        src = os.path.join(img_path, rows[0].split("/")[-1])
        dst = os.path.join(img_path, label, rows[0].split("/")[-1])
        shutil.move(src, dst)
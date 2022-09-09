# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:06:30 2021

@author: Sivasangaran V
"""

import io
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow import keras
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description=("Set up classifier model"))
    parser.add_argument("-dataset_path",help="Dataset Directory")
    parser.add_argument("-model_path",help="Model path")
    parser.add_argument("-model_name",help="Model name")
    parser.add_argument("-epochs", help="Epochs")
    parser.add_argument("-patience",help="Patience for callbacks")
    parser.add_argument("-out_file",help="File path in which model summary will be written")
    parser.add_argument("-min_lr",help="Minimum learning rate")
    return parser.parse_args()
alpha = ("a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z")
#alpha = ("r","s","t","u","v","w","x","y","z")
folder_list=alpha
args = parse_args()
ds_asl_dir = args.dataset_path
nepochs = int(args.epochs)
model_path = args.model_path
model_name = args.model_name+'.h5'
asl_ds = tf.keras.preprocessing.image_dataset_from_directory(ds_asl_dir)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "none"
pd.DataFrame(asl_ds.class_names)
for image_batch, labels_batch in asl_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

from PIL import Image
img =  Image.open(ds_asl_dir+"\\a\\hand1_a_bot_seg_1_cropped.jpeg")
width, height = img.size
print(f"Image sample with width={width} and height={height}.")

batch_size = 32
img_height = 64
img_width = 64

#Filtering out corrupted images

num_skipped = 0
for folder_name in folder_list:
    folder_path = os.path.join(ds_asl_dir, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()
        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)
print("Deleted %d images" % num_skipped)

#Augmenting the images

from keras.preprocessing.image import ImageDataGenerator
data_augmentation = ImageDataGenerator(rotation_range=10, rescale=1/255, zoom_range=0.2, horizontal_flip=True,
                                       width_shift_range=0.2, height_shift_range=0.2, validation_split=0.2, brightness_range=[0,2],
                                       channel_shift_range=30)

#Setting train/test split

asl_train_ds = data_augmentation.flow_from_directory(directory=ds_asl_dir, target_size=(img_height, img_width),
                                                     class_mode="categorical", batch_size=batch_size, subset="training")
asl_test_ds = data_augmentation.flow_from_directory(directory=ds_asl_dir, target_size=(img_height, img_width),
                                                    class_mode="categorical", batch_size=batch_size, subset="validation")

from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from keras import backend as K
if K.image_data_format() == "channels_first":
    input_shape = (3, img_height, img_width)
else:
    input_shape = (img_height, img_width, 3)

#Creating a model

model_dl = keras.Sequential()
model_dl.add(Conv2D(32,(3,3),activation="relu",padding="same",input_shape=(input_shape)))
model_dl.add(Conv2D(32,(3,3),activation="relu",padding="same"))
model_dl.add(MaxPool2D(2,2))
model_dl.add(Conv2D(64,(3,3),activation="relu",padding="same"))
model_dl.add(Conv2D(64,(3,3),activation="relu",padding="same"))
model_dl.add(MaxPool2D(2,2))
model_dl.add(Conv2D(128,(3,3),activation="relu",padding="same"))
model_dl.add(Conv2D(128,(3,3),activation="relu",padding="same"))
model_dl.add(MaxPool2D(2,2))
model_dl.add(Flatten())
model_dl.add(Dropout(0.1))
model_dl.add(Dense(512,activation="relu"))
model_dl.add(Dropout(0.2))
#model_dl.add(Dense(512,activation="relu"))
#model_dl.add(Dropout(0.4))
model_dl.add(Dense(len(folder_list),activation="softmax"))
with open(args.out_file,"w") as f:
    model_dl.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\n\n\n\n")
    f.close()
model_dl.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
from keras.callbacks import EarlyStopping,ReduceLROnPlateau #Import callback functions
earlystop=EarlyStopping(patience=int(args.patience)) #Monitor the performance. If it dips, then stop training
learning_rate_reduce=ReduceLROnPlateau(monitor="val_accuracy",min_lr=float(args.min_lr)) #Change learning rate if not performing good enough
callbacks=[earlystop,learning_rate_reduce]

model_hist=model_dl.fit(asl_train_ds, validation_data=asl_test_ds, callbacks=callbacks, epochs=nepochs)
#import matplotlib.pyplot as plt
def plot_accuracy(y):
    if(y == True):
        plt.plot(model_hist.history['accuracy'])
        plt.plot(model_hist.history['val_accuracy'])
        plt.legend(['train', 'validation'], loc='lower right')
        plt.title('accuracy plot - train vs validation')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()
    else:
        pass
    return

def plot_loss(y):
    if(y == True):
        plt.plot(model_hist.history['loss'])
        plt.plot(model_hist.history['val_loss'])
        plt.legend(['training loss', 'validation loss'], loc = 'upper right')
        plt.title('loss plot - training vs vaidation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
    else:
        pass
    return


plot_accuracy(True)
plot_loss(True)
model_dl.save(os.path.join(model_path,model_name))

print("Done modelling...")

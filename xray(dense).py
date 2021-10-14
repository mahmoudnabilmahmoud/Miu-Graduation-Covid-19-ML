# -*- coding: utf-8 -*-
"""Xray(Dense)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wuzZSp_uF0b_CMR8vMtE_8s_6n00wgNr
"""

from google.colab import drive
drive.mount('/content/drive')

!unzip '/content/drive/MyDrive/Machine learning/KFold.zip'

from google.colab import files
files.upload() #upload kaggle.json

!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d tawsifurrahman/covid19-radiography-database

!unzip /content/covid19-radiography-database.zip

# Commented out IPython magic to ensure Python compatibility.
import os
import gc
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, roc_curve
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
import os
import shutil
import random
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from cv2 import *
from termcolor import colored
from google.colab import files
import sys
import csv
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model,Sequential, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50

!mkdir XTest
!mkdir XTest/COVID
!mkdir XTest/Normal

Orignal_FILE_PATH = "/content/COVID-19_Radiography_Dataset/COVID"
Target_Covid_Dir = "/content/XTest/COVID"

image_names = os.listdir(Orignal_FILE_PATH)
random.shuffle(image_names)
for i in range(1800):
    image_name = image_names[i]
    image_path = os.path.join(Orignal_FILE_PATH, image_name)
    target_path = os.path.join(Target_Covid_Dir, image_name)
    shutil.move(image_path, target_path)
    print("moving image covid ", i)

Orignal_FILE_PATH = "/content/COVID-19_Radiography_Dataset/Normal"
Target_Covid_Dir = "/content/XTest/Normal"

image_names = os.listdir(Orignal_FILE_PATH)
random.shuffle(image_names)
for i in range(1800):
    image_name = image_names[i]
    image_path = os.path.join(Orignal_FILE_PATH, image_name)
    target_path = os.path.join(Target_Covid_Dir, image_name)
    shutil.move(image_path, target_path)
    print("moving image normal ", i)

disease_types=['COVID', 'Normal']
data_dir = '/content/XTest'
train_dir = os.path.join(data_dir)

train_data = []
for defects_id, sp in enumerate(disease_types):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}/{}'.format(sp, file), defects_id, sp])
train = pd.DataFrame(train_data, columns=['File', 'DiseaseID','Disease Type'])
train.head()

SEED = 42
train = train.sample(frac=1, random_state=SEED) 
train.index = np.arange(len(train)) # Reset indices
train.head()

def plot_defects(defect_types, rows, cols):
    fig, ax = plt.subplots(rows, cols, figsize=(4, 4))
    defect_files = train['File'][train['Disease Type'] == defect_types].values
    n = 0
    for i in range(rows):
        for j in range(cols):
            image_path = os.path.join(data_dir, defect_files[n])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].imshow(cv2.imread(image_path))
            n += 1
# Displays first n images of class from training set
plot_defects('COVID', 3, 3)

plot_defects('Normal', 3, 3)

def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath)) # Loading a color image is the default flag
# Resize image to target size
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

from tqdm.notebook import tqdm
IMAGE_SIZE = 224

X = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        X[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
# Normalize the data
X = X / 255.
print('Train Shape: {}'.format(X.shape))

from keras.utils.np_utils import to_categorical
Y = train['DiseaseID'].values
Y = to_categorical(Y, num_classes=2)

BATCH_SIZE = 64

# Split the train and validation sets 
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=SEED)

EPOCHS = 50
SIZE=224
N_ch=3

def build_densenet():
    densenet = ResNet50(weights='imagenet', include_top=False)

    input = Input(shape=(SIZE, SIZE, N_ch))
    x = Conv2D(3, (3, 3), padding='same')(input)
    
    x = densenet(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # multi output
    output = Dense(2,activation = 'softmax', name='root')(x)
 

    # model
    model = Model(input,output)
    
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

model = build_densenet()
annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint('XmodelK4.h5', verbose=1, save_best_only=True)

plot_model(model, 
           show_shapes = True, 
           show_layer_names = True, 
           rankdir = 'TB', 
           expand_nested = False, 
           dpi = 60)

# Generates batches of image data with data augmentation
datagen = ImageDataGenerator(rotation_range=360, # Degree range for random rotations
                        width_shift_range=0.2, # Range for random horizontal shifts
                        height_shift_range=0.2, # Range for random vertical shifts
                        zoom_range=0.2, # Range for random zoom
                        horizontal_flip=True, # Randomly flip inputs horizontally
                        vertical_flip=True) # Randomly flip inputs vertically

datagen.fit(X_train)

hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
               steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
               epochs=EPOCHS,
               verbose=2,
               callbacks=[annealer, checkpoint],
               validation_data=(X_val, Y_val))

model.save('XK4.h5')

model = load_model('XmodelK4.h5')
final_loss, final_accuracy = model.evaluate(X_val, Y_val)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))

!mkdir K
!mkdir K/X_N
!mkdir K/X_C

Orignal_FILE_PATH = "/content/KFold/K4/Test/TestXray"
Target_Covid_Dir = "K/X_C"


image_names = os.listdir(Orignal_FILE_PATH)
#random.shuffle(image_names)
for i in range(400):
    image_name = image_names[i]
    if image_name.startswith('X_C'):
      image_path = os.path.join(Orignal_FILE_PATH, image_name)
      target_path = os.path.join(Target_Covid_Dir, image_name)
      shutil.move(image_path, target_path)
      print("moving X image covid ", i)

Orignal_FILE_PATH = "/content/KFold/K4/Test/TestXray"
Target_Covid_Dir = "K/X_N"


image_names = os.listdir(Orignal_FILE_PATH)
#random.shuffle(image_names)
for i in range(401):
    image_name = image_names[i]
    if image_name.startswith('X_N'):
      image_path = os.path.join(Orignal_FILE_PATH, image_name)
      target_path = os.path.join(Target_Covid_Dir, image_name)
      shutil.move(image_path, target_path)
      print("moving X image Normal ", i)

!mkdir /content/KFold/K4/Test/TestXray/X_N
!mkdir /content/KFold/K4/Test/TestXray/X_C

Orignal_FILE_PATH = "/content/K/X_C"
Target_Covid_Dir = "/content/KFold/K4/Test/TestXray/X_C"


image_names = os.listdir(Orignal_FILE_PATH)
#random.shuffle(image_names)
for i in range(202):
    image_name = image_names[i]
    if image_name.startswith('X_C'):
      image_path = os.path.join(Orignal_FILE_PATH, image_name)
      target_path = os.path.join(Target_Covid_Dir, image_name)
      shutil.move(image_path, target_path)
      print("moving CT image covid ", i)

Orignal_FILE_PATH = "/content/K/X_N"
Target_Covid_Dir = "/content/KFold/K4/Test/TestXray/X_N"


image_names = os.listdir(Orignal_FILE_PATH)
#random.shuffle(image_names)
for i in range(202):
    image_name = image_names[i]
    if image_name.startswith('X_N'):
      image_path = os.path.join(Orignal_FILE_PATH, image_name)
      target_path = os.path.join(Target_Covid_Dir, image_name)
      shutil.move(image_path, target_path)
      print("moving CT image Normal ", i)

disease_types=['X_C', 'X_N']
data_dir = '/content/KFold/K4/Test/TestXray'
train_dir = os.path.join(data_dir)

train_data = []
for defects_id, sp in enumerate(disease_types):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}/{}'.format(sp, file), defects_id, sp])
train = pd.DataFrame(train_data, columns=['File', 'DiseaseID','Disease Type'])
train.head()

SEED = 42
train = train.sample(frac=1, random_state=SEED) 
train.index = np.arange(len(train)) # Reset indices
train.head()

def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath)) # Loading a color image is the default flag
# Resize image to target size
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

from tqdm.notebook import tqdm
IMAGE_SIZE = 224

X = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        X[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
# Normalize the data
X = X / 255.
print('Train Shape: {}'.format(X.shape))

from keras.utils.np_utils import to_categorical
Y = train['DiseaseID'].values
Y = to_categorical(Y, num_classes=2)

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

Y_pred = model.predict(X)

Y_pred = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y, axis=1)

cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(6, 6))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=disease_types, yticklabels=disease_types)
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)

from sklearn.metrics import *
ture_negitive, false_Positive, false_negitive, true_Positive = confusion_matrix(Y_true, Y_pred).ravel()
Specificity = ture_negitive/(ture_negitive+false_Positive)
Sensitivity = true_Positive/(true_Positive+false_negitive)

print('f1 score =  %.3f'%f1_score(Y_true, Y_pred))
print('Sensitivity =  %.3f'%Sensitivity)
import math
from math import *
from sklearn.metrics import *
mse = mean_squared_error(Y_true, Y_pred)
rmse = math.sqrt(mse)
print("RMSE=",rmse)
from sklearn.metrics import mean_absolute_error
print('MAE=' ,mean_absolute_error(Y_true, Y_pred))
from sklearn.metrics import r2_score
R2= r2_score(Y_true, Y_pred)
print("R2 correlation=",R2)
from scipy.stats.stats import pearsonr
corr, _ = pearsonr(Y_true, Y_pred)
print('Pearsons correlation: %.15f' % corr)

from skimage import io
from keras.preprocessing import image
#path='imbalanced/Scratch/Scratch_400.jpg'
img = image.load_img('/content/KFold/K4/Test/TestXray/X_N/X_N532.png', grayscale=False, target_size=(224, 224))
show_img=image.load_img('/content/KFold/K4/Test/TestXray/X_N/X_N532.png', grayscale=False, target_size=(200, 200))
disease_class=['Covid','Non Covid']
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x /= 255

custom = model.predict(x)

plt.imshow(show_img)
plt.show()

a=custom[0]
ind=np.argmax(a)
        
print('Prediction:',disease_class[ind])
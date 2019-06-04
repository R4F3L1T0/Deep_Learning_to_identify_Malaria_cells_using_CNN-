#READING DATA TO NP ARRAYS
import warnings
warnings.filterwarnings('ignore')
#from __future__ import absolute_import, division, print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
#from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
import os
from keras.utils import to_categorical
from keras import backend as K
from keras import layers
#from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

infected = os.listdir('/home/raffaele/Scaricati/cell_images/Parasitized/') 
uninfected = os.listdir('/home/raffaele/Scaricati/cell_images/Uninfected/')

data = []
labels = []

for i in infected:
    try:
    
        image = cv2.imread("/home/raffaele/Scaricati/cell_images/Parasitized/"+i)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((64 , 64))
        data.append(np.array(resize_img))
        label = to_categorical(1, num_classes=2)
        labels.append(label)
        
    except AttributeError:
        print('COCCOS\' NUN STA IENN BUON')
    
for u in uninfected:
    try:
        
        image = cv2.imread("/home/raffaele/Scaricati/cell_images/Uninfected/"+u)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((64 , 64))
        data.append(np.array(resize_img))
        label = to_categorical(0, num_classes=2)
        labels.append(label)
        
    except AttributeError:
        print('COCCOS\' NUN STA IENN BUON')

data = np.array(data)
labels = np.array(labels)

#EXAMPLE PLOT
plt.figure(1, figsize = (15 , 7))
plt.subplot(1 , 2 , 1)
plt.imshow(data[0])
plt.title('Infected Cell')
plt.xticks([]) , plt.yticks([])

plt.subplot(1 , 2 , 2)
plt.imshow(data[15000])
plt.title('Uninfected Cell')
plt.xticks([]) , plt.yticks([])

plt.show()

n = np.arange(data.shape[0])
np.random.shuffle(n)
data = data[n]
labels = labels[n]


#NORMALIZING
data = data.astype(np.float32)
labels = labels.astype(np.int32)
data = data/255

from sklearn.model_selection import train_test_split
train_x , eval_x , train_y , eval_y = train_test_split(data, labels, test_size = 0.2, shuffle=True)
print('train data shape {} ,eval data shape {} '.format(train_x.shape, eval_x.shape))


tensorboard = keras.callbacks.TensorBoard(log_dir='../results/TensorBoard', histogram_freq=0, batch_size=150, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=False)

dr_strage = Sequential()
dr_strage.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
dr_strage.add(MaxPooling2D(pool_size = (2, 2)))
dr_strage.add(Conv2D(32, (3, 3), activation = 'relu'))
dr_strage.add(MaxPooling2D(pool_size = (2, 2)))
dr_strage.add(Conv2D(32, (3, 3), activation = 'relu'))
dr_strage.add(MaxPooling2D(pool_size = (2, 2)))
dr_strage.add(Flatten())
dr_strage.add(Dense(units = 128, activation = 'relu'))
dr_strage.add(Dropout(0.2))
dr_strage.add(Dense(units = 2, activation = 'softmax'))

dr_strage.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

dr_strage.fit(train_x,train_y,validation_data=(eval_x,eval_y), epochs=2000, batch_size=150,callbacks=[tensorboard])
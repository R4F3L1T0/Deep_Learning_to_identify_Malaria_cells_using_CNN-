import os
import cv2
import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import to_categorical

def read_data(path, discard=False):

    infected = os.listdir(path+'Parasitized/') 
    uninfected = os.listdir(path+'Uninfected/')

    data = []
    labels = []

    for i in infected:
        try:
        
            image = cv2.imread(path+'/Parasitized/'+i)
            image_array = Image.fromarray(image , 'RGB')
            resize_img = image_array.resize((64 , 64))
            data.append(np.array(resize_img))
            label = to_categorical(1, num_classes=2)
            labels.append(label)
            
        except AttributeError:
            print('COCCOS\' NUN STA IENN BUON')
        
    for u in uninfected:
        try:
            
            image = cv2.imread(path+'/Uninfected/'+u)
            image_array = Image.fromarray(image , 'RGB')
            resize_img = image_array.resize((64 , 64))
            data.append(np.array(resize_img))
            label = to_categorical(0, num_classes=2)
            labels.append(label)
            
        except AttributeError:
            print('COCCOS\' NUN STA IENN BUON')

    data = np.array(data)
    labels = np.array(labels)

    return data, labels

def normalize_data(data,labels):
    data = data.astype(np.float32)
    labels = labels.astype(np.int32)
    normalized_data = data/255

    return normalized_data

def plot_cells(data):
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

if __name__ == "__main__":
    
    path = '/home/raffaele/Scaricati/cell_images/'
    
    data, labels = read_data(path)

    plot_cells(data)

    data_norm = normalize_data(data,labels)
    plot_cells(data)

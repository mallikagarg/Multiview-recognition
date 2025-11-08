import keras.backend as K
import os
import tensorflow as tf
import sys
import numpy as np
import keras
from tensorflow import keras									
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import *
from keras.models import load_model
from tensorflow.keras import optimizers
from Rec_model import RecModel
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import argparse
#import tensorflow_addons as tfa
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
  
def recall(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision_m = precision(y_true, y_pred)
    recall_m = recall(y_true, y_pred)
    return 2*((precision_m*recall_m)/(precision_m+recall_m+K.epsilon()))    
       
def train(opt):  

    input_size=(opt.row,opt.col,opt.ch)

    #Data Generator

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rotation_range=50,
        rescale=1./255,
        shear_range=0.25,
        zoom_range=0.2,
        horizontal_flip=False,
        width_shift_range=0.15,
        height_shift_range=0.15,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        opt.train_ad,  # this is the target directory
        target_size=(opt.row, opt.col),  # all images will be resized to 150x150
        batch_size=opt.batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            opt.validation_ad,
            target_size=(opt.row, opt.col),
            batch_size=opt.batch_size,
            class_mode='categorical',shuffle=False)
    test_generator = test_datagen.flow_from_directory(
            opt.test_ad,
            target_size=(opt.row, opt.col),
            batch_size=opt.batch_size,
            class_mode='categorical',shuffle=False)

    # check the name of each class with corresponding indices using:
    # train_generator.class_indices
    #Compile####
    RecM=RecModel(input_size,opt.num_class)
    model=RecM.model_F
    _adam=optimizers.SGD(lr=opt.lr)
    #_adam=optimizers.Adam(lr=opt.lr, beta_1=0.9, beta_2=0.999, decay=0.0)
    #model.compile(loss='binary_crossentropy',optimizer = _adam,metrics=['accuracy',
    #                         tf.keras.metrics.Precision(),
    #                          tf.keras.metrics.Recall()])
    model.compile(loss='categorical_crossentropy',optimizer = _adam,metrics=['accuracy'])
    #log_dir = "logs/" +  datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=0)
    #callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',verbose=1, restore_best_weights=True)

    #model_checkpoint = ModelCheckpoint(opt.chekp+'.hdf5', monitor='val_accuracy',verbose=1, save_best_only=True)
    #with tf.device('/device:GPU:0'):

    model.load_weights(opt.chekp+'dp.hdf5')        
    meval = model.evaluate(test_generator,batch_size=opt.batch_size)
    print(meval)      
    
    ytrue=[]
    class_names=test_generator.labels
    #labels = (train_generator.class_indices)
    #labels = dict((v,k) for k,v in labels.items())
    #print(class_names)
    #print(labels)
    ypred=[]
    errors=0
    count=0  
    preds=model.predict(test_generator,verbose=1) # predict on the test data
    #print(preds)
    for i, p in enumerate(preds):
    	count +=1
    	#print(i)
    	#print(p)
    	index=np.argmax(p) # get index of prediction with highest probability
    	#print(index)
    	ypred.append(index) 
    	#print(class_names[i])
    	if index != class_names[i]:
        	errors +=1
    acc= (count-errors)* 100/count
    msg=f'there were {count-errors} correct predictions in {count} tests for an accuracy of {acc:6.2f} % '
    print(msg) 
    pred=np.array(ypred)
    #print(ypred)
    ytrue=np.array(ytrue)
    clr = classification_report(class_names, ypred)
    cm = confusion_matrix(class_names, ypred,normalize='true')
    print("Classification Report:\n----------------------\n", clr) 
    print("Classification Report:\n----------------------\n", cm) 
    classes=['A','B','C','D','E','F','H','I','J','K']
    con_mat_df = pd.DataFrame(cm,  index = classes, 
                     columns = classes)
    
    figure = plt.figure(figsize=(9, 9))
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('b.png')
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=2)
    # parser.add_argument('--input_size', type=list, default='(320,320,3)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--train_ad', type=str, default='')
    parser.add_argument('--validation_ad', type=str, default='')
    parser.add_argument('--test_ad', type=str, default='')

    parser.add_argument('--chekp', type=str, default='')
    parser.add_argument('--row', type=int, default=320)
    parser.add_argument('--col', type=int, default=320)
    parser.add_argument('--ch', type=int, default=3)
    parser.add_argument('--num_val', type=int, default=3)
    parser.add_argument('--num_img', type=int, default=3)
    parser.add_argument('--num_class', type=int, default=3)



    opt = parser.parse_args()
    train(opt)            

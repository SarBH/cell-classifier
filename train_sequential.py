from keras.preprocessing.image import ImageDataGenerator
from keras import applications, optimizers
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils

from matplotlib import pyplot as plt

import numpy as np
import logging, sys
import random
import sys


# Add your path folders for training, validation and test images. 
train_data_dir = '/home/nyscf/Desktop/Classification_Model/data/train/'
validation_data_dir = '/home/nyscf/Desktop/Classification_Model/data/validation/'
# test_data_dir = '../../../Desktop/Classification_Model/data/test/'

# NUMBER OF EXAMPLES - manual edit
num_train_samples = 4135
num_validation_samples=264

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-6, verbose=1)
early_stopper = EarlyStopping(min_delta=0.001, patience=20, verbose=1)
csv_logger = CSVLogger('/home/nyscf/Documents/sarita/cell-classifier/resnet_with_preprocessing.csv')



def train_model():
    """ Trains a CNN model and saves callbacks into model_path"""
  
    # SET THESE PARAMETERS. 
    m = 4135 #str(len(x_train[0]))
    mb_size = 10
    num_classes = 2
    num_epochs = 100
    img_size = (224,224)
    dropout_rate = 0.4
    model_type = "resnet"
    # EDITS END HERE. DO NOT TOUCH ANYTHING BELOW

    # create the data generator. Different models require different preprocessing and generating
    if model_type == "inception":
        datagen = ImageDataGenerator(rescale=1./255)
        train_datagen = datagen.flow_from_directory(train_data_dir, target_size=img_size, 
                                                class_mode='categorical', batch_size=mb_size)
        valid_datagen = datagen.flow_from_directory(validation_data_dir, target_size=img_size, 
                                                class_mode='categorical', batch_size=mb_size)

    elif model_type == "resnet":
        datagen = ImageDataGenerator(preprocessing_function=applications.resnet50.preprocess_input)
        train_datagen = datagen.flow_from_directory(train_data_dir, target_size=img_size, 
                                                class_mode='binary', batch_size=mb_size)
        valid_datagen = datagen.flow_from_directory(validation_data_dir, target_size=img_size, 
                                                class_mode='binary', batch_size=mb_size)
    
    elif model_type == "sequential":
        datagen = ImageDataGenerator(rescale=1./255)
        train_datagen = datagen.flow_from_directory(train_data_dir, target_size=img_size, 
                                                class_mode='categorical', batch_size=mb_size, color_mode='grayscale')
        valid_datagen = datagen.flow_from_directory(validation_data_dir, target_size=img_size, 
                                                class_mode='categorical', batch_size=mb_size, color_mode='grayscale')
    
    elif model_type == "densenet":
        datagen = ImageDataGenerator(preprocessing_function=applications.densenet.preprocess_input)
        train_datagen = datagen.flow_from_directory(train_data_dir, target_size=img_size, 
                                                class_mode='binary', batch_size=mb_size)
        valid_datagen = datagen.flow_from_directory(validation_data_dir, target_size=img_size, 
                                                class_mode='binary', batch_size=mb_size)
    
    
    # check that the generator is working properly
    batchX, batchy = train_datagen.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    
    # create base model
    if model_type == "inception":
        base_model = applications.inception_v3.InceptionV3(weights = None, include_top = False,
                                                input_shape = (img_size[0], img_size[1], 3))
    
    elif model_type == "resnet":
        base_model = applications.resnet50.ResNet50(include_top=False, weights=None, 
                                                input_shape = (img_size[0], img_size[1], 3), classes=num_classes)
    elif model_type == "sequential":
        NB_FILTER =32 # number of trainable features filters, so-called convolution but it is correlation in fact
        KERNAL_SIZE =(3,3) # convolution kernel size 3x3.
        POOL_SIZE =(2, 2) # size of pooling area for max pooling.
        STRIDES=(3,3) # pooling filter pixel move step, 3=move 3 pixels. For dimensional reduction
        NB_EPOCHS=3 # number of repeative trainings. 5 means 5 loops. Having 5 times of weight updates

        
        base_model = Sequential()
        base_model.add(Conv2D(NB_FILTER, kernel_size=KERNAL_SIZE, padding='same', input_shape=(img_size[0],img_size[1],1)))
        base_model.add(Activation('relu'))
        base_model.add(MaxPooling2D(pool_size=POOL_SIZE))
        base_model.add(Dropout(0.25))

        base_model.add(Conv2D(NB_FILTER*2, kernel_size=KERNAL_SIZE, strides=STRIDES))
        base_model.add(Activation('relu'))
        base_model.add(MaxPooling2D(pool_size=POOL_SIZE))
        base_model.add(Dropout(0.25))

        base_model.add(Flatten())
        base_model.add(Dense(64, activation='relu')) #try from 64,128,256,512. O?P layer=Relu. A.9.
        base_model.add(Dropout(0.5))
        base_model.add(Dense(2)) # 2=Number of classes to be classified, O?P layer=softmax
        base_model.add(Activation('softmax'))

    elif model_type == "densenet":
        base_model = applications.densenet.DenseNet121(include_top=False, weights=None, 
                                                input_shape = (img_size[0], img_size[1], 3), classes=num_classes)


    # specify last layer
    if model_type == "inception":
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout_rate)(x)
        predictions = Dense(2, activation = 'softmax')(x)
    
    elif model_type == "resnet":
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # x = Dropout(dropout_rate)(x)
        predictions = Dense(1, activation = 'sigmoid')(x)
        
    elif model_type == "sequential":
        predictions = base_model.output

    model = Model(inputs = base_model.input, outputs = predictions)

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)    

    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])
    
    model_path = "/home/nyscf/Documents/sarita/cell-classifier/sequential/chkpt_model.{epoch:02d}-acc{val_acc:.2f}.hdf5"#  _{model_type}_mb{mb_size}_m{m}_do{dropout_rate}   .format(model_type=model_type, mb_size=mb_size, m=m, dropout_rate=dropout_rate)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    history = model.fit_generator(train_datagen, 
                                steps_per_epoch=num_train_samples // mb_size, 
                                epochs=num_epochs,
                                validation_data=valid_datagen, 
                                validation_steps=8, 
                                shuffle=True,
                                verbose=1,
                                callbacks=[lr_reducer, early_stopper, csv_logger, checkpoint])
    
    # model.save("/home/nyscf/Documents/sarita/cell-classifier/model_resnet_mb10_m3500_e22_do2.h5")
    model.save("/home/nyscf/Documents/sarita/cell-classifier/model_{model_type}_mb{mb_size}_m{m}_e{num_epochs}_do{dropout_rate}.h5".format(model_type=model_type, mb_size=mb_size, m=m, num_epochs=num_epochs, dropout_rate=dropout_rate))


    return history





def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

  
def plot_results(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs,
            smooth_curve(acc), 'b', label='Training acc', color='red')
    plt.plot(epochs,
            smooth_curve(val_acc), 'b', label='Validation acc')
    plt.axhline(y=1, ls='dotted', color='k')
    plt.title('Training and validation accuracy - 1 hour')
    plt.legend()

    plt.figure()
    plt.plot(epochs,
            smooth_curve(loss), 'b', label='Training loss', color='red')
    plt.plot(epochs,
            smooth_curve(val_loss), 'b', label='Validation loss')
    plt.title('Training and validation loss - 1 hour')
    plt.legend()
    plt.figure()
    plt.show()


if __name__ == "__main__":
    history = train_model()
    plot_results(history)

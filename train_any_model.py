from keras.preprocessing.image import ImageDataGenerator
from keras import applications, optimizers
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import itertools
import matplotlib.pyplot as plt

import numpy as np
import logging, sys
import random
import sys


# Add your path folders for training, validation and test images. 
train_data_dir = '/home/nyscf/Desktop/Classification_Model/data/train_3_classes(BRIG)'


# SET THESE PARAMETERS. 
mb_size = 10
num_classes = 3
num_epochs = 50
img_size = (400,400)
model_type = "sequential"
validation_split = 0.15
num_of_test_samples = 892
model_version = "model_3"


# Define training checkpoint parameters
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-6, verbose=1, monitor='val_loss')
early_stopper = EarlyStopping(min_delta=0.001, patience=12, verbose=1)
csv_logger = CSVLogger("/home/nyscf/Documents/sarita/models/model_{model_type}_mb{mb_size}_c{num_classes}_imsize{img_size}.csv"
                        .format(model_type=model_type, mb_size=mb_size, num_classes=num_classes, img_size=str(img_size)))

 

def create_datagen(train_data_dir, img_size, mb_size):
    """create the data generator. Different models require different preprocessing and generating"""

    if model_type == "inception":
        datagen = ImageDataGenerator(preprocessing_function=applications.inception_v3.preprocess_input, validation_split=validation_split, horizontal_flip=True, vertical_flip=True)
        train_datagen = datagen.flow_from_directory(train_data_dir, target_size=img_size, 
                                                class_mode='categorical', batch_size=mb_size, subset='training')
        valid_datagen = datagen.flow_from_directory(train_data_dir, target_size=img_size, 
                                                class_mode='categorical', batch_size=mb_size, subset='validation') #save_to_dir="/home/nyscf/Desktop/Classification_Model/data/validation_from_datagen", save_prefix="datagen", save_format="jpeg"

    elif model_type == "resnet":
        datagen = ImageDataGenerator(preprocessing_function=applications.resnet50.preprocess_input, validation_split=0.1, horizontal_flip=True, vertical_flip=True)
        train_datagen = datagen.flow_from_directory(train_data_dir, target_size=img_size, 
                                                class_mode='binary', batch_size=mb_size, subset='training')
        valid_datagen = datagen.flow_from_directory(train_data_dir, target_size=img_size, 
                                                class_mode='binary', batch_size=mb_size, subset='validation')
    
    elif model_type == "sequential":
        datagen = ImageDataGenerator(rescale=1./255, fill_mode='wrap', validation_split=validation_split)
        train_datagen = datagen.flow_from_directory(train_data_dir, target_size=img_size, 
                                                class_mode='categorical', batch_size=mb_size, color_mode='grayscale', subset='training')
        valid_datagen = datagen.flow_from_directory(train_data_dir, target_size=img_size, 
                                                class_mode='categorical', batch_size=mb_size, color_mode='grayscale', subset='validation')
    
    elif model_type == "densenet":
        datagen = ImageDataGenerator(preprocessing_function=applications.densenet.preprocess_input, validation_split=validation_split, horizontal_flip=True, vertical_flip=True)
        train_datagen = datagen.flow_from_directory(train_data_dir, target_size=img_size, 
                                                class_mode='binary', batch_size=mb_size, subset='training')
        valid_datagen = datagen.flow_from_directory(train_data_dir, target_size=img_size, 
                                                class_mode='binary', batch_size=mb_size, subset='validation')
    
    # elif model_type == "resnet50_builder":

    
    # check that the generator is working properly
    batchX, batchy = train_datagen.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    return (train_datagen, valid_datagen)
    

def set_base_model(model_type):
    # create base model
    if model_type == "inception":
        base_model = applications.inception_v3.InceptionV3(weights = None, include_top = False,
                                                input_shape = (img_size[0], img_size[1], 3), classes=num_classes)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_classes, activation = 'softmax')(x)

    elif model_type == "resnet":
        base_model = applications.resnet50.ResNet50(include_top=False, weights=None, 
                                                input_shape = (img_size[0], img_size[1], 3), classes=num_classes)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(1, activation = 'sigmoid')(x)
    
    elif model_type == "sequential":
        NB_FILTER =32 # number of trainable features filters, so-called convolution but it is correlation in fact
        KERNAL_SIZE =(3,3) # convolution kernel size 3x3.
        POOL_SIZE =(2, 2) # size of pooling area for max pooling.
        STRIDES=(3,3) # pooling filter pixel move step, 3=move 3 pixels. For dimensional reduction
        
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
        base_model.add(Dense(num_classes)) # 2=Number of classes to be classified, O?P layer=softmax
        base_model.add(Activation('softmax'))
        predictions = base_model.output

    elif model_type == "densenet":
        base_model = applications.densenet.DenseNet121(include_top=False, weights=None, 
                                                input_shape = (img_size[0], img_size[1], 3), classes=num_classes)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)      
        predictions = Dense(num_classes, activation = 'softmax')(x)


    # elif model_type == "resnet50_builder":
    #     model = ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)   
    model = Model(inputs = base_model.input, outputs = predictions)



    
    return (model, predictions)


def train_model(train_datagen, model, predictions, valid_datagen, checkpoint_file=None):
    
    if checkpoint_file is not None:
        model.load_weights(checkpoint_file)

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)    
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"]) 
    
    print(train_datagen.classes)
    print(train_datagen.class_indices)

    model_path = "/home/nyscf/Documents/sarita/models/" + model_type + "/" + model_version + "/chkpt_model.{epoch:02d}-acc{val_acc:.2f}.hdf5" 
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    history = model.fit_generator(train_datagen, 
                                steps_per_epoch=train_datagen.samples // mb_size, 
                                epochs=num_epochs,
                                validation_data=valid_datagen, 
                                validation_steps=valid_datagen.samples // mb_size, 
                                shuffle=True,
                                verbose=1,
                                class_weight='balanced',
                                callbacks=[lr_reducer, early_stopper, csv_logger, checkpoint], 
                                initial_epoch=initial_epoch)

    # #Confution Matrix and Classification Report
    # Y_pred = model.predict_generator(valid_datagen, num_of_test_samples // mb_size+1)
    # y_pred = np.argmax(Y_pred, axis=1)
    # print('Confusion Matrix')
    # print(confusion_matrix(valid_datagen.classes, y_pred))
    # print('Classification Report')
    # target_names = ['dead', 'diff', 'viable']
    # print(classification_report(valid_datagen.classes, y_pred, target_names=target_names))

    # model.save("/home/nyscf/Documents/sarita/cell-classifier/model_resnet_mb10_m3500_e22_do2.h5")
    model.save("/home/nyscf/Documents/sarita/models/inception_3_class/" + model_type + "/model_{model_type}_mb{mb_size}_m{m}_e{num_epochs}_c{num_classes}_imsize{img_size}.h5"
                .format(model_type=model_type, mb_size=mb_size, m=train_datagen.samples, num_epochs=num_epochs, num_classes=num_classes, img_size=str(img_size)))

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
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs,
            smooth_curve(loss), 'b', label='Training loss', color='red')
    plt.plot(epochs,
            smooth_curve(val_loss), 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.figure()
    plt.show()



if __name__ == "__main__":
    if len(sys.argv) == 1:   
        print("\n\nInitializing training from zero\n\n") 
        train_datagen, valid_datagen= create_datagen(train_data_dir, img_size, mb_size)
        model, predictions = set_base_model(model_type)
        initial_epoch = 0        
        history = train_model(train_datagen, model, predictions, valid_datagen)

        plot_results(history)
    
    else:
        print("\n\nInitializing training from checkpoint\n\n")
        model_type = sys.argv[1]
        checkpoint_file = sys.argv[2]
        initial_epoch = int(sys.argv[3])
        train_datagen, valid_datagen= create_datagen(train_data_dir, img_size, mb_size)
        model, predictions = set_base_model(model_type)
        history = train_model(train_datagen, model, predictions, valid_datagen, checkpoint_file=checkpoint_file)
        
        plot_results(history)
    



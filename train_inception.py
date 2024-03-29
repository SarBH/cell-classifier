import os
import cv2
import numpy as np
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
from keras import applications, callbacks
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from sklearn.model_selection import train_test_split
import logging, sys
import random
import sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logging.debug('A debug message!')



def load_data(dir, max_examples, resize_height, resize_width, label, m_idx):
    """ Loads data from directory dir into a global data list.
        Saves labels into a global labels list. 
        max_examples defines the maximum number of images to import.
        """
    for idx, i in enumerate(os.listdir(dir)):
        if idx == max_examples: # only read max_examples number of examples
            break
        print("Reading " + label + " " + str(idx))
        img = cv2.imread(dir + "/" + i)
        img = cv2.resize(img, (resize_height, resize_width))
        
        # append data and labels to lists
        # assert(img[:,:,0].all() == img[:,:,1].all() == img[:,:,2].all()) # assert image is grayscale and removing dimension wont affect training
        # img = img[:,:,0] # remove the RGB color dimension (no new data)
        assert(data[m_idx,:,:,:].shape == img.shape) # asserts image matrix is of the intended shape, and fits data matrix
        data[m_idx,:,:,:] = img #add the array to data
        labels.append(label)
        m_idx = m_idx + 1
    
    # assert all labels went into the lists
    assert(len(labels) == m_idx)

    return (data, labels, m_idx)


def encode_labels(labels, tag_1, tag_2):
    """Takes in labels from global list and translate them into a OneHot encoded list of np.arrays
    """
    y_train = []
    for tag in labels:
        if tag == tag_1:
            oa = np.array([0,1])
        elif tag == tag_2:
            oa = np.array([1,0])
        else:
            print("FOUND INSTANCE IN MISMATCHING CATEGORY")
        y_train.append(oa)
    y_train = np.array(y_train)
    return y_train


def standardize_data(data):
    """ Transform the data from a list of objects to a np.array of floats. 
    Then normalizes so grayscale pixel values range between 0 and 1
    """
    x_train = np.array(data, dtype = "float") / 255.0
    print(x_train.shape)
    # x_train = x_train.reshape(2000,300,300,3)
    return x_train


def shuffle_data(x_train, y_train, seed=1):
    """ Shuffles training data and labels using the same seed."""
    random.Random(seed).shuffle(x_train)
    random.Random(seed).shuffle(y_train)
    return (x_train, y_train)


def train_model(x_train, y_train, with_validation=False):
    """ Trains a CNN model and saves callbacks into model_path"""
    if with_validation == True:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state = 25)

    # SET THESE PARAMETERS. 
    m = str(len(x_train[0]))
    mb_size = 10
    num_epochs = 30
    dropout_rate = 0.3
    model_type = "inception"
    # YOUR EDITS END HERE. DO NOT TOUCH ANYTHING BELOW

    if model_type == "inception":
        base_model = applications.inception_v3.InceptionV3(weights = None, include_top = False,
                                                input_shape = (300, 300, 3))
    elif model_type == "resnet":
        base_model = applications.resnet50.resnet50.ResNet50(input_shape=(300, 300, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(2, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)

    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    
    
    if with_validation == True:
        model_path = "/home/nyscf/Documents/sarita/cell-classifier/chkpt_model.{epoch:02d}-acc{val_acc:.2f}.hdf5"#  _{model_type}_mb{mb_size}_m{m}_do{dropout_rate}   .format(model_type=model_type, mb_size=mb_size, m=m, dropout_rate=dropout_rate)
        checkpoint = callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        callback_list = [checkpoint]
        H = model.fit(x_train, y_train, epochs = num_epochs, batch_size = mb_size, validation_data = (x_test, y_test), callbacks=callback_list)
    else:
        H = model.fit(x_train, y_train, epochs = num_epochs, batch_size = mb_size)

    scores = model.evaluate(x_train, y_train, verbose=1)
    print("the model scores are:", scores)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    model.save("/home/nyscf/Documents/sarita/cell-classifier/model_resnet_mb10_m3500_e22_do2.h5")
    model.save("/home/nyscf/Documents/sarita/cell-classifier/model_{model_type}_mb{mb_size}_m{m}_e{num_epochs}_do{dropout_rate}.h5".format(model_type=model_type, mb_size=mb_size, m=m, num_epochs=num_epochs, dropout_rate=dropout_rate))



if __name__ == "__main__":
    if len(sys.argv) == 1:
        curr_m_idx = 0
        data = np.empty((4000, 300, 300, 3), dtype=np.uint8)
        print(sys.getsizeof(data))
        labels = []

        data, labels, m_idx = load_data('/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/Colonies_Viable/Original', 500, 300, 300, 'viable', curr_m_idx)
        curr_m_idx = m_idx
        data, labels, m_idx = load_data('/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/Colonies_Viable/rotate_90', 500, 300, 300, 'viable',curr_m_idx)
        curr_m_idx = m_idx
        data, labels, m_idx = load_data('/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/Colonies_Viable/rotate_180', 500, 300, 300, 'viable',curr_m_idx)
        curr_m_idx = m_idx
        data, labels, m_idx = load_data('/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/Colonies_Viable/rotate_270', 500, 300, 300, 'viable',curr_m_idx)
        curr_m_idx = m_idx
        print("data has this many bytes: ", sys.getsizeof(data))
        print(len(data))

        data, labels, m_idx = load_data('/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/Diff_And_Bad_Colonies/Original', 500, 300, 300, 'unviable',curr_m_idx)
        curr_m_idx = m_idx
        data, labels, m_idx = load_data('/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/Diff_And_Bad_Colonies/rotate_90', 500, 300, 300, 'unviable',curr_m_idx)
        curr_m_idx = m_idx
        data, labels, m_idx = load_data('/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/Diff_And_Bad_Colonies/rotate_180', 500, 300, 300, 'unviable',curr_m_idx)
        curr_m_idx = m_idx    
        data, labels, m_idx = load_data('/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/Diff_And_Bad_Colonies/rotate_270', 500, 300, 300, 'unviable',curr_m_idx)
        print("data has this many bytes: ", sys.getsizeof(data))
        print(len(data))

        y_train = encode_labels(labels, 'viable', 'unviable')
        x_train = standardize_data(data)

        np.save("/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/x_train_4000.npy", x_train)
        np.save("/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/y_train_4000.npy", y_train)

    elif sys.argv[1] == 'load_examples':
        x_train = np.load("/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/x_train_4000.npy")
        y_train = np.load("/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/train/y_train_4000.npy")
        x_train = x_train[:3968,:,:,:]
        y_train = y_train[:3968]

    x_train, y_train = shuffle_data(x_train, y_train)
    train_model(x_train, y_train, with_validation=True)


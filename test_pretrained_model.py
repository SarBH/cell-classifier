import os
import cv2
import numpy as np
from keras import applications
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from train_inception import encode_labels, standardize_data, shuffle_data


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



if __name__ == "__main__":
	curr_m_idx = 0
	data = np.empty((64, 300, 300, 3), dtype=np.uint8)
	labels = []

	data, labels, m_idx = load_data("/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/valid/Colonies_Viable_Val", 32, 300,300, "viable", curr_m_idx)
	curr_m_idx = m_idx
	data, labels, m_idx = load_data("/home/nyscf/Desktop/Classification_Model/Initial_Training_Set/valid/Diff_And_Bad_Val", 32, 300,300, "unviable", curr_m_idx)
	curr_m_idx = m_idx

	y_val = encode_labels(labels, 'viable', 'unviable')
	x_val = standardize_data(data)



	loaded_model = load_model("/home/nyscf/Documents/sarita/clas/test_colony_inception_mb10_m4000_e30_do1.h5")

	score = loaded_model.evaluate(x_val, y_val, verbose=1)
	print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# predictions = model.predict(Xtest, batch_size=None, verbose=0, steps=None, callbacks=None)


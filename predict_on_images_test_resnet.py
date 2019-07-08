from keras.models import load_model
import os
import cv2
import numpy as np
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from keras import applications, optimizers



model_path = "/home/nyscf/Documents/sarita/models/resnet/chkpt_model.23-acc0.95.hdf5"

img_size = (300,300)
valid_batch_size = 1
num_classes = 2





base_model = applications.resnet50.ResNet50(include_top=False, weights=None, 
                                                input_shape = (img_size[0], img_size[1], 3), classes=num_classes)
	
top_model = Sequential()
top_model.add(Dense(1, activation='sigmoid'))
output_layer = base_model.output
output_layer = GlobalAveragePooling2D()(output_layer)
# top_model.add(output_layer)

model = Model(inputs=base_model.input, outputs=top_model(output_layer))

model.load_weights(model_path)

model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics = ["acc"]) #used accuracy metric for 2 classes, supposedly catetgorical acc gets selected automatically when using cat cross ent

# counter = 0
for i in os.listdir("/home/nyscf/Desktop/Classification_Model/data/test/first100"):
    image = cv2.imread("/home/nyscf/Desktop/Classification_Model/data/test/first100/" + i)
    out = image.copy()
    image = cv2.resize(image, img_size)

    image = image.astype("float") / 255.0
    images = [image]
    images = np.array(images)

    prediction = model.predict(images, verbose=1, batch_size=1)
    print("Prediction for ", i, "= " , str(prediction))

    if prediction[0][0] < 0.5:
        c = "bad "+str(np.around(prediction[0][1], decimals=2))
        color = (255, 0, 0)
        cv2.putText(out, c, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imwrite("/home/nyscf/Desktop/Classification_Model/data/test/resnet/first100_bad/" + i + ".jpg", out)
        
    else:
        color = (0, 255, 0)
        c = "good "+str(np.around(prediction[0][0], decimals=2))
        cv2.putText(out, c, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imwrite("/home/nyscf/Desktop/Classification_Model/data/test/resnet/first100_good/" + i + ".jpg", out)

    # cv2.putText(out, c, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # cv2.imwrite("/home/nyscf/Desktop/labeled_colony_by_model/day9/all_labels/" + i + "__" + str(counter) + ".jpg", out)
    # counter += 1

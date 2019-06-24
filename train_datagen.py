from keras.preprocessing.image import ImageDataGenerator
from keras import applications, callbacks
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from sklearn.model_selection import train_test_split
import logging, sys
import random
import sys



 # SET THESE PARAMETERS. 
# m = str(len(x_train[0]))
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


# create a data generator
datagen = ImageDataGenerator(rescale=1./255)

train_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/train/', target_size=(300,300), class_mode='categorical', batch_size=32)

val_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/validation/', target_size=(300,300), class_mode='categorical', batch_size=32)

batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



# test_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/test/', class_mode='binary', batch_size=32)

x = base_model.output

x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)


predictions = Dense(2, activation = 'softmax')(x)

# predictions = to_categorical(predictions)
model = Model(inputs = base_model.input, outputs = predictions)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = [categorical_accuracy])

model.fit_generator(train_it, steps_per_epoch=16, epochs=10, validation_data=val_it, validation_steps=8)

# scores = model.evaluate(x_train, y_train, verbose=1)
# print("the model scores are:", scores)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
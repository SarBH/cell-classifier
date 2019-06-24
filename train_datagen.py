from keras.preprocessing.image import ImageDataGenerator
from keras import applications, callbacks, optimizers
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from sklearn.model_selection import train_test_split
import logging, sys
import random
import sys


def train_model():
    """ Trains a CNN model and saves callbacks into model_path"""
  
    # SET THESE PARAMETERS. 
    m = 2000 #str(len(x_train[0]))
    mb_size = 10
    num_epochs = 30
    dropout_rate = 0.1
    model_type = "inception"
    # YOUR EDITS END HERE. DO NOT TOUCH ANYTHING BELOW


    # create a data generator
    datagen = ImageDataGenerator(rescale=1./255)
    train_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/train/', target_size=(300,300), class_mode='binary', batch_size=32)
    val_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/validation/', target_size=(300,300), class_mode='binary', batch_size=32)
    batchX, batchy = train_it.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    # create base model
    if model_type == "inception":
        base_model = applications.inception_v3.InceptionV3(weights = None, include_top = False,
                                                input_shape = (300, 300, 3))
    elif model_type == "resnet":
        base_model = applications.resnet50.resnet50.ResNet50(input_shape=(300, 300, 3))


    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation = 'sigmoid')(x)
    model = Model(inputs = base_model.input, outputs = predictions)

    opt = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)    

    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["binary_accuracy"])
    
    model_path = "/home/nyscf/Documents/sarita/cell-classifier/chkpt_model.{epoch:02d}-.hdf5"# acc{val_acc:.2f}.hdf5  _{model_type}_mb{mb_size}_m{m}_do{dropout_rate}   .format(model_type=model_type, mb_size=mb_size, m=m, dropout_rate=dropout_rate)
    checkpoint = callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callback_list = [checkpoint]
    model.fit_generator(train_it, steps_per_epoch=16, epochs=80, validation_data=val_it, validation_steps=8, callbacks=callback_list)


    scores = model.evaluate(verbose=1)
    print("the model scores are:", scores)

    # model.save("/home/nyscf/Documents/sarita/cell-classifier/model_resnet_mb10_m3500_e22_do2.h5")
    model.save("/home/nyscf/Documents/sarita/cell-classifier/model_{model_type}_mb{mb_size}_m{m}_e{num_epochs}_do{dropout_rate}.h5".format(model_type=model_type, mb_size=mb_size, m=m, num_epochs=num_epochs, dropout_rate=dropout_rate))

if __name__ == "__main__":
    train_model()



"""
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

train_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/train/', target_size=(300,300), class_mode='binary', batch_size=32)

val_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/validation/', target_size=(300,300), class_mode='binary', batch_size=32)

batchX, batchy = tr# create a data generator
datagen = ImageDataGenerator(rescale=1./255)

train_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/train/', target_size=(300,300), class_mode='binary', batch_size=32)

val_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/validation/', target_size=(300,300), class_mode='binary', batch_size=32)

batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



# test_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/test/', class_mode='binary', batch_size=32)

x = base_model.output

x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)


model = Model(inputs = base_model.input, outputs = predictions)

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit_generator(train_it, steps_per_epoch=16, epochs=10, validation_data=val_it, validation_steps=8)

# scores = model.evaluate(x_train, y_train, verbose=1)
# print("the model scores are:", scores)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = Dense(1, activati
# predictions = to_categorical(predictions)

print('Batch shape=# create a data generator
datagen = ImageDataGenerator(rescale=1./255)

train_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/train/', target_size=(300,300), class_mode='binary', batch_size=32)

val_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/validation/', target_size=(300,300), class_mode='binary', batch_size=32)

batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



# test_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/test/', class_mode='binary', batch_size=32)

x = base_model.output

x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)


model = Model(inputs = base_model.input, outputs = predictions)

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit_generator(train_it, steps_per_epoch=16, epochs=10, validation_data=val_it, validation_steps=8)

# scores = model.evaluate(x_train, y_train, verbose=1)
# print("the model scores are:", scores)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = Dense(1, activati
# predictions = to_categorical(predictions)
 max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



# test_it = datagen# create a data generator
datagen = ImageDataGenerator(rescale=1./255)

train_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/train/', target_size=(300,300), class_mode='binary', batch_size=32)

val_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/validation/', target_size=(300,300), class_mode='binary', batch_size=32)

batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



# test_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/test/', class_mode='binary', batch_size=32)

x = base_model.output

x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)


model = Model(inputs = base_model.input, outputs = predictions)

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit_generator(train_it, steps_per_epoch=16, epochs=10, validation_data=val_it, validation_steps=8)

# scores = model.evaluate(x_train, y_train, verbose=1)
# print("the model scores are:", scores)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = Dense(1, activati
# predictions = to_categorical(predictions)
rectory('/home/nyscf/Desktop/Classification_Model/data/test/', class_mode='binary', batch_size=32)

x = base_model.outp# create a data generator
datagen = ImageDataGenerator(rescale=1./255)

train_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/train/', target_size=(300,300), class_mode='binary', batch_size=32)

val_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/validation/', target_size=(300,300), class_mode='binary', batch_size=32)

batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



# test_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/test/', class_mode='binary', batch_size=32)

x = base_model.output

x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)


model = Model(inputs = base_model.input, outputs = predictions)

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit_generator(train_it, steps_per_epoch=16, epochs=10, validation_data=val_it, validation_steps=8)

# scores = model.evaluate(x_train, y_train, verbose=1)
# print("the model scores are:", scores)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = Dense(1, activati
# predictions = to_categorical(predictions)


x = GlobalAveragePo# create a data generator
datagen = ImageDataGenerator(rescale=1./255)

train_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/train/', target_size=(300,300), class_mode='binary', batch_size=32)

val_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/validation/', target_size=(300,300), class_mode='binary', batch_size=32)

batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



# test_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/test/', class_mode='binary', batch_size=32)

x = base_model.output

x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)


model = Model(inputs = base_model.input, outputs = predictions)

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit_generator(train_it, steps_per_epoch=16, epochs=10, validation_data=val_it, validation_steps=8)

# scores = model.evaluate(x_train, y_train, verbose=1)
# print("the model scores are:", scores)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = Dense(1, activati
# predictions = to_categorical(predictions)

x = Dropout(0.2)(x)# create a data generator
datagen = ImageDataGenerator(rescale=1./255)

train_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/train/', target_size=(300,300), class_mode='binary', batch_size=32)

val_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/validation/', target_size=(300,300), class_mode='binary', batch_size=32)

batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



# test_it = datagen.flow_from_directory('/home/nyscf/Desktop/Classification_Model/data/test/', class_mode='binary', batch_size=32)

x = base_model.output

x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)


model = Model(inputs = base_model.input, outputs = predictions)

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit_generator(train_it, steps_per_epoch=16, epochs=10, validation_data=val_it, validation_steps=8)

# scores = model.evaluate(x_train, y_train, verbose=1)
# print("the model scores are:", scores)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = Dense(1, activati
# predictions = to_categorical(predictions)



model = Model(inputs = base_model.input, outputs = predictions)

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit_generator(train_it, steps_per_epoch=16, epochs=10, validation_data=val_it, validation_steps=8)

# scores = model.evaluate(x_train, y_train, verbose=1)
# print("the model scores are:", scores)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = Dense(1, activati
# predictions = to_categorical(predictions)
"""
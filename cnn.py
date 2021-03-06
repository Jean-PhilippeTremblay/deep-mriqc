'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import data_grab
import os
import pickle
import numpy as np

batch_size = 32
num_classes = 2
epochs = 10
data_augmentation = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_deepmriqc_cnnv1_trained_model.h5'

# The data, shuffled and split between train and test sets:

#test_proportion of 3 means 1/3 so 33% test and 67% train
def shuffle(matrix, target, test_proportion):
    ratio = int(matrix.shape[0]/test_proportion)
    x_train = matrix[ratio:,:]
    x_test =  matrix[:ratio,:]
    y_train = target[ratio:,:]
    y_test =  target[:ratio,:]
    return x_train, x_test, y_train, y_test

def gen_portion(indexes, data, portion=3):
    ratio = int(indexes.shape[0] / portion)
    train = data[ratio:,:]
    test = data[:ratio,:]
    return train, test

##Grabbing
dat, lab = data_grab.all_data()
lab = [0 if x==-1 else 1 for x in lab]
dat_n = np.array(dat)
lab_n = np.array(lab)
lab_n = np.expand_dims(lab_n, axis=1)

## Shuffle at subject level - not slice level
randomize = np.arange(len(dat))
np.random.shuffle(randomize)
dat_n = dat_n[randomize]
lab_n = lab_n[randomize]

dat_index = randomize
dat_index = np.expand_dims(dat_index, axis=1)

indexes_train_subjects, indexes_test_subjects = gen_portion(dat_index, dat_index)
x_train_subjects, x_test_subjects = gen_portion(dat_index, dat_n)
y_train_subjects, y_test_subjects = gen_portion(dat_index, lab_n)

train_n = indexes_train_subjects.shape[0]
test_n = indexes_test_subjects.shape[0]

x_train = x_train_subjects.reshape(train_n*80, 80, 80)
x_test = x_test_subjects.reshape(test_n*80, 80, 80)
y_train = np.repeat(y_train_subjects, 80)
y_test = np.repeat(y_test_subjects, 80)

x_train = x_train.reshape(x_train.shape[0], 80, 80, 1)
x_test = x_test.reshape(x_test.shape[0], 80, 80, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train/=255
x_test/=255

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Evaluate model with test data set and share sample prediction results
evaluation = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Model Accuracy on slices = %.2f' % (evaluation[1]))

predictions_slices = model.predict(x_test, batch_size=batch_size)
predicted_labels = (predictions_slices[:,1]>0.5)*1
actual_labels = y_test[:,1]

actual_labels_avged = np.mean(actual_labels.reshape(-1, 80), axis=1)
predicted_labels_aved = np.sign(np.mean(predicted_labels.reshape(-1, 80), axis=1))

acc_count = np.sum((actual_labels_avged == predicted_labels_aved)*1)

image_acc = acc_count/test_n*100

print('Model Accuracy on images = %.2f' % (image_acc))

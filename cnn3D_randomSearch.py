'''A deep-learning approach to MRIQC. Using subjects from the publicly accessible ABIDE
dataset and ratings from expert reviewer(s), we trained a 4-layer convolutional neural
network (CNN) to classify raw MRI images as high- or low-quality.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cnn3D.py
'''

from __future__ import print_function

# for reproducibility
import numpy as np
#np.random.seed(1234)

import csv
import warnings
import datetime
import os
import copy
import random as rnd

from sklearn.model_selection import train_test_split

import sys,inspect, multiprocessing
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
data_dir='{0}/../data'.format(currentdir)
sys.path.insert(0,currentdir + '/gen_2Dslices')
import data_grab


# Hide messy TensorFlow and Numpy warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Set some hyperparameters
batch_size = 16
num_classes = 2
epochs = 200
data_augmentation = False
save_dir = os.path.join(currentdir, '/saved_models')
model_name = 'keras_deepmriqc_cnnv13D_trained_model.h5'

# Generate UNIX time stamp
def getUTC():
    return str(int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()))

# Save data in csv file
def saveExperiment(fileName, data):
    csvFile = open('{0}/Data/'.format(currentdir) + fileName + '.r.csv', 'a')
    writer = csv.writer(csvFile)
    # Check if file is empty, then write the header

    if os.stat('{0}/Data/'.format(currentdir) + fileName + '.r.csv').st_size == 0:
        dataLabels = ['UTC_local', 'Max val_acc', 'modelIndex', 'filters', 'filter_size', 'pool_size', 'dense_size', 'dropout', 'lr', 'decay']
        writer.writerow(dataLabels)


    writer.writerow(data)

    csvFile.close()

# Generate training and testing data sets
def get_datasets(test=True):
    import keras
    ##Grabbing
    dat, lab = data_grab.all_data(currentdir, data_dir)
    lab = [0 if x==-1 else 1 for x in lab]
    dat_n = np.array(dat)
    lab_n = np.array(lab)
    lab_n = np.expand_dims(lab_n, axis=1)

    ## Shuffle at subject level - not slice level
    dat_index = np.arange(len(dat))
    dat_index = np.expand_dims(dat_index, axis=1)

    if test:
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(dat_index, lab_n, test_size=0.3, stratify=lab_n)
        x_train = dat_n[X_train_idx]
        x_test = dat_n[X_test_idx]

        x_train = x_train.reshape(x_train.shape[0], 80, 80, 80, 1)
        x_test = x_test.reshape(x_test.shape[0], 80, 80, 80, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train /= 255
        x_test /= 255

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    else:
        x_train = dat_n
        x_train = x_train.reshape(x_train.shape[0], 80, 80, 80, 1)
        x_train = x_train.astype('float32')
        x_train /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        y_train = keras.utils.to_categorical(lab_n, num_classes)

        x_test=None
        y_test=None
        
        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)

    return x_train, y_train, x_test,y_test

# Construct the model using hyperparameters defined as arguments

def do_run(i, x_train=None, y_train=None, res_dict=None):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Conv3D, MaxPooling3D
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ReduceLROnPlateau

    UTC_local = getUTC()  # Randomly generate hyperparameters

    optimizer_dict = {  # this one is not used yet
        0: 'adam',
        1: 'rmsprop',
        2: 'adadelta'
    }

    modelIndex_dict = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5
    }

    filters_dict = {
        0: 8,
        1: 16,
        2: 24
    }

    filterSize_dict = {
        0: 4,
        1: 8
    }

    poolSize_dict = {
        0: 4,
        1: 2,
        2: 3
    }

    denseSize_dict = {
        0: 256,
        1: 512,
        2: 1024
    }

    dropout_dict = {
        0: 0,
        1: 0.1,
        2: 0.2,
        3: 0.3,
        4: 0.4
    }

    lr_dict = {
        0: 0.001,
        1: 0.0001,
        2: 0.00001
    }

    decay_dict = {
        0: 1e-04,
        1: 1e-05,
        2: 1e-06,
        3: 1e-07
    }

    modelIndex = modelIndex_dict.get(rnd.randint(0, len(modelIndex_dict) - 1))
    filters = filters_dict.get(rnd.randint(0, len(filters_dict) - 1))
    filter_size = filterSize_dict.get(rnd.randint(0, len(filterSize_dict) - 1))
    pool_size = poolSize_dict.get(rnd.randint(0, len(poolSize_dict) - 1))
    dense_size = denseSize_dict.get(rnd.randint(0, len(denseSize_dict) - 1))
    dropout = dropout_dict.get(rnd.randint(0, len(dropout_dict) - 1))
    lr = lr_dict.get(rnd.randint(0, len(lr_dict) - 1))
    decay = decay_dict.get(rnd.randint(0, len(decay_dict) - 1))

    print('######### DEBUG - MAKE_MODEL - params')
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    print('function name "%s"' % inspect.getframeinfo(frame)[2])
    for g in args:
        if 'train' in g:
            continue
        print("    %s = %s" % (g, values[g]))

    print('modelIndex -' + str(modelIndex))
    print('filters - ' + str(filters))
    print('filter_size - ' + str(filter_size))
    print('pool_size - ' + str(pool_size))
    print('dense_size - ' + str(dense_size))
    print('dropout - ' + str(dropout))
    print('lr - ' + str(lr))
    print('decay - ' + str(decay))


    print('#########')
    model = Sequential()
    input_shape = x_train.shape[1:]

    if modelIndex == 1:

        # 1x Conv+relu+MaxPool+Dropout -> 1x Dense+relu+Dropout
        model.add(Conv3D(filters, (filter_size, filter_size, filter_size), padding='same',
                         input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(pool_size, pool_size, pool_size)))
        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(dense_size))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

    elif modelIndex == 2:
        # 2x Conv+relu+MaxPool+Dropout -> 1x Dense+relu+Dropout
        model.add(Conv3D(filters, (filter_size, filter_size, filter_size), padding='same',
                         input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(pool_size, pool_size, pool_size)))
        model.add(Dropout(dropout))

        model.add(Conv3D(pool_size * filters, (filter_size, filter_size, filter_size), padding='same'))
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(pool_size, pool_size, pool_size)))
        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(dense_size))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

    elif modelIndex == 3:
        # 2x Conv+relu+MaxPool+Dropout -> 2x Dense+relu+Dropout
        model.add(Conv3D(filters, (filter_size, filter_size, filter_size), padding='same',
                         input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(pool_size, pool_size, pool_size)))
        model.add(Dropout(dropout))

        model.add(Conv3D(pool_size * filters, (filter_size, filter_size, filter_size), padding='same'))
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(pool_size, pool_size, pool_size)))
        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(dense_size))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

        model.add(Dense(dense_size))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

    elif modelIndex == 4:
        # 2x Conv+relu+Conv+relu+MaxPool+Droput -> 1x Dense+relu+Dropout
        model.add(Conv3D(filters, (filter_size, filter_size, filter_size), padding='same',
                         input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv3D(filters, (filter_size, filter_size, filter_size)))
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(pool_size, pool_size, pool_size)))
        model.add(Dropout(dropout))

        model.add(Conv3D(pool_size * filters, (filter_size, filter_size, filter_size), padding='same'))
        model.add(Activation('relu'))

        model.add(Conv3D(pool_size * filters, (filter_size, filter_size, filter_size)))
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(pool_size, pool_size, pool_size)))
        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(dense_size))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

    elif modelIndex == 5:
        model.add(Conv3D(filters, (filter_size, filter_size, filter_size), padding='same',
                         input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv3D(filters, (filter_size, filter_size, filter_size)))
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(pool_size, pool_size, pool_size)))
        model.add(Dropout(dropout))

        model.add(Conv3D(pool_size * filters, (filter_size, filter_size, filter_size), padding='same'))
        model.add(Activation('relu'))

        model.add(Conv3D(pool_size * filters, (filter_size, filter_size, filter_size)))
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(pool_size, pool_size, pool_size)))
        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(dense_size))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.Adam(lr=lr, decay=decay)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

        
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    print('Not using data augmentation.')
    print('In loop - {0}'.format(i))
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.25,
                        shuffle=True,
                        callbacks=[early_stopping, reduce_lr])

    val_acc = max(np.array(history.history['val_acc']))
    print('Max Validation Accuracy - ' + str(acc))

    data = (UTC_local, val_acc, modelIndex, filters, filter_size,
            pool_size, dense_size, dropout, lr, decay)
    res_dict[i] = data


# generate UNIX time stamp
UTC_global = getUTC()

# Generate training and testing datasets
x_train, y_train, x_test, y_test = get_datasets(test=False)

manager = multiprocessing.Manager()
res_dict = manager.dict()
jobs = []
for i in range(100):
    p = multiprocessing.Process(target=do_run, args=[i], kwargs=dict(x_train=x_train, y_train=y_train,
                                                                     res_dict=res_dict))
    jobs.append(p)
    p.start()
    p.join()

for k, ret_data in res_dict.items():
    saveExperiment(UTC_global, ret_data)


'''# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Evaluate model with test data set and share sample prediction results
evaluation = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Model Accuracy on slices = %.2f' % (evaluation[1]))'''

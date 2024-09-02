import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import sys

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import Adam ,RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import datetime

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

train_data = pd.read_csv('/media/slam/2BAA4C7433C20D90/data/MNIST/train.csv')
test_data = pd.read_csv('/media/slam/2BAA4C7433C20D90/data/MNIST/test.csv')

train_data.head()

# image = np.array(train_data.drop('label',axis=1).iloc[0])
# label = train_data.iloc[0]['label']
# plt.imshow(image.reshape(28,28,1), cmap='gray')
# plt.axis('off')
# plt.show()

def visualise_random_image():
    index = np.random.randint(0,42000)
    image = np.array(train_data.drop('label',axis=1).iloc[index])
    label = train_data.iloc[index]['label']
    plt.imshow(image.reshape(28,28,1), cmap='gray')
    plt.title(label)  
    plt.axis('off')

# plt.figure(figsize=(12, 8))
# for i in range(50):
#     ax = plt.subplot(5, 10, i + 1)
#     visualise_random_image()

train_data['label'].value_counts()
# plt.figure(figsize=(8,6))
# sns.countplot(x='label', data=train_data)
# plt.show()

print((train_data['label'].value_counts()/len(train_data))*100)

X = train_data.drop('label', axis=1)
y = train_data['label']

(x_train1, y_train1), (x_test1, y_test1) = tf.keras.datasets.mnist.load_data()

train1 = np.concatenate([x_train1, x_test1], axis=0)
y_train1 = np.concatenate([y_train1, y_test1], axis=0)

Y_train1 = y_train1
X_train1 = train1.reshape(-1, 28*28)

X_train = np.concatenate((X.values, X_train1))
y_train = np.concatenate((y, y_train1))

X_train = X_train.reshape(-1,28,28,1)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1, random_state=101)

X_train = X_train/255.0
X_test = X_test/255.0


test_data = test_data/255
test_data = test_data.values.reshape(-1,28,28,1)

def model1():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer = 'adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    return model

def model2():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer = 'adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model

def model3():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer = 'adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model

def model4():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer = 'adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model

def model5():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer = 'adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model

def model6():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer = 'adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model

def model7():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer = 'adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model


# func = [model1(), model2(), model3(), model4(), model5(), model6()]
func = [model6(), model7()]

epochs = 50
cnt = 6

for model in func:
    log_dir = f'logs/fit/model{cnt}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    checkpoint_path = f'model_checkpoint/model{cnt}/cp.ckpt'

    print(f'log dir: {log_dir}')
    print(f'checkpoint: {checkpoint_path}')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    save_best_only=True,
                                                    verbose=1)

    # Fit the model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=2,
                        callbacks=[tensorboard_callback, cp_callback])

    print(f'model{cnt} summary')
    print(f'---------------------------------------')
    model.summary()
    model.save(f'saved_model/model{cnt}.h5')


    print(f'load model and testing')
    model = tf.keras.models.load_model(f'saved_model/model{cnt}.h5')
    model.load_weights(checkpoint_path)
    # model.summary()
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


    # PREDICTION:
    # predict results
    results = model.predict(test_data)

    # select the index with the maximum probability
    results = np.argmax(results,axis = 1)

    results = pd.Series(results,name="Label")
    print(results.shape)

    # SUBMISSION:
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
    #submission
    submission.to_csv(f'submission_model{cnt}_e{epochs}.csv',index=False)
    print("Successfully Completed!")

    cnt += 1

- submission_0.99203.csv

```def CNNModel():
    model = Sequential()

    # convolutional layer
    # model.add(Conv2D(filters=25, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(28, 28, 1)))
    # model.add(Conv2D(filters=25, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(filters=25, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=25, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # convolutional layer
    model.add(Conv2D(filters=50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(filters=50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(1,1)))
    model.add(Dropout(0.25))

    # flatten output of conv
    model.add(Flatten())

    # hidden layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    
    # output layer
    model.add(Dense(10, activation='softmax'))

    return model```

model5 

590/590 [==============================] - 9s 12ms/step - loss: 0.6592 - accuracy: 0.7887 - val_loss: 0.0805 - val_accuracy: 0.9743
Epoch 2/30
590/590 [==============================] - 6s 11ms/step - loss: 0.1183 - accuracy: 0.9636 - val_loss: 0.0719 - val_accuracy: 0.9764
Epoch 3/30
590/590 [==============================] - 7s 11ms/step - loss: 0.0827 - accuracy: 0.9735 - val_loss: 0.0389 - val_accuracy: 0.9876
Epoch 4/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0643 - accuracy: 0.9798 - val_loss: 0.0375 - val_accuracy: 0.9864
Epoch 5/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0577 - accuracy: 0.9819 - val_loss: 0.0313 - val_accuracy: 0.9895
Epoch 6/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0468 - accuracy: 0.9855 - val_loss: 0.0389 - val_accuracy: 0.9893
Epoch 7/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0449 - accuracy: 0.9859 - val_loss: 0.0481 - val_accuracy: 0.9864
Epoch 8/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0407 - accuracy: 0.9865 - val_loss: 0.0385 - val_accuracy: 0.9871

Epoch 00008: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
Epoch 9/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0350 - accuracy: 0.9890 - val_loss: 0.0286 - val_accuracy: 0.9917
Epoch 10/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0250 - accuracy: 0.9923 - val_loss: 0.0265 - val_accuracy: 0.9921
Epoch 11/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0287 - accuracy: 0.9912 - val_loss: 0.0289 - val_accuracy: 0.9912
Epoch 12/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0276 - accuracy: 0.9908 - val_loss: 0.0299 - val_accuracy: 0.9914
Epoch 13/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0236 - accuracy: 0.9930 - val_loss: 0.0299 - val_accuracy: 0.9900

Epoch 00013: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
Epoch 14/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0194 - accuracy: 0.9937 - val_loss: 0.0283 - val_accuracy: 0.9919
Epoch 15/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0176 - accuracy: 0.9942 - val_loss: 0.0233 - val_accuracy: 0.9917
Epoch 16/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0189 - accuracy: 0.9940 - val_loss: 0.0303 - val_accuracy: 0.9902

Epoch 00016: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
Epoch 17/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0167 - accuracy: 0.9947 - val_loss: 0.0281 - val_accuracy: 0.9905
Epoch 18/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0145 - accuracy: 0.9959 - val_loss: 0.0258 - val_accuracy: 0.9917
Epoch 19/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0130 - accuracy: 0.9964 - val_loss: 0.0264 - val_accuracy: 0.9905

Epoch 00019: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
Epoch 20/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0125 - accuracy: 0.9963 - val_loss: 0.0271 - val_accuracy: 0.9902
Epoch 21/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0130 - accuracy: 0.9962 - val_loss: 0.0245 - val_accuracy: 0.9917
Epoch 22/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0110 - accuracy: 0.9965 - val_loss: 0.0264 - val_accuracy: 0.9902

Epoch 00022: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.
Epoch 23/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0104 - accuracy: 0.9968 - val_loss: 0.0256 - val_accuracy: 0.9912
Epoch 24/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0139 - accuracy: 0.9960 - val_loss: 0.0244 - val_accuracy: 0.9912
Epoch 25/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0118 - accuracy: 0.9963 - val_loss: 0.0256 - val_accuracy: 0.9914

Epoch 00025: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05.
Epoch 26/30
590/590 [==============================] - 7s 11ms/step - loss: 0.0111 - accuracy: 0.9963 - val_loss: 0.0250 - val_accuracy: 0.9914
Epoch 27/30
590/590 [==============================] - 7s 11ms/step - loss: 0.0129 - accuracy: 0.9962 - val_loss: 0.0254 - val_accuracy: 0.9912
Epoch 28/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0109 - accuracy: 0.9965 - val_loss: 0.0249 - val_accuracy: 0.9912

Epoch 00028: ReduceLROnPlateau reducing learning rate to 1e-05.
Epoch 29/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0116 - accuracy: 0.9972 - val_loss: 0.0247 - val_accuracy: 0.9912
Epoch 30/30
590/590 [==============================] - 6s 11ms/step - loss: 0.0102 - accuracy: 0.9968 - val_loss: 0.0249 - val_accuracy: 0.9917




submission_model6_e15_0.99853.csv val accuracy: 0.9975
submission_model7_e10_0.99825.csv val accuracy: 0.9971


Epoch 1/30
3150/3150 - 18s - loss: 0.1063 - accuracy: 0.9679 - val_loss: 0.0372 - val_accuracy: 0.9896

Epoch 00001: saving model to model_checkpoint/model6/cp.ckpt
Epoch 2/30
3150/3150 - 15s - loss: 0.0494 - accuracy: 0.9859 - val_loss: 0.0314 - val_accuracy: 0.9899

Epoch 00002: saving model to model_checkpoint/model6/cp.ckpt
Epoch 3/30
3150/3150 - 13s - loss: 0.0388 - accuracy: 0.9891 - val_loss: 0.0219 - val_accuracy: 0.9937

Epoch 00003: saving model to model_checkpoint/model6/cp.ckpt
Epoch 4/30
3150/3150 - 12s - loss: 0.0311 - accuracy: 0.9911 - val_loss: 0.0164 - val_accuracy: 0.9949

Epoch 00004: saving model to model_checkpoint/model6/cp.ckpt
Epoch 5/30
3150/3150 - 13s - loss: 0.0252 - accuracy: 0.9925 - val_loss: 0.0217 - val_accuracy: 0.9939

Epoch 00005: saving model to model_checkpoint/model6/cp.ckpt
Epoch 6/30
3150/3150 - 14s - loss: 0.0221 - accuracy: 0.9935 - val_loss: 0.0190 - val_accuracy: 0.9946

Epoch 00006: saving model to model_checkpoint/model6/cp.ckpt
Epoch 7/30
3150/3150 - 14s - loss: 0.0189 - accuracy: 0.9943 - val_loss: 0.0156 - val_accuracy: 0.9953

Epoch 00007: saving model to model_checkpoint/model6/cp.ckpt
Epoch 8/30
3150/3150 - 13s - loss: 0.0167 - accuracy: 0.9950 - val_loss: 0.0176 - val_accuracy: 0.9954

Epoch 00008: saving model to model_checkpoint/model6/cp.ckpt
Epoch 9/30
3150/3150 - 12s - loss: 0.0141 - accuracy: 0.9958 - val_loss: 0.0184 - val_accuracy: 0.9952

Epoch 00009: saving model to model_checkpoint/model6/cp.ckpt
Epoch 10/30
3150/3150 - 13s - loss: 0.0135 - accuracy: 0.9961 - val_loss: 0.0129 - val_accuracy: 0.9962

Epoch 00010: saving model to model_checkpoint/model6/cp.ckpt
Epoch 11/30
3150/3150 - 13s - loss: 0.0113 - accuracy: 0.9966 - val_loss: 0.0177 - val_accuracy: 0.9954

Epoch 00011: saving model to model_checkpoint/model6/cp.ckpt
Epoch 12/30
3150/3150 - 11s - loss: 0.0117 - accuracy: 0.9965 - val_loss: 0.0183 - val_accuracy: 0.9955

Epoch 00012: saving model to model_checkpoint/model6/cp.ckpt
Epoch 13/30
3150/3150 - 14s - loss: 0.0094 - accuracy: 0.9971 - val_loss: 0.0163 - val_accuracy: 0.9961

Epoch 00013: saving model to model_checkpoint/model6/cp.ckpt
Epoch 14/30
3150/3150 - 14s - loss: 0.0093 - accuracy: 0.9972 - val_loss: 0.0102 - val_accuracy: 0.9971

Epoch 00014: saving model to model_checkpoint/model6/cp.ckpt
Epoch 15/30
3150/3150 - 12s - loss: 0.0088 - accuracy: 0.9973 - val_loss: 0.0114 - val_accuracy: 0.9964

Epoch 00015: saving model to model_checkpoint/model6/cp.ckpt
Epoch 16/30
3150/3150 - 13s - loss: 0.0075 - accuracy: 0.9977 - val_loss: 0.0102 - val_accuracy: 0.9972

Epoch 00016: saving model to model_checkpoint/model6/cp.ckpt
Epoch 17/30
3150/3150 - 13s - loss: 0.0079 - accuracy: 0.9976 - val_loss: 0.0127 - val_accuracy: 0.9968

Epoch 00017: saving model to model_checkpoint/model6/cp.ckpt
Epoch 18/30
3150/3150 - 13s - loss: 0.0064 - accuracy: 0.9980 - val_loss: 0.0118 - val_accuracy: 0.9968

Epoch 00018: saving model to model_checkpoint/model6/cp.ckpt
Epoch 19/30
3150/3150 - 13s - loss: 0.0067 - accuracy: 0.9979 - val_loss: 0.0095 - val_accuracy: 0.9974

Epoch 00019: saving model to model_checkpoint/model6/cp.ckpt
Epoch 20/30
3150/3150 - 13s - loss: 0.0065 - accuracy: 0.9981 - val_loss: 0.0125 - val_accuracy: 0.9965

Epoch 00020: saving model to model_checkpoint/model6/cp.ckpt
Epoch 21/30
3150/3150 - 14s - loss: 0.0066 - accuracy: 0.9979 - val_loss: 0.0076 - val_accuracy: 0.9975

Epoch 00021: saving model to model_checkpoint/model6/cp.ckpt
Epoch 22/30
3150/3150 - 15s - loss: 0.0059 - accuracy: 0.9982 - val_loss: 0.0112 - val_accuracy: 0.9976

Epoch 00022: saving model to model_checkpoint/model6/cp.ckpt
Epoch 23/30
3150/3150 - 13s - loss: 0.0055 - accuracy: 0.9984 - val_loss: 0.0092 - val_accuracy: 0.9979

Epoch 00023: saving model to model_checkpoint/model6/cp.ckpt
Epoch 24/30
3150/3150 - 12s - loss: 0.0056 - accuracy: 0.9981 - val_loss: 0.0093 - val_accuracy: 0.9981

Epoch 00024: saving model to model_checkpoint/model6/cp.ckpt
Epoch 25/30
3150/3150 - 13s - loss: 0.0053 - accuracy: 0.9984 - val_loss: 0.0162 - val_accuracy: 0.9972

Epoch 00025: saving model to model_checkpoint/model6/cp.ckpt
Epoch 26/30
3150/3150 - 14s - loss: 0.0049 - accuracy: 0.9985 - val_loss: 0.0125 - val_accuracy: 0.9975

Epoch 00026: saving model to model_checkpoint/model6/cp.ckpt
Epoch 27/30
3150/3150 - 13s - loss: 0.0056 - accuracy: 0.9985 - val_loss: 0.0109 - val_accuracy: 0.9978

Epoch 00027: saving model to model_checkpoint/model6/cp.ckpt
Epoch 28/30
3150/3150 - 13s - loss: 0.0058 - accuracy: 0.9984 - val_loss: 0.0123 - val_accuracy: 0.9980

Epoch 00028: saving model to model_checkpoint/model6/cp.ckpt
Epoch 29/30
3150/3150 - 13s - loss: 0.0045 - accuracy: 0.9985 - val_loss: 0.0103 - val_accuracy: 0.9985

Epoch 00029: saving model to model_checkpoint/model6/cp.ckpt
Epoch 30/30
3150/3150 - 11s - loss: 0.0049 - accuracy: 0.9985 - val_loss: 0.0084 - val_accuracy: 0.9982

Epoch 00030: saving model to model_checkpoint/model6/cp.ckpt
model6 summary
---------------------------------------
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 64)        1664      
_________________________________________________________________
batch_normalization (BatchNo (None, 28, 28, 64)        256       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 64)        102464    
_________________________________________________________________
batch_normalization_1 (Batch (None, 24, 24, 64)        256       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 12, 12, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 10, 64)        36928     
_________________________________________________________________
batch_normalization_2 (Batch (None, 10, 10, 64)        256       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 64)          36928     
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 8, 64)          256       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 4, 4, 64)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 4, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               262400    
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                2570      
=================================================================
Total params: 443,978
Trainable params: 443,466
Non-trainable params: 512
_________________________________________________________________

Epoch 1/30
3150/3150 - 15s - loss: 0.1173 - accuracy: 0.9666 - val_loss: 0.0415 - val_accuracy: 0.9867

Epoch 00001: saving model to model_checkpoint/model7/cp.ckpt
Epoch 2/30
3150/3150 - 14s - loss: 0.0545 - accuracy: 0.9848 - val_loss: 0.0223 - val_accuracy: 0.9931

Epoch 00002: saving model to model_checkpoint/model7/cp.ckpt
Epoch 3/30
3150/3150 - 14s - loss: 0.0416 - accuracy: 0.9882 - val_loss: 0.0254 - val_accuracy: 0.9921

Epoch 00003: saving model to model_checkpoint/model7/cp.ckpt
Epoch 4/30
3150/3150 - 14s - loss: 0.0319 - accuracy: 0.9908 - val_loss: 0.0188 - val_accuracy: 0.9945

Epoch 00004: saving model to model_checkpoint/model7/cp.ckpt
Epoch 5/30
3150/3150 - 15s - loss: 0.0265 - accuracy: 0.9925 - val_loss: 0.0244 - val_accuracy: 0.9926

Epoch 00005: saving model to model_checkpoint/model7/cp.ckpt
Epoch 6/30
3150/3150 - 14s - loss: 0.0226 - accuracy: 0.9936 - val_loss: 0.0183 - val_accuracy: 0.9946

Epoch 00006: saving model to model_checkpoint/model7/cp.ckpt
Epoch 7/30
3150/3150 - 14s - loss: 0.0187 - accuracy: 0.9944 - val_loss: 0.0138 - val_accuracy: 0.9956

Epoch 00007: saving model to model_checkpoint/model7/cp.ckpt
Epoch 8/30
3150/3150 - 15s - loss: 0.0153 - accuracy: 0.9955 - val_loss: 0.0111 - val_accuracy: 0.9964

Epoch 00008: saving model to model_checkpoint/model7/cp.ckpt
Epoch 9/30
3150/3150 - 13s - loss: 0.0142 - accuracy: 0.9957 - val_loss: 0.0116 - val_accuracy: 0.9968

Epoch 00009: saving model to model_checkpoint/model7/cp.ckpt
Epoch 10/30
3150/3150 - 14s - loss: 0.0112 - accuracy: 0.9966 - val_loss: 0.0100 - val_accuracy: 0.9971

Epoch 00010: saving model to model_checkpoint/model7/cp.ckpt
Epoch 11/30
3150/3150 - 14s - loss: 0.0124 - accuracy: 0.9963 - val_loss: 0.0145 - val_accuracy: 0.9957

Epoch 00011: saving model to model_checkpoint/model7/cp.ckpt
Epoch 12/30
3150/3150 - 14s - loss: 0.0097 - accuracy: 0.9970 - val_loss: 0.0099 - val_accuracy: 0.9969

Epoch 00012: saving model to model_checkpoint/model7/cp.ckpt
Epoch 13/30
3150/3150 - 15s - loss: 0.0091 - accuracy: 0.9974 - val_loss: 0.0112 - val_accuracy: 0.9967

Epoch 00013: saving model to model_checkpoint/model7/cp.ckpt
Epoch 14/30
3150/3150 - 14s - loss: 0.0086 - accuracy: 0.9976 - val_loss: 0.0086 - val_accuracy: 0.9971

Epoch 00014: saving model to model_checkpoint/model7/cp.ckpt
Epoch 15/30
3150/3150 - 14s - loss: 0.0074 - accuracy: 0.9976 - val_loss: 0.0114 - val_accuracy: 0.9971

Epoch 00015: saving model to model_checkpoint/model7/cp.ckpt
Epoch 16/30
3150/3150 - 15s - loss: 0.0081 - accuracy: 0.9975 - val_loss: 0.0094 - val_accuracy: 0.9974

Epoch 00016: saving model to model_checkpoint/model7/cp.ckpt
Epoch 17/30
3150/3150 - 13s - loss: 0.0076 - accuracy: 0.9978 - val_loss: 0.0083 - val_accuracy: 0.9976

Epoch 00017: saving model to model_checkpoint/model7/cp.ckpt
Epoch 18/30
3150/3150 - 14s - loss: 0.0066 - accuracy: 0.9980 - val_loss: 0.0087 - val_accuracy: 0.9976

Epoch 00018: saving model to model_checkpoint/model7/cp.ckpt
Epoch 19/30
3150/3150 - 14s - loss: 0.0068 - accuracy: 0.9978 - val_loss: 0.0092 - val_accuracy: 0.9973

Epoch 00019: saving model to model_checkpoint/model7/cp.ckpt
Epoch 20/30
3150/3150 - 15s - loss: 0.0065 - accuracy: 0.9981 - val_loss: 0.0117 - val_accuracy: 0.9973

Epoch 00020: saving model to model_checkpoint/model7/cp.ckpt
Epoch 21/30
3150/3150 - 14s - loss: 0.0065 - accuracy: 0.9980 - val_loss: 0.0067 - val_accuracy: 0.9979

Epoch 00021: saving model to model_checkpoint/model7/cp.ckpt
Epoch 22/30
3150/3150 - 14s - loss: 0.0053 - accuracy: 0.9984 - val_loss: 0.0093 - val_accuracy: 0.9974

Epoch 00022: saving model to model_checkpoint/model7/cp.ckpt
Epoch 23/30
3150/3150 - 14s - loss: 0.0055 - accuracy: 0.9985 - val_loss: 0.0077 - val_accuracy: 0.9982

Epoch 00023: saving model to model_checkpoint/model7/cp.ckpt
Epoch 24/30
3150/3150 - 14s - loss: 0.0057 - accuracy: 0.9983 - val_loss: 0.0105 - val_accuracy: 0.9979

Epoch 00024: saving model to model_checkpoint/model7/cp.ckpt
Epoch 25/30
3150/3150 - 13s - loss: 0.0055 - accuracy: 0.9983 - val_loss: 0.0106 - val_accuracy: 0.9969

Epoch 00025: saving model to model_checkpoint/model7/cp.ckpt
Epoch 26/30
3150/3150 - 14s - loss: 0.0056 - accuracy: 0.9983 - val_loss: 0.0123 - val_accuracy: 0.9975

Epoch 00026: saving model to model_checkpoint/model7/cp.ckpt
Epoch 27/30
3150/3150 - 15s - loss: 0.0048 - accuracy: 0.9986 - val_loss: 0.0119 - val_accuracy: 0.9975

Epoch 00027: saving model to model_checkpoint/model7/cp.ckpt
Epoch 28/30
3150/3150 - 15s - loss: 0.0059 - accuracy: 0.9984 - val_loss: 0.0110 - val_accuracy: 0.9983

Epoch 00028: saving model to model_checkpoint/model7/cp.ckpt
Epoch 29/30
3150/3150 - 15s - loss: 0.0050 - accuracy: 0.9986 - val_loss: 0.0083 - val_accuracy: 0.9981

Epoch 00029: saving model to model_checkpoint/model7/cp.ckpt
Epoch 30/30
3150/3150 - 15s - loss: 0.0047 - accuracy: 0.9987 - val_loss: 0.0104 - val_accuracy: 0.9978

Epoch 00030: saving model to model_checkpoint/model7/cp.ckpt
model7 summary
---------------------------------------
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 28, 28, 64)        1664      
_________________________________________________________________
batch_normalization_4 (Batch (None, 28, 28, 64)        256       
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 28, 28, 64)        102464    
_________________________________________________________________
batch_normalization_5 (Batch (None, 28, 28, 64)        256       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 14, 14, 64)        36928     
_________________________________________________________________
batch_normalization_6 (Batch (None, 14, 14, 64)        256       
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 12, 12, 64)        36928     
_________________________________________________________________
batch_normalization_7 (Batch (None, 12, 12, 64)        256       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 6, 6, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2304)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 256)               590080    
_________________________________________________________________
dropout_5 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                2570      
=================================================================
Total params: 771,658
Trainable params: 771,146
Non-trainable params: 512
_________________________________________________________________

submission_model6_e25_acc_0.9982_0.99935.csv val accuracy 0.9982
Epoch 25/25
3150/3150 - 14s - loss: 0.0059 - accuracy: 0.9983 - val_loss: 0.0071 - val_accuracy: 0.9982

Epoch 00025: saving model to model_checkpoint/model6/cp.ckpt
model6 summary
---------------------------------------
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 64)        1664      
_________________________________________________________________
batch_normalization (BatchNo (None, 28, 28, 64)        256       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 64)        102464    
_________________________________________________________________
batch_normalization_1 (Batch (None, 24, 24, 64)        256       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 12, 12, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 10, 64)        36928     
_________________________________________________________________
batch_normalization_2 (Batch (None, 10, 10, 64)        256       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 64)          36928     
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 8, 64)          256       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 4, 4, 64)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 4, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               262400    
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                2570      
=================================================================
Total params: 443,978
Trainable params: 443,466
Non-trainable params: 512
_________________________________________________________________

submission_model6_e50.csv
0.99967
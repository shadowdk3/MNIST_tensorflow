model6 summary
---------------------------------------
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
conv2d (Conv2D)              (None, 28, 28, 64)        1664      
batch_normalization (BatchNo (None, 28, 28, 64)        256       
conv2d_1 (Conv2D)            (None, 24, 24, 64)        102464    
batch_normalization_1 (Batch (None, 24, 24, 64)        256       
max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0         
dropout (Dropout)            (None, 12, 12, 64)        0         
conv2d_2 (Conv2D)            (None, 10, 10, 64)        36928     
batch_normalization_2 (Batch (None, 10, 10, 64)        256       
conv2d_3 (Conv2D)            (None, 8, 8, 64)          36928     
batch_normalization_3 (Batch (None, 8, 8, 64)          256       
max_pooling2d_1 (MaxPooling2 (None, 4, 4, 64)          0         
dropout_1 (Dropout)          (None, 4, 4, 64)          0         
flatten (Flatten)            (None, 1024)              0         
dense (Dense)                (None, 256)               262400    
dropout_2 (Dropout)          (None, 256)               0         
dense_1 (Dense)              (None, 10)                2570      

Total params: 443,978
Trainable params: 443,466
Non-trainable params: 512
_________________________________________________________________

submission_model6_e50.csv
0.99967
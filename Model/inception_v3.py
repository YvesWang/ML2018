#!/usr/bin/env python

import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
import pandas as pd

###### LOAD ######
y_train = []
y_test = []
with open("y_train150.txt", "r") as f:
     for line in f:
            y_train.append(int(line.strip()))
            
with open("y_test150.txt", "r") as f:
     for line in f:
            y_test.append(int(line.strip()))

X_train = np.load("X_train150.npy")
X_test = np.load("X_test150.npy")

###### Prepare for training ######
def batch_generator(X, y, batch_size):
    while True:
        batch_x = [None] * batch_size
        batch_y = [None] * batch_size
        for i in range(len(X)):
            batch_x[i%batch_size] = X[i]
            batch_y[i%batch_size] = y[i]
            if i%batch_size == batch_size-1:
                batch_x = np.stack(batch_x, axis=0)
                batch_y = keras.utils.to_categorical(batch_y, 10)
                yield batch_x, batch_y
                batch_x = [None] * batch_size
                batch_y = [None] * batch_size
        if batch_x[0].any():
            last_batch_x = [item for item in batch_x if type(item) == type(batch_x[0])]
            last_batch_y = [label for label in batch_y if type(label) == type(batch_y[0])]
            last_batch_x = np.stack(last_batch_x, axis=0)
            last_batch_y = keras.utils.to_categorical(last_batch_y, 10)
            yield last_batch_x, last_batch_y

class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, keras.utils.to_categorical(y, 10), verbose=0)
        print '\nTesting loss: {}, acc: {}\n'.format(loss, acc)
        
def GAP_vector(pred, conf, true, return_x=False):
    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap
    
###### Model ######
K.clear_session()

def inception(use_imagenet=True):
    model = keras.applications.InceptionV3(include_top=False, input_shape=(150, 150, 3),
                                          weights='imagenet' if use_imagenet else None)
    new_output = keras.layers.GlobalAveragePooling2D()(model.output)
    new_output = keras.layers.Dense(10, activation='softmax')(new_output)
    model = keras.models.Model(model.inputs, new_output)
    return model

model = inception()

###### Training ######
for layer in model.layers:
    layer.trainable = True
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.momentum = 0.8
    
for layer in model.layers[:-50]:
    layer.trainable = False
    
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(lr=1e-2),
    metrics=['accuracy'])

BATCH_SIZE = 32

model.fit_generator(
    batch_generator(X_train, y_train, BATCH_SIZE), 
    steps_per_epoch=len(X_train) // BATCH_SIZE, 
    epochs=300,
    validation_data=batch_generator(X_test, y_test, BATCH_SIZE), 
    validation_steps=len(X_test) // BATCH_SIZE // 2,
    callbacks=[TestCallback((X_test, Y_test))],
    verbose=2
    )


preds = model.predict(X_test)
test_preds = np.argmax(preds, axis=1)
conf = test_preds
gap = GAP_vector(test_preds, conf, y_test)
print '\nGAP: {}\n'.format(gap)

###### Save Model ######
model.save('inception_v3.h5')
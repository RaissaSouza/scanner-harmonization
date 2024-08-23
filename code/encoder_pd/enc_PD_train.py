from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import random
random.seed(1)
import tensorflow as tf
import csv
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from numpy import argmax
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D, LeakyReLU, Add
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
from datagenerator_pd import DataGenerator
from tensorflow.keras.utils import to_categorical
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import argparse
import sys
import math
import time

EPOCHS = 30
LEARNING_RATE = 0.001

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-fn_test', type=str, help='testing set')
parser.add_argument('-batch_size', type=int, help='batch size')
args = parser.parse_args()


params = {'batch_size': args.batch_size,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }

patience = 10
performance = []

def sfcn(inputLayer):
    #block 1
    x=Conv3D(filters=32, kernel_size=(3, 3, 3),padding='same',name="conv1")(inputLayer[0])
    x=BatchNormalization(name="norm1")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool1")(x)
    x=ReLU()(x)
    #block 2
    x=Conv3D(filters=64, kernel_size=(3, 3, 3),padding='same',name="conv2")(x)
    x=BatchNormalization(name="norm2")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool2")(x)
    x=ReLU()(x)
    #block 3
    x=Conv3D(filters=128, kernel_size=(3, 3, 3),padding='same',name="conv3")(x)
    x=BatchNormalization(name="norm3")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool3")(x)
    x=ReLU()(x)
    #block 4
    x=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="conv4")(x)
    x=BatchNormalization(name="norm4")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool4")(x)
    x=ReLU()(x)
    #block 5
    x=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="conv5")(x)
    x=BatchNormalization(name="norm5")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool5")(x)
    x=ReLU()(x)
    #block 6
    x=Conv3D(filters=64, kernel_size=(1, 1, 1),padding='same',name="conv6")(x)
    x=BatchNormalization(name="norm6")(x)
    x=ReLU()(x)
    #block 7
    x=AveragePooling3D()(x)
    x=Dropout(.2)(x)
    x=Flatten(name="flat1")(x)

    return x


optimizer_encoder = Adam(learning_rate=0.001)
optimizer_PD = Adam(learning_rate=0.001)
train_loss_pd = tf.keras.losses.BinaryCrossentropy(from_logits=False)
val_loss_pd = tf.keras.losses.BinaryCrossentropy(from_logits=False)
train_acc_metric = tf.keras.metrics.BinaryAccuracy()
val_acc_metric = tf.keras.metrics.BinaryAccuracy()


# encoder
inputA = Input(shape=(params['imagex'], params['imagey'], params['imagez'], 1), name="InputA")
feature_dense_enc = sfcn([inputA])

encoder = Model(inputs=[inputA], outputs=[feature_dense_enc])
encoder.compile(optimizer=optimizer_encoder)

def add_PD_layer(input_shape):
    inputs_PD = Input(shape=(input_shape))
    output_PD = Dense(1, activation='sigmoid')(inputs_PD)
    return Model(inputs_PD, output_PD)


classifier_PD = add_PD_layer(encoder.output.shape[1])
classifier_PD.compile(optimizer=optimizer_PD)


def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1)

 
# Dataset generator for PD classification
fn_train_PD = args.fn_train
train_PD = pd.read_csv(fn_train_PD)
train_IDs_list_PD = train_PD['Subject'].to_numpy()
train_IDs_PD = train_IDs_list_PD

fn_val_PD = args.fn_test
val_PD = pd.read_csv(fn_val_PD)
val_IDs_list_PD = val_PD['Subject'].to_numpy()
val_IDs_PD = val_IDs_list_PD

val_losses=[]
max_acc=0
acc=[]

# train step for PD classifier
@tf.function
def train_step(X, y_PD):
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=True)
        logits_PD = classifier_PD(logits_enc, training=True)
        y_PD = tf.reshape(y_PD, [5, 1])
        train_loss_PD = train_loss_pd(y_PD, logits_PD)
        train_acc_metric.update_state(y_PD, logits_PD)
       

    # compute gradient 
    grads = tape.gradient(train_loss_PD, [encoder.trainable_weights, classifier_PD.trainable_weights])
    #tf.print(grads)

    # update weights
    optimizer_encoder.apply_gradients(zip(grads[0], encoder.trainable_weights))
    optimizer_PD.apply_gradients(zip(grads[1], classifier_PD.trainable_weights))

    return train_loss_PD, logits_PD


# test step for PD classifier
@tf.function
def test_step(X, y_PD):

    val_logits_enc = encoder(X, training=False) 
    val_logits_PD = classifier_PD(val_logits_enc, training=False)
    y_PD = tf.reshape(y_PD, [5, 1])

    # Compute the loss value 
    val_loss_PD = val_loss_pd(y_PD, val_logits_PD)
    val_acc_metric.update_state(y_PD, val_logits_PD)
    
    return val_loss_PD, val_logits_PD


####################################################################################################################
#training PD classifier
for epoch in range(EPOCHS):
    # training
    training_generator_PD = DataGenerator(train_IDs_PD, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_train_PD, 'Group_bin')
    t1 = time.time()

    for batch in range (training_generator_PD.__len__()):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_PD = training_generator_PD.__getitem__(step_batch)
        print(type(y_PD))
        train_loss_PD, logits_PD= train_step(X, y_PD)
        print('\nBatch '+str(batch+1)+'/'+str(training_generator_PD.__len__()))
        print("LOSS PD -->", train_loss_PD)
        for _ in range(params['batch_size']):
            print("LOGITS PD -->", logits_PD[_])
            print("ACTUAL PD -->", y_PD[_])
    
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    t2 = time.time()
    template = 'TRAINING disease - ETA: {} - epoch: {}\n'
    print(template.format(round((t2-t1)/60, 4), epoch+1))

    #validation
    val_generator_PD = DataGenerator(val_IDs_PD, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_val_PD, 'Group_bin')

    t3 = time.time()
    ep_val_loss=0
    acc_ep=0
    for batch in range (val_generator_PD.__len__()):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_PD = val_generator_PD.__getitem__(step_batch)
        val_loss_PD, val_logits_PD = test_step(X, y_PD)
        print('\nBatch '+str(batch+1)+'/'+str(val_generator_PD.__len__()))
        print("VAL LOSS PD -->", val_loss_PD)
        for _ in range(params['batch_size']):
            print("LOGITS PD -->", val_logits_PD[_])
            print("ACTUAL PD -->", y_PD[_])
        ep_val_loss+=val_loss_PD
    
    val_acc = val_acc_metric.result()
    print("Validation acc over epoch: %.4f" % (float(val_acc),))
    # print("EP_VAL_LOSS")
    # print(ep_val_loss)
    print("EP_ACC")
    print(np.around(float(val_acc),4))
    norm_ep_val_loss = math.ceil((ep_val_loss/val_generator_PD.__len__())*100)/100
    # print(norm_ep_val_loss)
    val_losses.append(norm_ep_val_loss)
    acc.append(np.around(float(val_acc),4))

    t4 = time.time()
    template = 'VALIDATION disease - ETA: {} - epoch: {}\n'
    print(template.format(round((t4-t3)/60, 4), epoch+1))


    # learning rate scheduling
    LEARNING_RATE = scheduler(epoch, LEARNING_RATE)
    optimizer_encoder = Adam(learning_rate=LEARNING_RATE)
    optimizer_PD = Adam(learning_rate=1*LEARNING_RATE)

    # Reset training metrics and loss at the end of each epoch
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()
    

    if(len(acc)==1): #first epoch
        performance.append(0)
        max_acc=acc[-1]
        encoder.save('best_encoder_BS'+str(args.batch_size)+'.h5')
        classifier_PD.save('best_PD_classifier_BS'+str(args.batch_size)+'.h5')
        print("Saving model epoch: "+str(epoch))
    else:
        if acc[-1] > max_acc: #emproved after last epoch
            performance.append(0)
            max_acc=acc[-1]
            encoder.save('best_encoder_BS'+str(args.batch_size)+'.h5')
            classifier_PD.save('best_PD_classifier_BS'+str(args.batch_size)+'.h5')
            print("Saving model epoch: "+str(epoch))
        else: 
            performance.append(1) #did not improve
        if len(performance) > patience:
            if(sum(performance[-10:])==10): # if the last 10 performances did not improve
                print('Early stopping. No improvement in validation loss in epoch: '+str(epoch))
                break

####################################################################################################################
print(val_losses)
print(acc)
print(performance)

###################################################################################################################
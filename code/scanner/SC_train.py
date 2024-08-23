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
from sklearn.metrics import confusion_matrix
from datagenerator_pd import DataGenerator
from tensorflow.keras.utils import to_categorical
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import argparse
import sys
import math
import time

EPOCHS = 30
LEARNING_RATE = 0.0002

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-en', type=str, help='encoder model args path')
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

encoder = tf.keras.models.load_model(args.en)
encoder.trainable = False

#optimizer_SC = Adam(lr=0.001, decay=0.003)
optimizer_SC = Adam(learning_rate=0.001)


def add_SC_layer(input_shape):
    inputs_PD = Input(shape=(input_shape))
    output_PD = Dense(23, activation='softmax')(inputs_PD)
    return Model(inputs_PD, output_PD)


classifier_SC = add_SC_layer(encoder.output.shape[1])
classifier_SC.compile(optimizer=optimizer_SC)

train_loss_sc = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
val_loss_sc = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()


def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1)


# Dataset generator for SC classification
fn_train_SC = args.fn_train
train_SC = pd.read_csv(fn_train_SC)
train_IDs_list_SC = train_SC['Subject'].to_numpy()
train_IDs_SC = train_IDs_list_SC

fn_val_SC = args.fn_test
val_SC = pd.read_csv(fn_val_SC)
val_IDs_list_SC = val_SC['Subject'].to_numpy()
val_IDs_SC = val_IDs_list_SC

val_losses=[]
max_acc=0
acc=[]

# train step for PD classifier
@tf.function
def train_step_SC( X, y_SC):
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=False)
        logits_SC = classifier_SC(logits_enc, training=True)
        train_loss_SC = train_loss_sc(tf.one_hot(y_SC, 23), logits_SC)
        train_acc_metric.update_state(tf.one_hot(y_SC, 23), logits_SC)

    # compute gradient 
    grads = tape.gradient(train_loss_SC, classifier_SC.trainable_weights)

    # update weights
    optimizer_SC.apply_gradients(zip(grads, classifier_SC.trainable_weights))


    return train_loss_SC, logits_SC


# test step for PD classifier
@tf.function
def test_step_SC(X, y_SC):

    val_logits_enc = encoder(X, training=False) 
    val_logits_SC = classifier_SC(val_logits_enc, training=False)
    val_loss_SC = train_loss_sc(tf.one_hot(y_SC, 23), val_logits_SC)
    val_acc_metric.update_state(tf.one_hot(y_SC, 23), val_logits_SC)

 
    
    return val_loss_SC, val_logits_SC


####################################################################################################################

# training PD classifier
for epoch in range(EPOCHS):
    # training
    training_generator_SC = DataGenerator(train_IDs_SC, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_train_SC, 'Scanner')
    t1 = time.time()

    for batch in range (training_generator_SC.__len__()):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_SC = training_generator_SC.__getitem__(step_batch)
        train_loss_SC, logits_SC= train_step_SC(X, y_SC)
        print('\nBatch '+str(batch+1)+'/'+str(training_generator_SC.__len__()))
        print("LOSS SC -->", train_loss_SC)
        for _ in range(params['batch_size']):
            print("LOGITS SC -->", logits_SC[_])
            print("ACTUAL SC -->", tf.one_hot(y_SC, 23))
    
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    t2 = time.time()
    template = 'TRAINING disease - ETA: {} - epoch: {}\n'
    print(template.format(round((t2-t1)/60, 4), epoch+1))

    # validation
    val_generator_SC = DataGenerator(val_IDs_SC, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_val_SC, 'Scanner')
    t3 = time.time()

    ep_val_loss=0
    acc_ep=0
    for batch in range (val_generator_SC.__len__()):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_SC = val_generator_SC.__getitem__(step_batch)
        val_loss_SC, val_logits_SC = test_step_SC(X, y_SC)
        print('\nBatch '+str(batch+1)+'/'+str(val_generator_SC.__len__()))
        print("VAL LOSS SC -->", val_loss_SC)
        for _ in range(params['batch_size']):
            print("LOGITS SC -->", val_logits_SC[_])
            print("ACTUAL SC -->", tf.one_hot(y_SC, 23))
        ep_val_loss+=val_loss_SC
    
    val_acc = val_acc_metric.result()
    print("Validation acc over epoch: %.4f" % (float(val_acc),))
    print("EP_ACC")
    print(np.around(float(val_acc),4))
    norm_ep_val_loss = math.ceil((ep_val_loss/val_generator_SC.__len__())*100)/100
    # print(norm_ep_val_loss)
    val_losses.append(norm_ep_val_loss)
    acc.append(np.around(float(val_acc),4))

    t4 = time.time()
    template = 'VALIDATION disease - ETA: {} - epoch: {}\n'
    print(template.format(round((t4-t3)/60, 4), epoch+1))

    # learning rate scheduling
    LEARNING_RATE = scheduler(epoch, LEARNING_RATE)
    optimizer_SC = Adam(learning_rate=LEARNING_RATE)

    # Reset training metrics and loss at the end of each epoch
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()
    

    if(len(acc)==1): #first epoch
        performance.append(0)
        max_acc=acc[-1]
        classifier_SC.save('best_SC_classifier_BS'+str(args.batch_size)+'.h5')
        print("Saving model epoch: "+str(epoch))
    else:
        if acc[-1] > max_acc: #emproved after last epoch
            performance.append(0)
            max_acc=acc[-1]
            classifier_SC.save('best_SC_classifier_BS'+str(args.batch_size)+'.h5')
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

####################################################################################################################
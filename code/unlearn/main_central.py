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
#from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from datagenerator import DataGenerator
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
parser.add_argument('-alpha', type=float, help='alpha')
parser.add_argument('-beta', type=float, help='beta')
parser.add_argument('-en', type=str, help='pretrained encoder')
parser.add_argument('-pd', type=str, help='pretrained PD classifier')
parser.add_argument('-sc', type=str, help='pretrained SC classifier')
args = parser.parse_args()


params = {'batch_size': args.batch_size,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }


# pretrained models
encoder = tf.keras.models.load_model(args.en)
classifier_PD = tf.keras.models.load_model(args.pd)
classifier_SC = tf.keras.models.load_model(args.sc)

encoder.trainable = True
classifier_PD.trainable = True
classifier_SC.trainable = True


# optimizers
optimizer_encoder = Adam(learning_rate=0.001)
optimizer_PD = Adam(learning_rate=0.001)
optimizer_SC = Adam(learning_rate=0.001)

encoder.compile(optimizer=optimizer_encoder)
classifier_PD.compile(optimizer=optimizer_PD)
classifier_SC.compile(optimizer=optimizer_SC)

#optimizer_SC = Adam(learning_rate=0.001, decay=0.003)

train_loss_sc = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
val_loss_sc = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
train_acc_sc = tf.keras.metrics.CategoricalAccuracy()
val_acc_sc = tf.keras.metrics.CategoricalAccuracy()

train_loss_pd = tf.keras.losses.BinaryCrossentropy(from_logits=False)
val_loss_pd = tf.keras.losses.BinaryCrossentropy(from_logits=False)
train_acc_pd = tf.keras.metrics.BinaryAccuracy()
val_acc_pd = tf.keras.metrics.BinaryAccuracy()


def confusionLoss(logits_SC, batch_size):
    log_logits = tf.math.log(logits_SC)
    sum_log_logits = tf.math.reduce_sum(log_logits)
    #norm = sum_log_logits/batch_size
    return -1*sum_log_logits / (batch_size * 23)
    #return -1*norm


# scheduler
def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1)

 
# Dataset generator for PD classification
fn_train = args.fn_train
train = pd.read_csv(fn_train)
train_IDs_list = train['Subject'].to_numpy()
train_IDs = train_IDs_list

fn_val = args.fn_test
val = pd.read_csv(fn_val)
val_IDs_list = val['Subject'].to_numpy()
val_IDs = val_IDs_list


# train step for PD classifier
@tf.function
def train_step( X, y_PD, y_SC):
    ###################################################
    # FIRST STEP MAIN TASK - PD
    classifier_SC.trainable = False
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=True)
        logits_PD = classifier_PD(logits_enc, training=True)
        train_loss_PD = train_loss_pd(y_PD, logits_PD)
        train_acc_pd.update_state(y_PD, logits_PD)

    # compute gradient 
    grads = tape.gradient(train_loss_PD, [encoder.trainable_weights, classifier_PD.trainable_weights])

    # update weights
    optimizer_encoder.apply_gradients(zip(grads[0], encoder.trainable_weights))
    optimizer_PD.apply_gradients(zip(grads[1], classifier_PD.trainable_weights))
    ###################################################
    # SECOND STEP DOMAIN TASK - SCANNER
    encoder.trainable = False
    classifier_PD.trainable = False
    classifier_SC.trainable = True
    with tf.GradientTape() as tape:
        #logits_enc = encoder(X, training=False)
        logits_SC = classifier_SC(logits_enc, training=True)
        train_loss_SC = args.alpha * train_loss_sc(tf.one_hot(y_SC, 23), logits_SC)
        train_acc_sc.update_state(tf.one_hot(y_SC, 23), logits_SC)

    # compute gradient 
    grads = tape.gradient(train_loss_SC, classifier_SC.trainable_weights)

    # update weights
    optimizer_SC.apply_gradients(zip(grads, classifier_SC.trainable_weights))
    encoder.trainable = True
    classifier_PD.trainable = True
    ###################################################
    # THIRD STEP
    classifier_PD.trainable = False
    classifier_SC.trainable = False
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=True)
        logits_SC = classifier_SC(logits_enc, training=False)
        confusion_loss = args.beta * confusionLoss(logits_SC, args.batch_size)

    # compute gradient 
    grads = tape.gradient(confusion_loss, encoder.trainable_weights)

    # update weights
    optimizer_encoder.apply_gradients(zip(grads, encoder.trainable_weights))
    classifier_PD.trainable = True
    classifier_SC.trainable = True
    ###################################################

    # logits_enc = encoder(X, training=True)
    # logits_PD = classifier_PD(logits_enc, training=True)
    # logits_SC = classifier_SC(logits_enc, training=True)

    # train_loss = train_acc_pd(y_PD, logits_PD) + args.alpha * train_loss_sc(tf.one_hot(y_SC, 23), logits_SC) \
    #             + args.beta * confusionLoss(logits_SC, args.batch_size)
    # tf.print("PD_LOSS")
    # tf.print(train_loss_PD)
    #tf.print("SC_LOSS")
    #tf.print(train_loss_SC)
    # tf.print("CONF_LOSS")
    # tf.print(confusion_loss)
    train_loss = train_loss_PD + args.alpha * train_loss_SC + args.beta * confusion_loss
    # tf.print("TOTAL_LOSS")
    # tf.print(train_loss)

    return train_loss, logits_PD, logits_SC


# test step for PD classifier
@tf.function
def test_step( X, y_PD, y_SC):

    val_logits_enc = encoder(X, training=False) 
    val_logits_PD = classifier_PD(val_logits_enc, training=False)
    val_logits_SC = classifier_SC(val_logits_enc, training=False)
    val_acc_pd.update_state(y_PD, val_logits_PD)
    val_acc_sc.update_state(tf.one_hot(y_SC, 23), val_logits_SC)

    val_loss_PD = val_loss_pd(y_PD, val_logits_PD)
    val_loss_SC = val_loss_sc(tf.one_hot(y_SC, 23), val_logits_SC)
    val_confusion_loss = confusionLoss(val_logits_SC, args.batch_size)

    # tf.print(val_loss_PD)
    # tf.print(val_loss_SC)
    # tf.print(val_confusion_loss)
    # Compute the loss value 
    val_loss =  val_loss_PD + args.alpha * val_loss_SC + args.beta * val_confusion_loss
    # tf.print(val_loss)
    
    return val_loss, val_logits_PD, val_logits_SC

val_losses=[]
max_acc_pd=0
max_acc_pd2=0
min_acc_sc=0
acc_pd=[]
acc_sc=[]
patience = 10
performance_pd = []
performance_sc = []
####################################################################################################################

# training PD classifier
for epoch in range(EPOCHS):
    # training
    training_generator = DataGenerator(train_IDs, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_train, 'Group_bin', 'Scanner')
    t1 = time.time()
    total_train_loss=0
    total_val_loss=0
    for batch in range(training_generator.__len__()):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_PD, y_SC = training_generator.__getitem__(step_batch)
        train_loss, logits_PD, logits_SC = train_step( X, y_PD, y_SC)
        print('\nBatch '+str(batch+1)+'/'+str(training_generator.__len__()))
        print("LOSS PD -->", train_loss)
        # for _ in range(args.batch_size):
        #     print("LOGITS PD -->", logits_PD[_])
        #     print("ACTUAL PD -->", y_PD[_])
        total_train_loss+=train_loss
    
    print("Training loss over epoch: %.4f" % (float(total_train_loss/training_generator.__len__()),))
    train_pd_acc = train_acc_pd.result()
    print("Training acc PD over epoch: %.4f" % (float(train_pd_acc),))

    train_sc_acc = train_acc_sc.result()
    print("Training acc SC over epoch: %.4f" % (float(train_sc_acc),))
 

    t2 = time.time()
    template = 'TRAINING - ETA: {} - epoch: {}\n'
    print(template.format(round((t2-t1)/60, 4), epoch+1))


    # validation
    val_generator = DataGenerator(val_IDs, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_val, 'Group_bin', 'Scanner')
    t3 = time.time()
    acc_ep_pd=0
    acc_ep_sc=0
    for batch in range(val_generator.__len__()):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_PD, y_SC = val_generator.__getitem__(step_batch)
        val_loss, val_logits_PD, val_logits_SC = test_step(X, y_PD, y_SC)
        print('\nBatch '+str(batch+1)+'/'+str(val_generator.__len__()))
        print("VAL LOSS PD -->", val_loss)
        # for _ in range(args.batch_size):
        #     print("LOGITS PD -->", val_logits_PD[_])
        #     print("ACTUAL PD -->", y_PD[_])
        total_val_loss+=val_loss
    
    print("Validation loss over epoch: %.4f" % (float(total_val_loss/val_generator.__len__()),))
    
    val_pd_acc = val_acc_pd.result()
    print("Validation acc PD over epoch: %.4f" % (float(val_pd_acc),))

    val_sc_acc = val_acc_sc.result()
    print("Validation acc SC over epoch: %.4f" % (float(val_sc_acc),))

    # print(norm_ep_val_loss)
    val_losses.append(np.around(float(total_val_loss/val_generator.__len__()),4))
    acc_pd.append(np.around(float(val_pd_acc),4))
    acc_sc.append(np.around(float(val_sc_acc),4))

    t4 = time.time()
    template = 'VALIDATION - ETA: {} - epoch: {}\n'
    print(template.format(round((t4-t3)/60, 4), epoch+1))

    # learning rate scheduling
    LEARNING_RATE = scheduler(epoch, LEARNING_RATE)
    optimizer_encoder = Adam(learning_rate=LEARNING_RATE)
    optimizer_PD = Adam(learning_rate=LEARNING_RATE)

    # Reset training metrics and loss at the end of each epoch
    train_acc_pd.reset_states()
    val_acc_pd.reset_states()

    train_acc_sc.reset_states()
    val_acc_sc.reset_states()


    if(len(acc_pd)==1): #first epoch
        performance_pd.append(0)
        performance_sc.append(0)
        max_acc_pd=acc_pd[-1]
        max_acc_pd2=acc_pd[-1]
        min_acc_sc=acc_sc[-1]
        encoder.save('best_un_encoder_BS'+str(args.batch_size)+'.h5')
        classifier_PD.save('best_un_PD_classifier_BS'+str(args.batch_size)+'.h5')
        classifier_SC.save('best_un_SC_classifier_BS'+str(args.batch_size)+'.h5')
        print("Saving model epoch: "+str(epoch))
    else:
        if acc_pd[-1] > max_acc_pd and acc_sc[-1]< min_acc_sc: #emproved after last epoch
            performance_pd.append(0)
            performance_sc.append(0)
            max_acc_pd=acc_pd[-1]
            min_acc_sc=acc_sc[-1]
            encoder.save('best_un_encoder_BS'+str(args.batch_size)+'.h5')
            classifier_PD.save('best_un_PD_classifier_BS'+str(args.batch_size)+'.h5')
            classifier_SC.save('best_un_SC_classifier_BS'+str(args.batch_size)+'.h5')
            print("Saving model epoch: "+str(epoch))
        elif acc_pd[-1] > max_acc_pd2 and acc_sc[-1] > min_acc_sc: #only pd improved
            performance_pd.append(1)
            performance_sc.append(1)
            max_acc_pd2=acc_pd[-1]
            encoder.save('high_un_encoder_BS'+str(args.batch_size)+'.h5')
            classifier_PD.save('high_un_PD_classifier_BS'+str(args.batch_size)+'.h5')
            classifier_SC.save('low_un_SC_classifier_BS'+str(args.batch_size)+'.h5')
            print("Saving model epoch: "+str(epoch))
        else: 
            performance_pd.append(1)
            performance_sc.append(1) #did not improve
        if len(performance_pd) > patience:
            if(sum(performance_pd[-10:])==10): # if the last 10 performances did not improve
                print('Early stopping. No improvement in validation loss in epoch: '+str(epoch))
                break

####################################################################################################################
print(val_losses)
print(acc_pd)
print(acc_sc)
print(performance_pd)
print(performance_sc)
# encoder.save('encoder_unlearned_BS'+str(args.batch_size)+'.h5')
# classifier_PD.save('PD_classifier_unlearned_BS'+str(args.batch_size)+'.h5')
# classifier_SC.save('SC_classifier_unlearned_BS'+str(args.batch_size)+'.h5')

####################################################################################################################
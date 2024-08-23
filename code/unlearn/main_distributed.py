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

LEARNING_RATE = 0.001

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-batch_size', type=int, help='batch size')
parser.add_argument('-alpha', type=float, help='alpha')
parser.add_argument('-beta', type=float, help='beta')
parser.add_argument('-cycles', type=int, help='no of cycles')
parser.add_argument('-epochs', type=int, help='no of epochs')
parser.add_argument('-en', type=str, help='pretrained encoder')
parser.add_argument('-pd', type=str, help='pretrained PD classifier')
parser.add_argument('-sc', type=str, help='pretrained SC classifier')
args = parser.parse_args()


params = {'batch_size': args.batch_size,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }

CYCLES = args.cycles
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

# pretrained models
encoder = tf.keras.models.load_model(args.en)
classifier_PD = tf.keras.models.load_model(args.pd)
classifier_SC = tf.keras.models.load_model(args.sc)

encoder.trainable = True
classifier_PD.trainable = True
classifier_SC.trainable = True


# optimizers
optimizer_encoder = Adam(learning_rate=0.0001)
optimizer_PD = Adam(learning_rate=0.0001)
optimizer_SC = Adam(learning_rate=0.0001)

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
studies = train['Study'].unique()
np.random.seed(42)  
np.random.shuffle(studies)



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

    train_loss = train_loss_PD + args.alpha * train_loss_SC + args.beta * confusion_loss


    return train_loss, logits_PD, logits_SC


####################################################################################################################

# training
for c in range(CYCLES):
    np.random.seed(42+c)  
    np.random.shuffle(studies)
    print("CYCLE --> "+str(c)+'\n')

    ########################
    for s in studies:
        batch_size = BATCH_SIZE
        print("STUDY --> "+str(s))

        train_aux =  train[train['Study']==s]
        IDs_list = train_aux['Subject'].to_numpy()
        train_IDs = IDs_list
        if(len(train_IDs)<batch_size): 
            batch_size=len(train_IDs)
        
        for epoch in range(EPOCHS):
            training_generator = DataGenerator(train_IDs, batch_size, (params['imagex'], params['imagey'], params['imagez']), True, fn_train, 'Group_bin', 'Scanner')
            t1 = time.time()

            for batch in range(training_generator.__len__()):
                step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
                X, y_PD, y_SC = training_generator.__getitem__(step_batch)
                train_loss, logits_PD, logits_SC = train_step( X, y_PD, y_SC)
                print('\nBatch '+str(batch+1)+'/'+str(training_generator.__len__()))
                print("LOSS PD -->", train_loss)
                # for _ in range(args.batch_size):
                #     print("LOGITS PD -->", logits_PD[_])
                #     print("ACTUAL PD -->", y_PD[_])
    

    train_pd_acc = train_acc_pd.result()
    print("Training acc PD over a cycle: %.4f" % (float(train_pd_acc),))

    train_sc_acc = train_acc_sc.result()
    print("Training acc SC over cycle: %.4f" % (float(train_sc_acc),))
 

    t2 = time.time()
    template = 'TRAINING - ETA: {} - epoch: {}\n'
    print(template.format(round((t2-t1)/60, 4), epoch+1))
    ########################

    # learning rate scheduling
    LEARNING_RATE = scheduler(epoch, LEARNING_RATE)
    optimizer_encoder = Adam(learning_rate=LEARNING_RATE)
    optimizer_PD = Adam(learning_rate=1*LEARNING_RATE)

    #model save
    encoder.save('encoder_distributed_unlearned_BS'+str(BATCH_SIZE)+"_"+str(c)+".h5")
    classifier_PD.save('classifier_PD_distributed_unlearned_BS'+str(BATCH_SIZE)+"_"+str(c)+".h5")
    classifier_SC.save('classifier_SC_distributed_unlearned_BS'+str(BATCH_SIZE)+"_"+str(c)+".h5")


####################################################################################################################


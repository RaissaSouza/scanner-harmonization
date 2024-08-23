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

LEARNING_RATE = 0.0001

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-en', type=str, help='encoder model args path')
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-cycles', type=int, help='number of cycles')
parser.add_argument('-epochs', type=int, help='number of local epochs per cycle')
parser.add_argument('-batch_size', type=int, help='batch size')
args = parser.parse_args()


params = {'batch_size': args.batch_size,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }
CYCLES = args.cycles
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

encoder = tf.keras.models.load_model(args.en)
encoder.trainable = False

optimizer_SC = Adam(learning_rate=0.0001)


def add_SC_layer(input_shape):
    inputs_PD = Input(shape=(input_shape))
    output_PD = Dense(23, activation='softmax')(inputs_PD)
    return Model(inputs_PD, output_PD)


classifier_SC = add_SC_layer(encoder.output.shape[1])
classifier_SC.compile(optimizer=optimizer_SC)

train_loss_sc = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()


def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1)


# Dataset generator for SC classification
fn_train_SC = args.fn_train
train_SC = pd.read_csv(fn_train_SC)
train_IDs_list_SC = train_SC['Subject'].to_numpy()
train_IDs_SC = train_IDs_list_SC
studies =  train_SC['Study'].unique()
np.random.seed(42)  
np.random.shuffle(studies)

# train step for PD classifier
@tf.function
def train_step(X, y_SC):
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


####################################################################################################################
for c in range(CYCLES):
    np.random.seed(42+c)  
    np.random.shuffle(studies)
    print("CYCLE --> "+str(c)+'\n')

    ########################
    for s in studies:
        batch_size = BATCH_SIZE
        print("STUDY --> "+str(s))

        train_aux =  train_SC[train_SC['Study']==s]
        IDs_list = train_aux['Subject'].to_numpy()
        train_IDs = IDs_list
        if(len(train_IDs)<batch_size): 
            batch_size=len(train_IDs)
            
        
        for epoch in range(EPOCHS):
            training_generator_SC = DataGenerator(train_IDs, batch_size, (params['imagex'], params['imagey'], params['imagez']), True, fn_train_SC, 'Scanner')
            t1 = time.time()


            for batch in range (training_generator_SC.__len__()):
                step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
                X, y_SC = training_generator_SC.__getitem__(step_batch)
        
                train_loss_SC, logits_SC= train_step(X, y_SC)
                print('\nBatch '+str(batch+1)+'/'+str(training_generator_SC.__len__()))
                print("LOSS PD -->", train_loss_SC)
                # for _ in range(tf.size(logits_SC)):
                #     print("LOGITS SC -->", logits_SC[_])
                #     print("ACTUAL SC -->", tf.one_hot(y_SC, 23))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over cycle: %.4f" % (float(train_acc),))

    t2 = time.time()
    template = 'TRAINING disease - ETA: {} - cycle: {}\n'
    print(template.format(round((t2-t1)/60, 4), c+1))


    ########################

    # learning rate scheduling
    LEARNING_RATE = scheduler(epoch, LEARNING_RATE)
    optimizer_SC = Adam(learning_rate=1*LEARNING_RATE)

    # Reset training metrics and loss at the end of each epoch
    train_acc_metric.reset_states()

    #model save
    classifier_SC.save('SC_classifier_distributed_BS'+str(args.batch_size)+'_'+str(c)+'.h5')

####################################################################################################################
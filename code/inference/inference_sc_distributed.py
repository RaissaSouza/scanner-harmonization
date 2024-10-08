from pickle import FALSE
from datagenerator_pd import DataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from numpy.random import seed
seed(1)
tf.random.set_seed = 1
random.seed(1)
import argparse
import os
import sys
os.environ['TF_DETERMINISTIC_OPS'] = '1'



#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn', type=str, help='filename to infer')
parser.add_argument('-en', type=str, help='model to infer')
parser.add_argument('-sc', type=str, help='model to infer')
parser.add_argument('-o', type=str, help='output name')
args = parser.parse_args()
import csv

def model_eval(y_test, y_pred_raw):
    y_pred = np.argmax(y_pred_raw, axis=1)
    y_pred = y_pred.astype(int)
    y_test = y_test.to_frame()
    y_test = y_test.rename(columns={'Scanner': 'ground_truth'})
    y_test['preds'] = y_pred
    # y_test['preds_raw'] = y_pred_raw
    return y_test



def compute_metrics(df,fn):
    y_test = df['ground_truth'].values
    y_pred = df['preds'].values

    
    ac=accuracy_score(y_test, y_pred)
    print("accuracy =",ac)
    return ac
    

params = {'batch_size': 5,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }


fn_test = args.fn
test = pd.read_csv(fn_test)
IDs_list=test['Subject'].to_numpy()
test_IDs=IDs_list
test_generator=DataGenerator(test_IDs, 1, (params['imagex'], params['imagey'], params['imagez']), False, fn_test, 'Scanner')

results = pd.DataFrame(columns=['cycle','accuracy'])
cycle_count=1
mypath_encoder = args.en
mypath_classifier = args.sc
name=args.o

for i in range(30):
    encoder = tf.keras.models.load_model(mypath_encoder + "encoder_distributed_BS5_27.h5")
    classifier = tf.keras.models.load_model(mypath_classifier+ "SC_classifier_distributed_BS5_"+str(i)+".h5")
    encoder.trainable = False
    classifier.trainable = False




#Make predictions and save the results
    y_test = test['Scanner']
    y_pred_encoder=encoder.predict(test_generator)
    y_pred = classifier.predict(y_pred_encoder)
    preds = model_eval(y_test, y_pred)

    df = pd.merge(preds, test, left_index=True, right_index=True)
    # df.to_csv(name+'_predictions.csv')


# print("######Overall Metrics######")
    metrics = compute_metrics(df,"cm_agg_"+name)
    print("METRICS -->", metrics)

    new_result = {'cycle':cycle_count, 'accuracy':metrics}

    results.loc[len(results)] = new_result

    cycle_count += 1


results.to_csv(name+'_metrics.csv')




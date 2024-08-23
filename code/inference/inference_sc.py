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
    #y_test['preds_raw'] = y_pred_raw
    return y_test



def compute_metrics(df,fn):
    y_test = df['ground_truth'].values
    y_pred = df['preds'].values
    cm = confusion_matrix(y_test, y_pred)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    #print(cm)
    cm_df = pd.DataFrame(cm,
        index = ['GE Discovery 750','GE Genesis Signa','GE Optima MR450','GE Signa Excite','GE Signa Hdxt', 'Philips Achieva', 'Philips Gyroscan NT','Philips Intera','Siemens Avanto','Siemens Biograph_mMR','Siemens Espree','Siemens Prisma','Siemens Prisma_fit','Siemens Skyra','Siemens Sonata','Siemens Symphony','Siemens Trio','Siemens Trio Tim','Siemens Verio', 'GE Signa UHD','GE Signa Premier', 'Philips Ingenia', 'Philips Achieva dStream'], 
        columns = ['GE Discovery 750','GE Genesis Signa','GE Optima MR450','GE Signa Excite','GE Signa Hdxt', 'Philips Achieva', 'Philips Gyroscan NT','Philips Intera','Siemens Avanto','Siemens Biograph_mMR','Siemens Espree','Siemens Prisma','Siemens Prisma_fit','Siemens Skyra','Siemens Sonata','Siemens Symphony','Siemens Trio','Siemens Trio Tim','Siemens Verio', 'GE Signa UHD','GE Signa Premier', 'Philips Ingenia', 'Philips Achieva dStream'])
                    
    
    #Plotting the confusion matrix
    plt.figure(figsize=(10,8))
    g1=sns.heatmap(cm_df, cmap="Blues", annot=False,fmt='.2f', vmin=0, vmax=1.0, center=0.5,cbar=True)
    #g1.set(xticklabels=[])
    plt.title('Scanners')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig(fn,format='png', dpi=1200,bbox_inches='tight')
    plt.show()
    
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


# reload the best performing model for the evaluation
encoder=tf.keras.models.load_model(args.en)
sc_classifier=tf.keras.models.load_model(args.sc)
# make sure that model weights are non-trainable
encoder.trainable=False
sc_classifier.trainable=False

name=args.o

#Make predictions and save the results
y_test = test['Scanner']
y_pred_encoder=encoder.predict(test_generator)
y_pred = sc_classifier.predict(y_pred_encoder)
preds = model_eval(y_test, y_pred)

df = pd.merge(preds, test, left_index=True, right_index=True)
df.to_csv(name+'_predictions.csv')

# print("######Overall Metrics######")
metrics = compute_metrics(df,"cm_agg_"+name+".png")
print("METRICS -->", metrics)

metrics_df = pd.DataFrame(['Acc'], columns=['metrics'])
metrics_df = metrics_df.set_index('metrics')
metrics_df['Aggregate'] = metrics
metrics_df.to_csv(name+'_metrics.csv')



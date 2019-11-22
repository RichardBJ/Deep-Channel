# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:53:49 2019

@author: ncelik34



"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import filedialog
import  os


# Importing the Keras libraries and packages
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import math

def get_popen(idata, N=1):
    samples=5

    totalin=0		
    datain=list(idata)
    for j in range(N+1):
    	z=datain.count(j)*j
    	totalin+=z
    return totalin/(len(datain)*N)

from keras import backend as K
def mcor(y_true, y_pred):
     #matthews_correlation
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
    
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
    
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
    
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
     
     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
     
     return numerator / (denominator + K.epsilon())

def precision(y_true, y_pred):
    """Precision metric.
    
    Only computes a batch-wise average of precision.
    
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision



def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

import tensorflow as tf

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def print_nice(cm, classes=6):
    j=0
    for i in range(classes):
        for k in range (classes):
            print("{0:5d},".format(cm[i,k]),end="")
        j=j+sum(cm[i])
        print()
    print (j)

def matchlen(data1,data2):
    if len(data1)<len(data2):
        data2=data2[:len(data1)]
    elif len(data1)>len(data2):     
        data1=data1[:len(data2)]
    
    return [data1, data2]

def make_roc(gt,cpl,cl):
    from sklearn.preprocessing import label_binarize
    y_predict = label_binarize(gt, classes=[0, 1, 2, 3, 4, 5])
    print('c=',cl)
    y = label_binarize(cl, classes=[0, 1, 2, 3, 4, 5])
    n_classesi = y.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    from sklearn.metrics import roc_curve, auc
    for i in range(n_classesi):
        fpr[i], tpr[i], thre = roc_curve(y_predict[:, i], cpl[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print('state=, {}, auc=,{}'.format(i,roc_auc[i]))
    

def do_file(file, half,skm, dataset, timep, idataset,
            chunk, notch, butter, plot, fc, batch_size,classes, 
            loaded_model):
    train_size = int(len(dataset))
    in_train = dataset[:,1]
    '''
    BE SURE TO ADD THIS LINE TO PREDICTOR
    '''
    bites=batch_size
    for i in range(0,(len(dataset)-bites),bites):
        dataset[i:i+bites,2]=dataset[i:i+bites,2]-np.mean(dataset[i:i+bites,2]) 
    scalerize=False
    if scalerize==True:    
        forscaler=dataset[:,2].reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset[:,2] = scaler.fit_transform(forscaler)[:,0]    
        
    #plt.plot(in_train[0:2000])
    lowcut=0
    if notch==True:
        from scipy import signal
        si=(timep[15]-timep[5])/10
        fs = 1/si  # Sample frequency (Hz)
        if butter==True:
            lowcut= 1000.0
            nyq = 0.5 * fs
            low = lowcut / nyq
            b, a = signal.butter(4, low, 'low')
            in_train = signal.filtfilt(b, a, in_train)   
        f0 = fc  # Frequency to be removed from signal (Hz)
        bw = 20.0 # Band width to filter
        Q = f0/bw # Quality factor
        b, a = signal.iirnotch(f0, Q, fs=fs)
        in_train = signal.filtfilt(b, a, in_train)
    else:
        lowcut='None'
    #plt.plot(in_train[0:2000])
        
    target_train = idataset
    in_train = in_train.reshape(len(in_train),1,1,1)
    
    output=[]
    chunksize=batch_size*1
    in_data=in_train
    
    #in_train = in_train.reshape(len(in_train),1,1)
    
    #loaded_model.summary()
    if chunk==True:
        for i in range(0,len(dataset),chunksize):
            #scaling if used
            output.extend(loaded_model.predict_classes(in_data[i:i+chunksize],batch_size=batch_size,verbose=True))
        c=output
    else:
        c = loaded_model.predict_classes(in_train, batch_size=batch_size, verbose=True)
    target_train,half=matchlen(target_train,half)
    target_train,skm=matchlen(target_train,skm)
    c,target_train=matchlen(c,target_train)
    
    
    cm = confusion_matrix(target_train, c)
    print("classification report for DC on:",file)
    print("notch and lowpass filtering at",notch,lowcut)
    report1=classification_report(target_train,np.around(c),output_dict=True)
    print(classification_report(target_train,np.around(c)))
    print(cm)
    print_nice(cm, classes=classes)
    target_train,half=matchlen(target_train,half)
    target_train,skm=matchlen(target_train,skm)
    c,target_train=matchlen(c,target_train)
    print('Popen = ,', get_popen(c, N=max(c)))
    cp = loaded_model.predict(in_train, batch_size=batch_size, verbose=True)
    cp,target_train=matchlen(cp,target_train)
    make_roc(target_train,cp,c)

    cm = confusion_matrix(target_train, half)
    print("classification report for HALFMAX:")
    print("notch and lowpass filtering at",notch,lowcut)
    report2=classification_report(target_train,np.around(half),output_dict=True)
    print(classification_report(target_train,np.around(half)))
    print(cm)
    print_nice(cm, classes=classes)
    print('Popen = ,', get_popen(half, N=max(half)))

    cm = confusion_matrix(target_train, skm)
    print("classification report for  SKM:")
    print("notch and lowpass filtering at",notch,lowcut)
    report3=classification_report(target_train,np.around(skm),output_dict=True)
    print(classification_report(target_train,np.around(skm)))
    print(cm)
    print_nice(cm, classes=classes)
    print('Popen = ,', get_popen(skm, N=max(skm)))
    print("ML f1-0, f1-1, HALF f1-0, f1-1,SKM F1-0, f1-1")
    print("{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}".format( report1["0"]["f1-score"],
          report1["1"]["f1-score"], report2["0"]["f1-score"],
          report2["1"]["f1-score"], report3["0"]["f1-score"],
          report3["1"]["f1-score"]))
    print("F1 Macro Average f-1 score ML, HALF, SKM")
    print("{:.5f},{:.5f},{:.5f}".format( report1["macro avg"]["f1-score"],
          report2["macro avg"]["f1-score"],report3["macro avg"]["f1-score"],))
    
    if plot==True:
        lenny=2000
        ulenny=5000
        plt.figure(figsize=(30,6))
        plt.subplot(4,1,1)
        
        plt.plot(dataset[lenny:ulenny,1], color='blue', label="the raw data")
        plt.title("The raw test")
        
        plt.subplot(4,1,2)
        plt.plot(target_train[lenny:ulenny], color='black', label="the actual idealisation")
        plt.subplot(4,1,3)
        plt.plot(c[lenny:ulenny], color='red', label="predicted idealisation")
        
        plt.xlabel('timepoint')
        plt.ylabel('current')
        plt.legend()
        plt.show()
    return 0
'''
########
#START HERE !!!!!
########
'''
chunk=False
notch=False
butter=False
plot=False
loadmodel=False
fc=50
batch_size=256
print("works nicely with  appropriate file naming conventions\n")
if loadmodel==True:
    print("loading model")
    loaded_model = load_model(r'P:\Code\MLG channels-Lowery\CED Tests\260219\testtest\nmn_oversampled_deepchanel2_5.h5', 
                              custom_objects={'mcor': mcor, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}) 
    print("model loaded")
root = filedialog.Tk()
files = filedialog.askopenfilenames(parent=root,title='Choose a file')
filenames = root.tk.splitlist(files)
root.withdraw()
for file2open in filenames:
	df30 = pd.read_csv(file2open,header=None)
	dataset=df30.values
	dataset = dataset.astype('float64')
	#dataset = dataset[:800000]
	timep=dataset[:,0]
	#maxchannels=10
	maxer=np.amax(dataset[:,2])
	print (maxer)
	maxeri=maxer.astype('int')
	maxchannels=maxeri
	idataset=dataset[:,2]
	idataset=idataset.astype(int)
	print (file2open)
	print('real Popen , {:0.5f}'.format(get_popen(idataset, N=max(idataset))))
	'''
	[NF=50Hz]
	'''
	half_f=file2open.replace(".csv","[NF=50Hz]_halfamp.txt")
	df30 = pd.read_csv(half_f,header=None,delimiter="\t")
	tdataset=df30.values
	half=tdataset[:,0].astype(int)
	skm_f=file2open.replace(".csv","[NF=50Hz]_SKM.txt")
	df30 = pd.read_csv(skm_f,header=None,delimiter="\t")
	tdataset=df30.values
	skm=tdataset[:,0].astype(int)

	trial = do_file(file2open,half,skm, dataset,timep,idataset,
					chunk, notch, butter, plot, fc, batch_size, maxeri,
					loaded_model=loaded_model)    



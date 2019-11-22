# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:53:49 2019

@author: ncelik34
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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




batch_size=256

Qubname='outfinaltest3(NF=50Hz)_SKM_F1.csv'
Qub2name='outfinaltest3(NF=50Hz)_halfamp_F1.csv'
Dname='outfinaltest3_noise.csv'
df30 = pd.read_csv(Dname,header=None)
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

#scaler = preprocessing.RobustScaler()
#dataset = scaler.fit_transform(dataset)
#scaler = preprocessing.StandardScaler()
#dataset = scaler.fit_transform(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

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


train_size = int(len(dataset))

in_train = dataset[:,1]
target_train = idataset
in_train = in_train.reshape(len(in_train),1,1,1)
#in_train = in_train.reshape(len(in_train),1,1)

loaded_model = load_model('nmn_oversampled_deepchanel2_5.h5', custom_objects={'mcor': mcor, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc})
#loaded_model = load_model(model) #should be fine... save model in non default way?
#loaded_model = load_model('attention_lstm1.h5')
#c = loaded_model.predict(in_train, batch_size=batch_size, verbose=True)
#cmax=np.argmax(c,axis=1)
loaded_model.summary()

c = loaded_model.predict_classes(in_train, batch_size=batch_size, verbose=True)
#cp = loaded_model.predict(in_train, batch_size=batch_size, verbose=True)
#post processing more than 1 channel idealisation
#cn1 = c
#lenc = len(cn1)
#for k in range (lenc):
#    avc = c[k]
#    if avc ==maxeri+1:
#        avc=avc-1
#    c[k] = avc

print(target_train[:20])
print(c[:20])
#i_class1 = np.where(c == 1)[0]
#i_class2 = np.where(target_train == 1)[0]
#c=c.reshape(len(in_train),1)
#from sklearn.metrics import confusion_matrix
cm_dc = confusion_matrix(target_train, c)
#

df_qub = pd.read_csv(Qubname,header=None)
dataset_q=df_qub.values
idataset_q=dataset_q[:,0]
idataset_q=idataset_q.astype(int)
cm_q = confusion_matrix(target_train, idataset_q)





df_qub2 = pd.read_csv(Qub2name,header=None)
dataset_q2=df_qub2.values
idataset_q2=dataset_q2[:,0]
idataset_q2=idataset_q2.astype(int)
cm_q2 = confusion_matrix(target_train, idataset_q2)



#cm24 = confusion_matrix(target_train, cn1)

#print(roc_auc_score(target_train,c))
print(Qubname)
print(Dname)
print("classification report of DC:")
print(classification_report(target_train,np.around(c)))
print("classification report of QuB SKM:")
print(classification_report(target_train,np.around(idataset_q)))
print("classification report of QuB half-amp:")
print(classification_report(target_train,np.around(idataset_q2)))
#pre-processing for real trace for one-channel process(fiona trace):
#cn1 = c
#lenc = len(cn1)
#for k in range (lenc):
#    avc = c[k]
#    if avc>1:
#        avc=avc-1
#    cn1[k] = avc





lenny=2000
ulenny=5000
plt.figure(figsize=(30,6))
plt.subplot(4,1,1)

plt.plot(dataset[lenny:ulenny,1], color='blue', label="the raw data")
plt.title("The raw test")

plt.subplot(4,1,2)
plt.plot(target_train[lenny:ulenny], color='black', label="the actual idealisation")

#for fret data in total 1000 points
#plt.subplot(3,1,2)
#plt.plot(dataset[lenny:ulenny,3], color='black', label="the actual idealisation")

#line,=plt.plot(c[:lenny], color='red', label="predicted idealisation")
plt.subplot(4,1,3)
plt.plot(c[lenny:ulenny], color='red', label="predicted idealisation")
#plt.setp(line, linestyle='--')
#plt.subplot(4,1,4)
#plt.plot(idataset_q[lenny:ulenny], color='brown', label="QuB idealisation")


plt.xlabel('timepoint')
plt.ylabel('current')
#plt.savefig(str(rnd)+'data.png')
#plt.savefig('destination_path.tiff', format='tiff', dpi=300)
plt.legend()
plt.show()


#x1=dataset[lenny:ulenny,1]
#x2=target_train[lenny:ulenny]
#x3=c[lenny:ulenny]
#cnd2 = np.asarray(cn2)

#histogram distribution:
#from scipy.stats import norm
#mu, std = norm.fit(c)
#counts, bins = np.histogram(c)
#plt.hist(bins[:-1], bins=6, range=(-0.8,1.8),density=False,weights=counts)
#
#plt.hist(bins[:-1], bins, weights=counts)
#p = norm.pdf(mu, std)
#plt.plot(p, 'k', linewidth=2)
#
#plt.show()
#
#import matplotlib.mlab as mlab
#plt.figure(1)
#plt.hist(c, normed=True)
#plt.xlim((min(c), max(c)))
#
#mean = np.mean(c)
#variance = np.var(c)
#sigma = np.sqrt(variance)
#x = np.linspace(min(c), max(c), 100)
#plt.plot(x, mlab.normpdf(x, mean, sigma))
#
#plt.show()
#list11=list(tpr.values())
#list12=fpr[1]

#box plot
#joinedlist=np.concatenate((predict, class_predict[:,None]),axis=1)
#list2 = predict[:,0]
#list3 = list2[list2>0.1]
#np.mean(list3)
#fig1, ax1 = plt.subplots()
#ax1.set_title('Basic Plot')
#ax1.boxplot(list2)

#d={'Time':timep[:train_size],'Raw':dataset[:train_size,1],'real state':target_train[:train_size],'prediction':c[:train_size]}
#df=pd.DataFrame(data=d)
#df.to_csv('presults_op_outfinaltest161_noise.csv')







##roc curve plotting for multiple
#from sklearn.preprocessing import label_binarize
#y = label_binarize(c, classes=[0, 1, 2, 3, 4, 5])
##y_predict = label_binarize(target_test, classes=[0, 1, 2, 3, 4, 5])
#y_predict = label_binarize(target_train, classes=[0, 1, 2, 3, 4, 5])
#n_classesi = y.shape[1]
#
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#from sklearn.metrics import roc_curve, auc
#for i in range(n_classesi):
#    #fpr[i], tpr[i], thre = roc_curve(y_predict[:, i], predict[:, i])
#    fpr[i], tpr[i], thre = roc_curve(y_predict[:, i], cp[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
#    
#
#plt.figure()
#lw = 2
#plt.plot(fpr[2], tpr[2], color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()
#
######
#from itertools import cycle
#plt.figure(2)
#plt.xlim(0, 1)
#plt.ylim(0, 1)
#colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','black','yellow'])
#for i, color in zip(range(n_classesi), colors):
#    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#             label='ROC curve of class {0} (area = {1:0.2f})'
#             ''.format(i, roc_auc[i]))
#
##plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##plt.semilogy([0, 1], [0, 1], 'k--', lw=lw)
##plt.xlabel('False Positive Rate')
##plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate (1 - Specificity)')
#plt.ylabel('True Positive Rate (Sensitivity)')
#plt.title('Zooom in View: Some extension of ROC to multi-class')
#plt.legend(loc="lower right")
#plt.show()






# standard deviation of the dataset:
x_input = dataset[:,1]
mean_x = sum(x_input) / np.count_nonzero(x_input)

sd_x = math.sqrt(sum((x_input - mean_x)**2) / np.count_nonzero(x_input))

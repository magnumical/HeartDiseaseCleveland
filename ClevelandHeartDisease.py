# -*- coding: utf-8 -*-
"""
@author: REZA ََAMINI
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam


#%%
def create_model():
    model=Sequential()
    model.add(Dense(13,input_dim=13,kernel_initializer='normal',activation='relu'))
    model.add(Dense(10,input_dim=13,kernel_initializer='normal',activation='relu'))
    model.add(Dense(8,input_dim=10,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    adam=Adam(lr=0.01)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
   
    return model


#%%
names=['age', 'sex','chest pain','rest_bps','cholestrol',
       'fast_bs','rest_ecg','max_hr','exc',
       'ST depression','Peak exercise','No_vessl','thalassemia','class']

df = pd.read_csv('cleveland.csv',names=names)
info=df.describe()
#%%
df.dropna(inplace=True)
info2=df.describe()
#%%


df['class'].replace([2,3,4],1,inplace=True)

#%%
dataset=df.values


            
#%%
X=dataset[:,0:13]
Y=dataset[:,13]

scalar=StandardScaler().fit(X)
X_norm=scalar.transform(X)

data=pd.DataFrame(X_norm)
dg=data.describe()
#%%
model=create_model()
print(model.summary())

#%%
    
s=time.time()
#%%
model= KerasClassifier(build_fn=create_model,validation_split=0.2)
batch_size=[10,20,30]
epoch=[10,50,100]
param_grid= dict(batch_size=batch_size,epochs=epoch)
grid=GridSearchCV(estimator=model,param_grid=param_grid,verbose=2)
grid_results=grid.fit(X_norm,Y)
print(grid.best_score_)

el=time.time()-s 

#%%
print("Best: %f using %s" % (grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:55:34 2024

@author: iyaku
"""
import numpy as np
test_ratn_values=np.load('test_ratn_values.npy')
pred_cb=np.load('content_based_predictions.npy')
pred_cf=np.load('collab_filtering_predictions.npy')

preds=np.zeros(len(test_ratn_values))
nmae=np.zeros(11)
mae=np.zeros(11)
for i in range(0,11):
    alfa=0.1*i
    for j in range(0,len(test_ratn_values)):
        preds[j]=alfa*pred_cb[j]+(1-alfa)*pred_cf[j]
    #Computing NMAE
    mae_sum=0
    for j in range(0,len(test_ratn_values)):
        mae_sum=mae_sum+abs(test_ratn_values[j]-preds[j])
    mae[i]=mae_sum/len(test_ratn_values)
    nmae[i]=mae[i]/(5-0.5) #Our rating range is [0.5, 5] with each step 0.5
        
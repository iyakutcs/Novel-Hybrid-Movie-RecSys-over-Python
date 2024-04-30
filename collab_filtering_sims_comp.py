# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:12:12 2024

@author: iyaku
"""
#Data Initializations
import numpy as np
import pandas as pd
import math
train_ratings= pd.read_csv(r"C:\Users\iyaku\Desktop\HybridCF/train_ratings.txt", delimiter='\t') 
test_ratings= pd.read_csv(r"C:\Users\iyaku\Desktop\HybridCF/test_ratings.txt", delimiter='\t')
#test_ratn_indices=np.load('test_ratn_indices.npy') we may not need this!
test_ratn_values=np.load('test_ratn_values.npy')
df = pd.read_csv(r"C:\Users\iyaku\OneDrive\Masaüstü\PythonTrain\PyRecs/movie_genres.dat", delimiter='\t', encoding='ISO-8859-1') 
movi = df['movieID'].to_numpy()
movi #now movi holds repeating index number depending on the size of genre set of the item.
mis=np.unique(movi)

#We use initial knowledge of number of users and items 
no_users=2113
no_items=10197
item_sims=np.zeros((no_items,no_items))
for i in range(0, no_items):
    print(i)
    dfi = train_ratings[(train_ratings['movieID']==mis[i])]
    mni_df=dfi[['rating']].mean(axis='index')
    mni=float(mni_df.iloc[0])
    #norm_itmi=dfi[['rating']]-mn_itm
    nume=0
    deni=0
    denj=0
    for j in range(i+1,no_items):
        dfj = train_ratings[(train_ratings['movieID']==mis[j])]
        mnj_df=dfj[['rating']].mean(axis='index')
        mnj=float(mnj_df.iloc[0])
        #norm_itmj=dfj[['rating']]-mn_itm
        usri=dfi['userID'].to_numpy()
        usrj=dfj['userID'].to_numpy()
        com_rated_set=set(usri)&set(usrj)
        com_rated=list(com_rated_set)
        for jk in range(0, len(com_rated)):
            comi=dfi[(dfi['userID']==com_rated[jk])]
            ri=float(comi["rating"].iloc[0])
            rni=ri-mni
            comj=dfj[(dfj['userID']==com_rated[jk])]
            rj=float(comj["rating"].iloc[0])
            rnj=rj-mnj
            nume=nume+rni*rnj
            deni=deni+rni*rni
            denj=denj+rnj*rnj
        deno=math.sqrt(deni)*math.sqrt(denj)
        if deno!=0:
            item_sims[i,j]=nume/deno
np.save('item_sims_original.npy', item_sims)
for i in range(1,no_items):#row
    for j in range (0,i): #column
        item_sims[i,j]=item_sims[j,i]
np.save('item_sims_symmetric.npy', item_sims)
        
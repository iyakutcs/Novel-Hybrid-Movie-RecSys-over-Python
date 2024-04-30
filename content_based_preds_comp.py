# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 08:50:04 2024


"""
import numpy as np
import pandas as pd
df = pd.read_csv(r"C:\Users\iyaku\OneDrive\Masaüstü\PythonTrain\PyRecs/movie_genres.dat", delimiter='\t', encoding='ISO-8859-1') 
movi = df['movieID'].to_numpy()
mis=np.unique(movi)
df7 = pd.read_csv(r"C:\Users\iyaku\OneDrive\Masaüstü\PythonTrain\PyRecs/user_ratedmovies.dat", delimiter='\t', encoding='ISO-8859-1')

test_size=round(855598*0.2) #Train/test ratio=0.8 which is based on Sarwar et al's 2001 paper.
train_size=855598-test_size
rated_perm=np.random.permutation(855598)
test_rat_indc=rated_perm[0:test_size]
train_ratings=df7.drop(test_rat_indc, axis=0) #holds 80% of randomly selected ratings to be used in training
test_ratings=df7.loc[test_rat_indc]

#We run one-time randomized permutation mechanism above to obtain test and train ratings then go over them, only!
igdca=np.load('intersize_gdca_norm_symmetric.npy') #igdca is content-based item similarity matrix. It is normalized and symmetric!
k=30
preds=np.zeros(len(test_rat_indc))
for i in range(0,len(test_rat_indc)):
    print(i)
    rat_row=test_ratings.loc[i]
    usr_ID=int(rat_row["userID"])
    dfu = train_ratings[(train_ratings['userID']==usr_ID)]
    mn_usr=dfu[['rating']].mean(axis=0)
    dfn7=dfu[['rating']]-mn_usr #all ratings of user is normalized by user mean
    norm_ratings=dfn7['rating'].to_numpy()
    mov_ID=int(rat_row["movieID"])
    mvin=np.where(mis==mov_ID)#returns 2
    sorted_index_array = np.argsort(igdca[mvin,:])
    desc_indices=np.flip(sorted_index_array) #take reverse of array
    nume=0
    deno=0
    usratd= dfu['movieID'].to_numpy()#list of user rated movies
    dic=[]
    for ix in range(0,k):
        dic.append(desc_indices[0,0,ix]) #dic holds k similar item indices
    rtneig=np.intersect1d(mis[dic],usratd) 
    for j in rtneig: #k=500 similar items
        ji=np.where(usratd==j) #finds index in specified user ratings
        ju=np.where(mis==j) #finds original whole index in igdca
        fnum=float(norm_ratings[ji])*float(igdca[mvin,ju])
        nume=nume+fnum #Data frame olup nan veriyorlar çok saçma
        deno=deno+igdca[mvin,ju]
    if deno!=0:
        pred=float(mn_usr.iloc[0]) +float(nume/deno)
    else:
        pred=float(mn_usr.iloc[0])    
    preds[i]=pred
tst_rats_np=test_ratings['rating'].to_numpy()
#Computing NMAE
mae_sum=0
for i in range(0,len(tst_rats_np)):
    mae_sum=mae_sum+abs(tst_rats_np[i]-preds[i])
mae=mae_sum/len(tst_rats_np)
nmae=mae/(5-0.5) #Our rating range is [0.5, 5] with each step 0.5

train_ratings.to_csv('train_ratings.txt',sep='\t',index=True)
test_ratings.to_csv('test_ratings.txt',sep='\t',index=True)
np.save('content_based_predictions.npy',preds)
np.save('test_ratn_values.npy',tst_rats_np)
np.save('test_ratn_indices.npy',test_rat_indc)


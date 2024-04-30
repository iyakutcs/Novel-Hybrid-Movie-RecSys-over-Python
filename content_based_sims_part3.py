# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:33:24 2024

@author: iyakut
"""

import pandas as pd
import numpy as np
df5 = pd.read_csv(r"C:\Users\iyaku\OneDrive\Masaüstü\PythonTrain\PyRecs/movie_actors.dat", delimiter='\t', encoding='ISO-8859-1') 
df5 #231742 x4 
df5.isnull().values.any() #Returns true. We apply dropna()
df5=df5.dropna() 
df5 #5 rows are removed due to n/a values.
moviac = df5['movieID'].to_numpy()
moviact=np.unique(moviac)
###WARNING: IT CAN BE COMMON OPERATION.
df = pd.read_csv(r"C:\Users\iyaku\OneDrive\Masaüstü\PythonTrain\PyRecs/movie_genres.dat", delimiter='\t', encoding='ISO-8859-1') #UNICODE HATASI şu encoding argümanıyla çözüldü
movi = df['movieID'].to_numpy()
movi #now movi holds repeating index number depending on the size of genre set of the item.
mis=np.unique(movi)
mis #we get rid of repeating indices in movi. mis holds unique movie ids
len(mis) #Note that mis size 10197, 
len(moviact)#the size of unique movie list 10174 so 23 movie's actor information is missing in movie_actors.data
dfa = df5.groupby('movieID')['actorName'].apply(list) # *** This is very useful for our PyRecs project!
dfa #grouped actors list. MovieID corresponds related actors list having size 1 to free size
n=len(mis) #number of movies in whole data
intersize=np.zeros((n,n)) 
for i in range(0, len(moviact)): #range is ok
    print(i)
    for j in range(i+1,len(moviact)):
        indx=np.where(mis==moviact[i])
        indy=np.where(mis==moviact[j])
        intersize[indx,indy]=len(set(dfa.loc[moviact[i]])&set(dfa.loc[moviact[j]]))
overfive=np.where(intersize>5) #We want to crop intersection size values over five
intersize(overfive)=5
np.save('intersize_actors_only.npy', intersize)

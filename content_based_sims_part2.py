# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:47:32 2024

@author: iyakut
"""
import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\iyaku\OneDrive\Masaüstü\PythonTrain\PyRecs/movie_genres.dat", delimiter='\t', encoding='ISO-8859-1') 
movi = df['movieID'].to_numpy()
movi #now movi holds repeating index number depending on the size of genre set of the item.
mis=np.unique(movi)
mis #we get rid of repeating indices in movi. mis holds unique movie ids
#This part is for third computation component of content-based engine about countries.
df4 = pd.read_csv(r"C:\Users\iyaku\OneDrive\Masaüstü\PythonTrain\PyRecs/movie_countries.dat", delimiter='\t', encoding='ISO-8859-1')
df4.isnull().values.any() #Returns true. df4 involves some null values
df4=df4.dropna() #By this row, we get rid of null values. 8 rows involving null value are removed from data frame df4
movic = df4['movieID'].to_numpy() #there are 10189 unique movies
n=10197
intersize=np.zeros((n,n))
countries=df4['country'].to_numpy()
set_cou=set(countries)
for i in range(0,len(movic)):
    print(i)
    indx=np.where(mis==movic[i])
    prc=countries[i]
    for j in range(i+1,len(movic)):
        if countries[j]==prc:
            indy=np.where(mis==movic[j])
            intersize[indx,indy]=1
np.save('intersize_countries_only.npy', intersize)

            
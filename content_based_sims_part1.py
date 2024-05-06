#In this part we compute genres and directors contribution to the intersection size ('intersize')

import numpy as np
import pandas as pd
df = pd.read_csv(r"C:\Users\iyaku\OneDrive\Masaüstü\PythonTrain\PyRecs/movie_genres.dat", delimiter='\t', encoding='ISO-8859-1') 
#Note that avg. 2.040 genres per movie
df2 = df.groupby('movieID')['genre'].apply(list) # *** This is very useful for our PyRecs project!
df2 #grouped genres list. MovieID corresponds related genre list having size 1 to free size
movi = df['movieID'].to_numpy()
movi #now movi holds repeating index number depending on the size of genre set of the item.
mis=np.unique(movi)
mis #we get rid of repeating indices in movi. mis holds unique movie ids
n=10197 #number of movies in whole data
intersize=np.zeros((n,n)) 

for i in range(0,n): #range is ok
    for j in range(i+1,n):
        intersize[i,j]=len(set(df2.loc[mis[i]])&set(df2.loc[mis[j]]))
        #intersize[i+1,j+1]=size_intersect(df2.loc[mis[i]],df2.loc[mis[j]])

np.save('intersize_genres.npy', intersize)

# Matching director names
#This part is the next computation component of content-based engine about directors.
import pandas as pd
import numpy as np
from itertools import combinations
df3 = pd.read_csv(r"C:\Users\iyaku\OneDrive\Masaüstü\PythonTrain\PyRecs/movie_directors.dat")
dirnames=df3['directorName'].to_numpy()
set_dirs=set(dirnames)
#there are totally 4053 distinct directors totally 10155 directors listed. There are missing director names
for i in set_dirs: #iterate for each director i in the set.
    filtered_df = df3.loc[df3['directorName'].str.contains(i)]
    dirmov = filtered_df['movieID'].to_numpy()
    #now we obtained data frame for director i
    movpair = list(combinations(dirmov, 2)) #we obtain each distinct movieID pairs for director i 
    for j,k in movpair:
        indx = np.where(mis == j)
        indy=np.where(mis==k)
        #print(indx, indy)
        intersize[indx,indy]=intersize[indx,indy]+1
        
np.save('intersize_gen_dir.npy', intersize)

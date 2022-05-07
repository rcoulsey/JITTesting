#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np

from time import time


# In[ ]:


##Direct implimentation

def elimination(M, N):
    A = np.random.rand(M,N)
    tic = time()
    

    for i in range(M):
        for k in range(i+1, M):
            ratio = A[k][i]/A[i][i]
            for j in range(M):
                A[k][j] = A[k][j] - A[i][j]*ratio


    toc = time()
    return A, tic, toc


tic = []
tok = []

for i in np.arange(10, 300, 10):
    A, ttic, ttoc = elimination(i, i)
    tic.append(ttic)
    tok.append(ttoc)


O1 = [tok[i] - tic[i] for i in range(len(tic))]

file = open('GaussNPRes.txt', 'w')
file.write(str(O1))
file.close()


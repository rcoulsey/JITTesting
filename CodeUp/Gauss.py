from time import time
from random import randint



def elimination(M, N):
    A = []
    for i in range(M):
        row = []
        for j in range(N):
            row.append(int(randint(1, 99)))
        A.append(row)
        

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

for i in range(10, 300, 10):
    A, ttic, ttoc = elimination(i, i)
    tic.append(ttic)
    tok.append(ttoc)


O = [tok[i] - tic[i] for i in range(len(tic))]

file = open('GaussRe_naitives.txt', 'w')
file.write(str(O))
file.close()

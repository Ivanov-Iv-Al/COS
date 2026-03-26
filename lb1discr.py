import numpy as np
import matplotlib.pyplot as plt

s1t = 0.25
s2t = 0
    
D = 0.25

er = 0

N = 0

#n = 0.25+np.random.normal(0, np.sqrt(D))

S = 0

Sw = 0

#zt1 = s1t + n
zt2 = 0.2

while er < 3:

    n = 0.25 + np.random.normal(0, np.sqrt(D))

    zt1 = s1t + n

    res1 = np.exp(-((np.power(zt1 - s1t,2)/(2*D))))

    res2 = np.exp(-((np.power(zt1 - s2t,2)/(2*D))))

    pravd = res1/res2


    print(f"Реузльтат 1: {res1}")

    print(f"Реузльтат 2: {res2}")

    print(f"Отношение правдоподобия: {pravd}")

    if pravd >= 1:
        S = 1
        #print(S)
        #N+=1
    else:
        S = 0
        #N+=1
        #print(S)

    wt = (s1t+s2t)/2

    if zt1 > wt:
        Sw = 1
        N+=1
        #print(Sw,zt1)
    else:
        Sw = 0
        N+=1
        er+=1
        #print(Sw,zt1)

    print(f"Кол-во ошибок: {er}")
    print(f"Общее кол-во переданных бит:{N}")
    #print(f"BER: {er/N}")




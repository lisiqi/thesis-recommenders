# Implement the four evaluation metrics: MAP, MP, NDCG, ERR #

from __future__ import division

import numpy as np
from math import log
from math import pow

def precision_at_k(truelist, predlist, k):
    return len(set(truelist) & set(predlist[:k]))/k

# predlist has length 10 in iALS
def AveP(truelist, predlist):
    sum = 0
    for k in range(0,len(predlist)):
        if(predlist[k] in truelist):
            sum = sum + precision_at_k(truelist, predlist, k+1)
    # return sum/len(truelist) # as definition
    return sum / min(len(truelist), 10) # as in GPPW

def MAP(dict_actual,dict_recommended):
    sum = 0
    for key in dict_actual.keys():
        truelist = dict_actual[key]
        predlist = dict_recommended[key]
        sum = sum + AveP(truelist, predlist)
    return sum/len(dict_actual)

def MP(dict_actual,dict_recommended):
    sum = 0
    for key in dict_actual.keys():
        truelist = dict_actual[key]
        predlist = dict_recommended[key]
        sum = sum + precision_at_k(truelist, predlist, 10)
    return sum / len(dict_actual)

def NDCG(dict_actual, dict_recommended):
    ndcg = list()
    for user in dict_actual.keys():
        recommended_items = list(dict_recommended[user])

        dcg = 0
        for i in range(0,10):
            if recommended_items[i] in dict_actual[user]: # relevant
                y = 1
            else:
                y = 0
            dcg = dcg + (pow(2, y) - 1) / log(i+1 + 1, 2)

        idcg = 0
        for i in range(0, len(dict_actual[user])):
            idcg = idcg + (pow(2, 1) - 1) / log(i+1 + 1, 2)

        ndcg.append(dcg/idcg)  # idcg will never be 0
    return np.mean(ndcg)

def ERR(dict_actual, dict_recommended):
    gmax = 1
    f_err = list()
    for user in dict_actual.keys():
        recommended_items = list(dict_recommended[user])

        r = [None] * 10
        cul = [None] * 10
        err = 0
        for i in range(0, 10):
            if recommended_items[i] in dict_actual[user]:  # relevant
                g = 1
            else:
                g = 0

            # use binary relevance
            r[i] = (pow(2,g)-1)/pow(2,gmax)

            if i==0:
                cul[i] = 1
            else:
                cul[i] = cul[i-1]*(1-r[i-1])

            err = err + (1/(i+1))*cul[i]*r[i]

        f_err.append(err)
    return np.mean(f_err)
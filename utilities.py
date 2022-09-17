import numpy as np

def flatten2Dlist(nestedList):
    return [val for sublist in nestedList for val in sublist]


def emptyLists(numList):
    return [[] for _ in range(numList)]


def zerosLists(numList, lengthList):
    return [[0] * lengthList for _ in range(numList)]


def emptyDicts(numDict, keys, lengthList):
    if lengthList == 0:
        dicts = [dict(zip(keys, emptyLists(numList=len(keys)))) for _ in range(numDict)]
    else:
        dicts = [dict(zip(keys, zerosLists(numList=len(keys), lengthList=lengthList))) for _ in range(numDict)]

    if numDict > 1:
        return dicts
    elif numDict == 1:
        return dicts[0]

    
def nanArray(lengthArray):
    return np.empty(lengthArray) * np.nan

    
def nanArrays(numArray, lengthArray):
    return [np.empty(lengthArray) * np.nan for _ in range(numArray)]
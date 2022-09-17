import itertools
import numpy as np
import pandas as pd
import sys

from utilities import *
from taskSetting import *
from expsInfo import *
from modelSpecificities import *
from funcs_loadData import *

expVersion = sys.argv[1]
model = sys.argv[2]
iWorker = int(sys.argv[3])
numGamePerType = int(sys.argv[4])

fittingAlgorithm = 'L-BFGS-B'
gameLength = 30
ifFlatPrior = False

_, workerIds = load_data(expVersion, getWorkerIds=True)
workerId = workerIds[iWorker]

print(model + '\niWorker' + str(iWorker) + '\nnumGamePerType' + str(numGamePerType), flush=True)

numPars = len(parameterName[model])

fileNameCollect = getFittingCollectFileName(model=model, expVersion=expVersion, ifFlatPrior=ifFlatPrior, fittingAlgorithm=fittingAlgorithm)
resultsCollect = pd.read_csv('fittingResultsCollect/' + fileNameCollect)

# get fitted parameters
pFit = [resultsCollect.loc[resultsCollect['workerId'] == workerId, parName].values[0] for parName in parameterName[model]]

# simulate
if 'infer' in model:  # SHT models
    dataSimu = modelFunction[model](model, pFit, gameLength=gameLength, numGamePerType=numGamePerType, ifSaveCSV=False)
else:  # other models
    dataSimu = modelFunction[model](model, pFit, ifFlatPrior=ifFlatPrior, gameLength=gameLength, numGamePerType=numGamePerType, ifSaveCSV=False)

fileNameSimu = getSimulationwParticiantsParFileName(model=model, ifFlatPrior=ifFlatPrior, workerId=workerId, fittingAlgorithm=fittingAlgorithm)
dataSimu.to_csv('modelSimulationwParticipantsPar/' + fileNameSimu, index=False)

import numpy as np
import pandas as pd
import csv
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import sys

from utilities import *
from taskSetting import *
from expsInfo import *
from modelSpecificities import *
from funcs_loadData import *

# Read input arguments
expVersion = sys.argv[1]
action = sys.argv[2]

# Load data
data, workerIds = load_data(expVersion, getWorkerIds=True)

# Fitting

if action == 'fit':

    model = sys.argv[3]
    iWorker = int(sys.argv[4])

    fileNameFitResults = getFittingFileName(ifCV=False, iCVgame=None, model=model, ifFlatPrior=ifFlatPrior,
                                            workerId=workerIds[iWorker], fittingAlgorithm=fittingAlgorithm)
    print(fileNameFitResults, flush=True)

    # Get the data and other arguments for fitting
    dataWorker = data[data['workerId'] == workerIds[iWorker]].copy()

    if ('Bayesian' not in model) and ('inferSerialHypoTesting' not in model):  # RL models
        dataFitting = prepDataFunction[model](dataWorker)
        likFunArgs = dataFitting
    elif 'Bayesian' in model:
        dataFitting = prepDataFunction[model](dataWorker, ifFlatPrior=ifFlatPrior)
        if model == 'workingMemoryBayesian':
            likFunArgs = (dataFitting, ifFlatPrior, window)
        else:
            likFunArgs = (dataFitting, ifFlatPrior)
    elif 'inferSerialHypoTesting' in model:
        dataFitting = prepDataFunction[model](model, dataWorker)
        likFunArgs = (model, dataFitting)

    numRep = 20 if not model == 'featureRLSlot' else 1

    p0List, negllhList, negllhFunReturnList, pList, successList, messageList, methodList = emptyLists(7)

    for iRep in range(numRep):

        np.random.seed(np.array([iWorker, iRep]))  # make sure things are replicable

        # initial parameter values
        p0 = getP0(model)
        print(p0, flush=True)

        # fitting
        if model == 'featureRLSlot':
            res = differential_evolution(likelihoodFunction[model], args=likFunArgs, bounds=bounds[model], disp=True)
            p0[:] = np.nan
        else:
            res = minimize(likelihoodFunction[model], x0=p0, args=likFunArgs, bounds=bounds[model], method=fittingAlgorithm, options={'disp': True})

        # check the fitted parameters indeed produce the correct likelihood
        if ('Bayesian' not in model) and ('inferSerialHypoTesting' not in model):
            negllh = likelihoodFunction[model](res.x, dataFitting, realdata=True)
        elif 'Bayesian' in model:
            negllh = likelihoodFunction[model](res.x, dataFitting, ifFlatPrior=ifFlatPrior, realdata=True)
        elif 'inferSerialHypoTesting' in model:
            negllh = likelihoodFunction[model](res.x, model, dataFitting, realdata=True)
        negllhList.append(negllh)

        # save fitting results
        p0List.append(p0)
        negllhFunReturnList.append(res.fun)
        pList.append(res.x)
        successList.append(res.success)
        messageList.append(res.message)
        methodList.append(fittingAlgorithm)

        print('iRep = ' + str(iRep) + ':\n', flush=True)
        print(res, flush=True)

        resultsDict = {'negllh': negllhList, 'negllhFunReturn': negllhFunReturnList, 'success': successList, 'message': messageList, 'method': methodList}
        for iPar in range(len(parameterName[model])):
            resultsDict['p0_' + parameterName[model][iPar]] = [p0List[i][iPar] for i in range(iRep+1)]
            resultsDict['pFit_'+parameterName[model][iPar]] = [pList[i][iPar] for i in range(iRep+1)]

        pd.DataFrame(resultsDict).to_csv('fittingResults/' + fileNameFitResults, index=False)

        # clear variables after each rep
        del p0, res, negllh


if action == 'fitInfer':

    numRep = 20

    model = sys.argv[3]
    input = int(sys.argv[4])
    iWorker = int(input % len(workerIds))
    iRep = int(np.floor(input / len(workerIds)))

    fileNameFitResults = getFittingFileName(ifCV=False, iCVgame=None, iRep=iRep, model=model, ifFlatPrior=None,
                                            workerId=workerIds[iWorker], fittingAlgorithm=fittingAlgorithm)
    print(fileNameFitResults, flush=True)

    if not os.path.exists('fittingResults/' + fileNameFitResults):

        # Get the data and other arguments for fitting
        dataWorker = data[data['workerId'] == workerIds[iWorker]].copy()
        dataFitting = prepDataFunction[model](model, dataWorker)
        likFunArgs = (model, dataFitting)

        # initial parameter values
        np.random.seed(np.array([iWorker, iRep]))  # make sure things are replicable
        p0 = getP0(model)
        print(p0, flush=True)

        # fitting
        res = minimize(likelihoodFunction[model], x0=p0, args=likFunArgs, bounds=bounds[model], method=fittingAlgorithm, options={'disp': True})

        # check the fitted parameters indeed produce the correct likelihood
        negllh = likelihoodFunction[model](res.x, model, dataFitting, realdata=True)

        print('iRep = ' + str(iRep) + ':\n', flush=True)
        print(res, flush=True)

        resultsDict = {'negllh': [negllh], 'negllhFunReturn': [res.fun], 'success': [res.success], 'message': [res.message], 'method': [fittingAlgorithm]}
        for iPar in range(len(parameterName[model])):
            resultsDict['p0_' + parameterName[model][iPar]] = [p0[iPar]]
            resultsDict['pFit_' + parameterName[model][iPar]] = [res.x[iPar]]
        pd.DataFrame(resultsDict).to_csv('fittingResults/' + fileNameFitResults, index=False)


if action == 'fitInferCollect':

    numRep = 20

    model = sys.argv[3]

    numPars = len(parameterName[model])

    for workerId in workerIds:

        p0List, pList, negllhList, negllhFunReturnList, successList, messageList, methodList = emptyLists(7)

        for iRep in range(numRep):

            fileNameFitResultsRep = getFittingFileName(ifCV=False, iCVgame=None, iRep=iRep, model=model, ifFlatPrior=None, workerId=workerId, fittingAlgorithm=fittingAlgorithm)
            
            if os.path.exists('fittingResults/' + fileNameFitResultsRep):
                
                resultsRep = pd.read_csv('fittingResults/' + fileNameFitResultsRep)

                # save fitting results
                p0List.append([resultsRep.loc[0, 'p0_' + parameterName[model][iPar]] for iPar in range(numPars)])
                pList.append([resultsRep.loc[0, 'pFit_' + parameterName[model][iPar]] for iPar in range(numPars)])
                negllhList.append(resultsRep.loc[0, 'negllh'])
                negllhFunReturnList.append(resultsRep.loc[0, 'negllhFunReturn'])
                successList.append(resultsRep.loc[0, 'success'])
                messageList.append(resultsRep.loc[0, 'message'])
                methodList.append(resultsRep.loc[0, 'method'])

                del fileNameFitResultsRep, resultsRep

            else:

                p0List.append([np.nan] * numPars)
                pList.append([np.nan] * numPars)
                negllhList.append(np.nan)
                negllhFunReturnList.append(np.nan)
                successList.append(False)
                messageList.append('file not existing...')
                methodList.append(np.nan)

                del fileNameFitResultsRep

        resultsDict = {'negllh': negllhList, 'negllhFunReturn': negllhFunReturnList, 'success': successList,
                       'message': messageList, 'method': methodList}
        for iPar in range(len(parameterName[model])):
            resultsDict['p0_' + parameterName[model][iPar]] = [p0List[i][iPar] for i in range(numRep)]
            resultsDict['pFit_' + parameterName[model][iPar]] = [pList[i][iPar] for i in range(numRep)]
        fileNameFitResults = getFittingFileName(ifCV=False, iCVgame=None, iRep=None, model=model, ifFlatPrior=None, workerId=workerId, fittingAlgorithm=fittingAlgorithm)
        pd.DataFrame(resultsDict).to_csv('fittingResults/' + fileNameFitResults, index=False)


# Collect the best fit

if action == 'collect':

    model = sys.argv[3]
    ifFlatPrior = None if 'Bayesian' not in model else False

    numPars = len(parameterName[model])
    resultsCollect = np.zeros((len(workerIds), numPars+1))

    for iWorker, workerId in enumerate(workerIds):
        # Get the fitting results file name
        fileNameFitResults = getFittingFileName(ifCV=False, iCVgame=None, model=model, ifFlatPrior=ifFlatPrior, workerId=workerIds[iWorker], fittingAlgorithm=fittingAlgorithm)
        resultsFitting = pd.read_csv('fittingResults/' + fileNameFitResults)

        if 'negllhFunReturn' in resultsFitting.keys():
            minIdx = resultsFitting.loc[(resultsFitting['success'] == True) & (np.abs(resultsFitting['negllh']) > 1) & (np.abs(resultsFitting['negllh'] - resultsFitting['negllhFunReturn'])<sys.float_info.epsilon), 'negllh'].idxmin()
        else:
            minIdx = resultsFitting.loc[(resultsFitting['success'] == True) & (np.abs(resultsFitting['negllh']) > 1), 'negllh'].idxmin()

        negllh = resultsFitting.loc[minIdx, 'negllh']
        pFit = [resultsFitting.loc[minIdx, 'pFit_' + parameterName[model][iPar]] for iPar in range(numPars)]
        resultsCollect[iWorker, :] = pFit + [negllh]

    
    fileNameCollect = getFittingCollectFileName(model=model, expVersion=expVersion, ifFlatPrior=ifFlatPrior, fittingAlgorithm=fittingAlgorithm)
    pd.DataFrame(resultsCollect, index=workerIds, columns=parameterName[model]+['negllh']).to_csv('fittingResultsCollect/'+fileNameCollect, index=True, index_label='workerId')

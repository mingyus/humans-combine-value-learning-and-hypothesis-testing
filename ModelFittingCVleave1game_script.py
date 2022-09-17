import numpy as np
import pandas as pd
from scipy.optimize import minimize
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

# Get game variables
numGames = pd.DataFrame.max(data['game'])
numGameConditions = data['numRelevantDimensions'].nunique() * data['informed'].nunique()

# Fit on training set

if action == 'fit':

    model = sys.argv[3]
    input = int(sys.argv[4])
    iWorker = int(np.floor(input / numGames))
    iCVgame = int(input % numGames)

    iRep = int(sys.argv[5])

    fileNameFitResults = getFittingFileName(ifCV=True, iCVgame=iCVgame, iRep=iRep, model=model, ifFlatPrior=ifFlatPrior, 
                        workerId=workerIds[iWorker], fittingAlgorithm=fittingAlgorithm)
    print(fileNameFitResults, flush=True)

    # Get the data
    dataWorker = data[data['workerId'] == workerIds[iWorker]].copy()

    # leave one game out
    gameTest = iCVgame + 1
    gamesFit = list(set(np.arange(numGames) + 1) - set([gameTest]))
    dataWorkerFit = pd.concat([dataWorker[dataWorker['game'] == iGame] for iGame in gamesFit])

    # prepare data for fitting
    if ('Bayesian' not in model) and ('inferSerialHypoTesting' not in model):  # RL models
        dataFitting = prepDataFunction[model](dataWorkerFit)
        likFunArgs = dataFitting
    elif 'Bayesian' in model:
        dataFitting = prepDataFunction[model](dataWorkerFit, ifFlatPrior=ifFlatPrior)
        likFunArgs = (dataFitting, ifFlatPrior)
    elif 'inferSerialHypoTesting' in model:  # infer models
        dataFitting = prepDataFunction[model](model, dataWorkerFit)
        likFunArgs = (model, dataFitting)

    # run fitting
    np.random.seed(np.array([iWorker, iCVgame, iRep]))  # make sure things are replicable
    p0 = getP0(model)

    res = minimize(likelihoodFunction[model], x0=p0, args=likFunArgs, bounds=bounds[model], method=fittingAlgorithm, options={'disp': True})

    # check the fitted parameters indeed produce the correct likelihood
    if ('Bayesian' not in model) and ('inferSerialHypoTesting' not in model):  # the majority of non-infer models
        negllh = likelihoodFunction[model](res.x, dataFitting, realdata=True)
    elif 'Bayesian' in model:
        negllh = likelihoodFunction[model](res.x, dataFitting, ifFlatPrior, realdata=True)
    elif 'inferSerialHypoTesting' in model:  # infer models
        negllh = likelihoodFunction[model](res.x, model, dataFitting, realdata=True)

    print(res, flush=True)

    resultsDict = {'negllh': [negllh], 'negllhFunReturn': [res.fun], 'success': [res.success], 'message': [res.message], 'method': [fittingAlgorithm]}
    for iPar in range(len(parameterName[model])):
        resultsDict['p0_' + parameterName[model][iPar]] = [p0[iPar]]
        resultsDict['pFit_' + parameterName[model][iPar]] = [res.x[iPar]]
    pd.DataFrame(resultsDict).to_csv('fittingResultsCV/' + fileNameFitResults, index=False)


# Collect fitting results

if action == 'fitCollect':

    model = sys.argv[3]

    numPars = len(parameterName[model])

    for workerId in workerIds:

        for iCVgame in range(numGames):

            p0List, pList, negllhList, negllhFunReturnList, successList, messageList, methodList = emptyLists(7)

            for iRep in range(numRep):

                fileNameFitResultsRep = getFittingFileName(ifCV=True, iCVgame=iCVgame, iRep=iRep, model=model, ifFlatPrior=ifFlatPrior,
                                                        workerId=workerId, fittingAlgorithm=fittingAlgorithm)
                
                if os.path.exists('fittingResultsCV/' + fileNameFitResultsRep):
                    
                    resultsRep = pd.read_csv('fittingResultsCV/' + fileNameFitResultsRep)

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
            fileNameFitResults = getFittingFileName(ifCV=True, iCVgame=iCVgame, iRep=None, model=model, ifFlatPrior=ifFlatPrior,
                                                    workerId=workerId, fittingAlgorithm=fittingAlgorithm)
            pd.DataFrame(resultsDict).to_csv('fittingResultsCV/' + fileNameFitResults, index=False)


# Evaluate on test set

if action == 'test':

    model = sys.argv[3]

    numPars = len(parameterName[model])

    iWorker = int(sys.argv[4])
    workerId = workerIds[iWorker]

    # Get the data
    dataWorker = data[data['workerId'] == workerId].copy()

    resultsCollectWorker = np.zeros((numGames, numPars+4))
    for iCVgame in range(numGames):
        # choose the game for testing
        gameTest = iCVgame + 1
        dataWorkerTest = dataWorker[dataWorker['game'] == gameTest]

        # get the fitting results file name
        fileNameFitResults = getFittingFileName(ifCV=True, iCVgame=iCVgame, iRep=None, model=model, ifFlatPrior=ifFlatPrior,
                                                workerId=workerId, fittingAlgorithm=fittingAlgorithm)
        resultsFitting = pd.read_csv('fittingResultsCV/' + fileNameFitResults)

        # find the best fit parameters
        minIdx = resultsFitting.loc[(resultsFitting['success'] == True) & (np.abs(resultsFitting['negllh']) > 1) & (np.abs(resultsFitting['negllh'] - resultsFitting['negllhFunReturn'])<sys.float_info.epsilon), 'negllh'].idxmin()
        pFit = [resultsFitting.loc[minIdx, 'pFit_' + parameterName[model][iPar]] for iPar in range(numPars)]
        negllh = resultsFitting.loc[minIdx, 'negllh']

        # calculate negllh on test data
        if ('Bayesian' not in model) and ('inferSerialHypoTesting' not in model):
            dataTest = prepDataFunction[model](dataWorkerTest)
            negllhTest = likelihoodFunction[model](pFit, dataTest, realdata=True)
        elif 'Bayesian' in model:
            dataTest = prepDataFunction[model](dataWorkerTest, ifFlatPrior=ifFlatPrior)
            negllhTest = likelihoodFunction[model](pFit, dataTest, ifFlatPrior=ifFlatPrior, realdata=True)
        elif 'inferSerialHypoTesting' in model:
            dataTest = prepDataFunction[model](model, dataWorkerTest)
            negllhTest = likelihoodFunction[model](pFit, model, dataTest, realdata=True)
        
        resultsCollectWorker[iCVgame, :] = [iWorker] + [iCVgame + 1] + pFit + [negllh, negllhTest]
    
    fileNameCollectWorker = getFittingCollectFileName(model=model, expVersion=expVersion, ifFlatPrior=ifFlatPrior, fittingAlgorithm=fittingAlgorithm, CV='fit', workerId=workerId)
    resultsCollectWorkerDF = pd.DataFrame(resultsCollectWorker, columns=['workerId', 'testGame'] + parameterName[model] + ['negllh', 'negllhTest'])
    resultsCollectWorkerDF['workerId'] = workerId
    resultsCollectWorkerDF.to_csv('fittingResultsCollectCV/'+fileNameCollectWorker, index=False)


# Collect test results

if action == 'testCollect':

    model = sys.argv[3]

    resultsCollectAll = []
    totalNegllhTest = []
    for workerId in workerIds:
        fileNameCollectWorker = getFittingCollectFileName(model=model, expVersion=expVersion, ifFlatPrior=ifFlatPrior, fittingAlgorithm=fittingAlgorithm, CV='test', workerId=workerId)
        resultsCollectWorker = pd.read_csv('fittingResultsCollectCV/' + fileNameCollectWorker)
        resultsCollectAll.append(resultsCollectWorker)
        totalNegllhTest.append(np.sum(resultsCollectWorker['negllhTest']))
    
    fileNameCollectAll = getFittingCollectFileName(model=model, expVersion=expVersion, ifFlatPrior=ifFlatPrior, fittingAlgorithm=fittingAlgorithm, CV='test')
    pd.concat(resultsCollectAll, ignore_index=True).to_csv('fittingResultsCollectCV/'+fileNameCollectAll, index=False)
    fileNameTotal = getFittingCollectFileName(model=model, expVersion=expVersion, ifFlatPrior=ifFlatPrior, fittingAlgorithm=fittingAlgorithm, CV='total')
    pd.DataFrame(list(zip(workerIds, totalNegllhTest)), columns=['workerId', 'negllhTest']).to_csv('fittingResultsCollectCV/'+fileNameTotal, index=False)


# Evaluate and get trial by trial likelihood

if 'getTrialLik' in action:

    iWorker = int(sys.argv[3])
    workerId = workerIds[iWorker]
    dataWorker = data[data['workerId'] == workerId].copy()

    model = sys.argv[4]

    fileName = 'fittingResultsCVTrialLik/' + model + '_' + workerId + ('_allChoice' if 'allChoices' in action else '') + '.csv'
    print(fileName, flush=True)
    
    if ('!' not in action) and (os.path.exists(fileName)):
        print('file already exists...')
        sys.exit()

    # prepare data for fitting
    ifFlatPrior = None if 'Bayesian' not in model else False
    priorStr = '' if ifFlatPrior is None else ('_priorFlat' if ifFlatPrior else '_priorNormalizedForDim')

    if ('Bayesian' not in model) and ('inferSerialHypoTesting' not in model):
        dataFitting = prepDataFunction[model](dataWorker)
    elif 'Bayesian' in model:
        dataFitting = prepDataFunction[model](dataWorker, ifFlatPrior=ifFlatPrior)
    elif 'inferSerialHypoTesting' in model:
        dataFitting = prepDataFunction[model](model, dataWorker)

    # Get the file name for stored fitting results
    fileNameCollectWorker = getFittingCollectFileName(model=model, expVersion=expVersion, ifFlatPrior=ifFlatPrior, fittingAlgorithm=fittingAlgorithm, CV='test', workerId=workerId)
    results = pd.read_csv('fittingResultsCollectCV/' + fileNameCollectWorker)

    for iCVgame in range(numGames):
        # choose the game for testing
        gameTest = iCVgame + 1
        dataWorkerTest = dataWorker[dataWorker['game'] == gameTest]

        # get the fitted parameters
        pFit = [results.loc[(results['workerId'] == workerId) & (results['testGame'] == gameTest), parameterName[model][iPar]].values[0] for iPar in range(len(parameterName[model]))]
        
        # calculate negllh on test data
        returnPrediction = True if ('allChoices' in action) else False
        if ('Bayesian' not in model) and ('inferSerialHypoTesting' not in model):
            dataTest = prepDataFunction[model](dataWorkerTest)
            res = likelihoodFunction[model](pFit, dataTest, realdata=True, returnTrialLikelihood=True, returnPrediction=returnPrediction)
        elif 'Bayesian' in model:
            dataTest = prepDataFunction[model](dataWorkerTest, ifFlatPrior=ifFlatPrior)
            res = likelihoodFunction[model](pFit, dataTest, realdata=True, ifFlatPrior=ifFlatPrior, returnTrialLikelihood=True, returnPrediction=returnPrediction)
        elif 'inferSerialHypoTesting' in model:
            dataTest = prepDataFunction[model](model, dataWorkerTest)
            res = likelihoodFunction[model](pFit, model, dataTest, realdata=True, returnTrialLikelihood=True, returnPrediction=returnPrediction)

        if 'allChoices' in action:
            p_allChoices = res[1]
            llh = res[2]
            for iC in range(len(allChoices)):
                if 'p_allChoices' + str(iC) not in data.keys():
                    data['p_allChoices' + str(iC)] = None
                data.loc[(data['workerId'] == workerId) & (data['game'] == gameTest), 'p_allChoices' + str(iC)] = np.stack(p_allChoices)[:, iC]
        else:
            llh = res[1]
    
        data.loc[(data['workerId'] == workerId) & (data['game'] == gameTest), model + '_loglikelihood'] = llh

        data.to_csv(fileName, index=False)


if 'collectTrialLik' in action:

    model = sys.argv[3]

    dataThisModel = deepcopy(data)

    fileName = 'fittingResultsCVTrialLik/' + model + '_' + expVersion + '_allWorkers' + ('_allChoice' if 'allChoices' in action else '') + '.csv'

    if ('!' not in action) and (os.path.exists(fileName)):
        sys.exit()

    dataThisModel[model + '_loglikelihood'] = None
    
    for workerId in workerIds:
        datatmp = pd.read_csv('fittingResultsCVTrialLik/' + model + '_' + workerId + ('_allChoice' if 'allChoices' in action else '') + '.csv')
        dataThisModel.loc[dataThisModel['workerId'] == workerId, model + '_loglikelihood'] = datatmp.loc[datatmp['workerId'] == workerId, model + '_loglikelihood']
        if 'allChoices' in action:
            for iC in range(len(allChoices)):
                if 'p_allChoices' + str(iC) not in dataThisModel.keys():
                    dataThisModel['p_allChoices' + str(iC)] = None
                dataThisModel.loc[dataThisModel['workerId'] == workerId, 'p_allChoices' + str(iC)] = datatmp.loc[datatmp['workerId'] == workerId, 'p_allChoices' + str(iC)]
    dataThisModel.to_csv(fileName, index=False)
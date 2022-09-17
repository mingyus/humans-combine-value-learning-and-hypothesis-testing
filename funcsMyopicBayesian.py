import numpy as np
import pandas as pd
import scipy.stats

from taskSetting import *
from utilities import *

def pRewardMatrix(hypotheses, choices = allChoices, rewardSetting = rewardSetting):
    pRewardMat = np.zeros((len(hypotheses),len(choices)))
    for iH, h in enumerate(hypotheses):
        for iChoice, choice in enumerate(choices):
            stimuli, _ = choiceToStimuli(choice)
            pRewardMat[iH,iChoice] = np.mean([rewardSetting[np.sum(~np.isnan(h))-1][np.sum(np.equal(h, stimulus))] for stimulus in stimuli])
    return pRewardMat

def trialLikelihood(hypotheses, stimulus, reward, rewardSetting = rewardSetting):
    lik = np.zeros(len(hypotheses))
    for iH, h in enumerate(hypotheses):
        pReward = rewardSetting[np.sum(~np.isnan(h))-1][np.sum(np.equal(h, stimulus))]
        lik[iH] = pReward if reward == True else (1-pReward)
    return lik


def getModelSimuCSVFileName_myopicBayesian(beta, cost, weightReward=1, ifFlatPrior=False, epsilonGreedy=0, gameLength=30, numGamePerType=3, ifPosterior = False):
    priorStr = '_priorFlat' if ifFlatPrior else '_priorNormalizedForDim'
    if weightReward in [0,1]:
        rewardFunStr = ('_max' + ('R' if weightReward == 1 else 'IG'))
    else:
        rewardFunStr = ('_rewardW' + str(weightReward))
    betaStr = ('_hardmax' if beta == None else ('_softmaxBeta' + str(beta)))
    costStr = '_cost' + str(cost)
    epsilonStr = ('' if epsilonGreedy == 0 else ('_epsilon' + str(epsilonGreedy)))

    if ifPosterior:
        return 'myopicBayesian' + priorStr + rewardFunStr + betaStr + costStr + epsilonStr + '_gameLength' + str(gameLength) + '_numGamePerType' + str(numGamePerType) + '_P.csv'
    else:
        return 'myopicBayesian' + priorStr + rewardFunStr + betaStr + costStr + epsilonStr + '_gameLength' + str(gameLength) + '_numGamePerType' + str(numGamePerType) + '.csv'


def getModelSimuCSVFileName_myopicBayesianProbDistortion(beta, cost, alpha, weightReward=1, ifFlatPrior=False, epsilonGreedy=0, gameLength=30, numGamePerType=3, ifPosterior = False):
    priorStr = '_priorFlat' if ifFlatPrior else '_priorNormalizedForDim'
    if weightReward in [0,1]:
        rewardFunStr = ('_max' + ('R' if weightReward == 1 else 'IG'))
    else:
        rewardFunStr = ('_rewardW' + str(weightReward))
    betaStr = ('_hardmax' if beta == None else ('_softmaxBeta' + str(beta)))
    costStr = '_cost' + str(cost)
    alphaStr = '_alpha' + str(alpha)
    epsilonStr = ('' if epsilonGreedy == 0 else ('_epsilon' + str(epsilonGreedy)))

    if ifPosterior:
        return 'myopicBayesianProbDistortion' + priorStr + rewardFunStr + betaStr + costStr + alphaStr + epsilonStr + '_gameLength' + str(gameLength) + '_numGamePerType' + str(numGamePerType) + '_P.csv'
    else:
        return 'myopicBayesianProbDistortion' + priorStr + rewardFunStr + betaStr + costStr + alphaStr + epsilonStr + '_gameLength' + str(gameLength) + '_numGamePerType' + str(numGamePerType) + '.csv'


def getModelSimuCSVFileName_myopicBayesian_2beta(betaR, cost, betaIG, ifFlatPrior=False, epsilonGreedy=0, gameLength=30, numGamePerType=3, ifPosterior = False):
    priorStr = '_priorFlat' if ifFlatPrior else '_priorNormalizedForDim'
    # exceptions
    if ((betaR == 0) & (betaIG == 0)) | ((betaR is None) & (betaIG is None)):
        Warning('Please reparametrize your betas.')
        return
    if (betaIG == 0) | (betaR is None):
        rewardFunStr = '_maxR'
        betaStr = ('_hardmax' if betaR is None else ('_softmaxBetaR' + str(betaR)))
    elif (betaR == 0) | (betaIG is None):
        rewardFunStr = '_maxIG'
        betaStr = ('_hardmax' if betaIG is None else ('_softmaxBetaIG' + str(betaIG)))
    else:
        rewardFunStr = '_RandIG'
        betaStr = '_softmaxBetaR' + str(betaR) + 'BetaIG' + str(betaIG)
    costStr = '_cost' + str(cost)
    epsilonStr = ('' if epsilonGreedy == 0 else ('_epsilon' + str(epsilonGreedy)))

    if ifPosterior:
        return 'myopicBayesian' + priorStr + rewardFunStr + betaStr + costStr + epsilonStr + '_gameLength' + str(gameLength) + '_numGamePerType' + str(numGamePerType) + '_P.csv'
    else:
        return 'myopicBayesian' + priorStr + rewardFunStr + betaStr + costStr + epsilonStr + '_gameLength' + str(gameLength) + '_numGamePerType' + str(numGamePerType) + '.csv'



########## MODEL FITTING ##########

def prepForFitting_myopicBayesian(data, ifFlatPrior):

    data = data.reset_index(drop=True)

    dataFitting = data[['game','trial','informed','numRelevantDimensions','rt','reward']].copy()

    for iRow in range(data.shape[0]):

        if pd.isnull(dataFitting.loc[iRow,'rt']):
            dataFitting.loc[iRow, 'choiceIndex'] = np.nan
            dataFitting.loc[iRow, 'stimulusIndex'] = np.nan

        else:
            # code choice as index according to allChoices
            choice = [(np.nan if pd.isnull(data.loc[iRow,'selectedFeature_'+dim]) else
                       DIMENSIONS_TO_FEATURES[dim].index(data.loc[iRow, 'selectedFeature_' + dim])) for dim in DIMENSIONS]
            dataFitting.loc[iRow, 'choiceIndex'] = allChoices.index(choice)

            # code stimulus as index according to allStimuli
            stimulus = [(np.nan if pd.isnull(data.loc[iRow, 'builtFeature_' + dim]) else
                         DIMENSIONS_TO_FEATURES[dim].index(data.loc[iRow, 'builtFeature_' + dim])) for dim in DIMENSIONS]
            dataFitting.loc[iRow, 'stimulusIndex'] = allStimuli.index(stimulus)

    return dataFitting


def prepForFitting_myopicBayesianwIG(data, ifFlatPrior):

    data = data.reset_index(drop=True)

    dataFitting = data[['game','trial','informed','numRelevantDimensions','rt','reward']].copy()

    for iRow in range(data.shape[0]):

        if data.loc[iRow, 'trial'] == 1:
            # game settings
            if data.loc[iRow, 'informed']:
                hypotheses = hypothesisSpace[data.loc[iRow, 'numRelevantDimensions'] - 1]
            else:
                hypotheses = [val for sublist in hypothesisSpace for val in sublist]
            numHypotheses = len(hypotheses)

            pRewardMat = pRewardMatrix(hypotheses)  # a matrix of p(reward): #hypothesis * #all stimuli

            # the true hypothesis for the game
            trueHypothesis = np.zeros(len(DIMENSIONS))
            for iDim, dim in enumerate(DIMENSIONS):
                trueHypothesis[iDim] = np.nan if pd.isnull(data.loc[iRow, 'rewardingFeature_' + dim]) else \
                    DIMENSIONS_TO_FEATURES[dim].index(data.loc[iRow, 'rewardingFeature_' + dim])

            # initialize prior
            if ifFlatPrior:
                p = np.ones(numHypotheses) / numHypotheses
            else:
                p = (np.ones(numHypotheses) / numHypotheses) if data.loc[iRow, 'informed'] else (
                        np.ones(numHypotheses) / numDimensions / flatten2Dlist(
                    ([[len(hypothesisSpace[i])] * len(hypothesisSpace[i]) for i in np.arange(numDimensions)])))

        if pd.isnull(data.loc[iRow, 'rt']):
            dataFitting.loc[iRow, 'choiceIndex'] = np.nan
            dataFitting.loc[iRow, 'stimulusIndex'] = np.nan
            for iChoice, choice in enumerate(allChoices):
                dataFitting.loc[iRow, 'ER' + str(iChoice)] = np.nan
                dataFitting.loc[iRow, 'IG' + str(iChoice)] = np.nan

        else:
            # code choice as index according to allChoices
            choice = [(np.nan if pd.isnull(data.loc[iRow, 'selectedFeature_' + dim]) else
                       DIMENSIONS_TO_FEATURES[dim].index(data.loc[iRow, 'selectedFeature_' + dim])) for dim in
                      DIMENSIONS]
            dataFitting.loc[iRow, 'choiceIndex'] = allChoices.index(choice)

            # code stimulus as index according to allStimuli
            stimulus = [DIMENSIONS_TO_FEATURES[dim].index(data.loc[iRow, 'builtFeature_' + dim]) for dim in DIMENSIONS]
            dataFitting.loc[iRow, 'stimulusIndex'] = allStimuli.index(stimulus)

            # calculate expected reward (before considering cost) and information gain for all the choices
            expectedR = np.sum(pRewardMat.T * p, axis=1) # - cost * (numDimensions - np.sum(np.isnan(allChoices), axis=1))
            IG = informationGain(p, hypotheses)

            # save ER and IG for allChoices
            for iChoice, choice in enumerate(allChoices):
                dataFitting.loc[iRow, 'ER' + str(iChoice)] = expectedR[iChoice]
                dataFitting.loc[iRow, 'IG' + str(iChoice)] = IG[iChoice]

            # Bayesian update
            p = trialLikelihood(hypotheses, stimulus, data.loc[iRow, 'reward']) * p
            p = p / np.sum(p)

    return dataFitting


def likelihood_myopicBayesian(pars, dataFitting, ifFlatPrior, realdata=True, returnPrediction=False, returnTrialLikelihood=False):

    # parameters
    beta, cost = pars[0], pars[1]
    if len(pars) == 2:
        thisRewardSetting = rewardSetting
    else:
        alpha = pars[2]
        thisRewardSetting = [
            [np.exp(-(np.power(-np.log(rewardSetting[numRelevantDimensions][numRewardingFeatures]), alpha))) for
             numRewardingFeatures in range(numRelevantDimensions + 2)] for numRelevantDimensions in
            np.arange(len(DIMENSIONS))]

    # experiment info
    numGame = np.max(dataFitting['game'].values)
    gameLength = np.max(dataFitting['trial'].values)
    llh = np.zeros(dataFitting.shape[0])
    if returnPrediction:
        samplePList = []
        
    iRow = 0

    for iGame in dataFitting['game'].unique():

        # game data
        dataThisGame = dataFitting[dataFitting['game'] == iGame].reset_index(drop=True)
        rt = dataThisGame['rt'].values
        choiceIndex = dataThisGame['choiceIndex'].values
        stimulusIndex = dataThisGame['stimulusIndex'].values
        reward = dataThisGame['reward'].values

        # game info
        informed = dataThisGame.loc[0, 'informed']
        numRelevantDimensions = dataThisGame.loc[0, 'numRelevantDimensions']
        if informed:
            hypotheses = hypothesisSpace[numRelevantDimensions - 1]
        else:
            hypotheses = [val for sublist in hypothesisSpace for val in sublist]
        numHypotheses = len(hypotheses)

        pRewardMat = pRewardMatrix(hypotheses)  # a matrix of p(reward): #hypothesis * #all stimuli

        # prior
        if ifFlatPrior:
            p = np.ones(numHypotheses) / numHypotheses
        else:
            p = (np.ones(numHypotheses) / numHypotheses) if informed else (
                    np.ones(numHypotheses) / numDimensions / flatten2Dlist(
                ([[len(hypothesisSpace[i])] * len(hypothesisSpace[i]) for i in np.arange(numDimensions)])))

        for iTrial in range(gameLength):

            if (realdata == True) & np.isnan(rt[iTrial]):
                if returnPrediction:
                    samplePList.append([np.nan]*len(allChoices))
                llh[iRow] = np.nan

            else:
                # choice phase
                expectedR = np.sum(pRewardMat.T * p, axis=1) - cost * (numDimensions - np.sum(np.isnan(allChoices), axis=1))

                sampleP = np.exp(beta * expectedR) / np.sum(np.exp(beta * expectedR))
                if returnPrediction:
                    samplePList.append(sampleP)
                llh[iRow] = np.log(sampleP[int(choiceIndex[iTrial])])

                # learning phase: Bayesian update
                p = trialLikelihood(hypotheses, stimulus=allStimuli[int(stimulusIndex[iTrial])], reward=reward[iTrial], rewardSetting=thisRewardSetting) * p
                p = p/np.sum(p)

            iRow += 1

    if not (returnPrediction | returnTrialLikelihood):
        results = -np.nansum(llh)
    else:
        results = []
        results.append(-np.nansum(llh))
        if returnPrediction:
            results.append(samplePList)
        if returnTrialLikelihood:
            results.append(llh)
    return results


def likelihood_myopicBayesianNoCost(pars, dataFitting, ifFlatPrior, realdata=True, returnPrediction=False, returnTrialLikelihood=False):

    # parameters
    beta, cost = pars[0], 0
    if len(pars) <= 2:
        thisRewardSetting = rewardSetting
    else:
        alpha = pars[2]
        thisRewardSetting = [
            [np.exp(-(np.power(-np.log(rewardSetting[numRelevantDimensions][numRewardingFeatures]), alpha))) for
             numRewardingFeatures in range(numRelevantDimensions + 2)] for numRelevantDimensions in
            np.arange(len(DIMENSIONS))]

    # experiment info
    numGame = np.max(dataFitting['game'].values)
    gameLength = np.max(dataFitting['trial'].values)
    llh = np.zeros(dataFitting.shape[0])
    if returnPrediction:
        samplePList = []
        
    iRow = 0

    for iGame in dataFitting['game'].unique():

        # game data
        dataThisGame = dataFitting[dataFitting['game'] == iGame].reset_index(drop=True)
        rt = dataThisGame['rt'].values
        choiceIndex = dataThisGame['choiceIndex'].values
        stimulusIndex = dataThisGame['stimulusIndex'].values
        reward = dataThisGame['reward'].values

        # game info
        informed = dataThisGame.loc[0, 'informed']
        numRelevantDimensions = dataThisGame.loc[0, 'numRelevantDimensions']
        if informed:
            hypotheses = hypothesisSpace[numRelevantDimensions - 1]
        else:
            hypotheses = [val for sublist in hypothesisSpace for val in sublist]
        numHypotheses = len(hypotheses)

        pRewardMat = pRewardMatrix(hypotheses)  # a matrix of p(reward): #hypothesis * #all stimuli

        # prior
        if ifFlatPrior:
            p = np.ones(numHypotheses) / numHypotheses
        else:
            p = (np.ones(numHypotheses) / numHypotheses) if informed else (
                    np.ones(numHypotheses) / numDimensions / flatten2Dlist(
                ([[len(hypothesisSpace[i])] * len(hypothesisSpace[i]) for i in np.arange(numDimensions)])))

        for iTrial in range(gameLength):

            if (realdata == True) & np.isnan(rt[iTrial]):
                if returnPrediction:
                    samplePList.append([np.nan]*len(allChoices))
                llh[iRow] = np.nan

            else:
                # choice phase
                expectedR = np.sum(pRewardMat.T * p, axis=1) - cost * (numDimensions - np.sum(np.isnan(allChoices), axis=1))

                sampleP = np.exp(beta * expectedR) / np.sum(np.exp(beta * expectedR))
                if returnPrediction:
                    samplePList.append(sampleP)
                llh[iRow] = np.log(sampleP[int(choiceIndex[iTrial])])

                # learning phase: Bayesian update
                p = trialLikelihood(hypotheses, stimulus=allStimuli[int(stimulusIndex[iTrial])], reward=reward[iTrial], rewardSetting=thisRewardSetting) * p
                p = p/np.sum(p)

            iRow += 1

    if not (returnPrediction | returnTrialLikelihood):
        results = -np.nansum(llh)
    else:
        results = []
        results.append(-np.nansum(llh))
        if returnPrediction:
            results.append(samplePList)
        if returnTrialLikelihood:
            results.append(llh)
    return results


def likelihood_myopicBayesianwIG(pars, dataFitting, ifFlatPrior, realdata=True, returnPrediction=False, returnTrialLikelihood=False):

    # parameters
    beta, cost, weightReward = pars[0], pars[1], pars[2]

    # experiment info
    numGame = np.max(dataFitting['game'].values)
    gameLength = np.max(dataFitting['trial'].values)
    llh = np.zeros(dataFitting.shape[0])
    if returnPrediction:
        samplePList = []
    
    iRow = 0

    for iGame in dataFitting['game'].unique():

        # game data
        dataThisGame = dataFitting[dataFitting['game'] == iGame].reset_index(drop=True)
        rt = dataThisGame['rt'].values
        choiceIndex = dataThisGame['choiceIndex'].values
        stimulusIndex = dataThisGame['stimulusIndex'].values
        reward = dataThisGame['reward'].values

        # game info
        informed = dataThisGame.loc[0, 'informed']
        numRelevantDimensions = dataThisGame.loc[0, 'numRelevantDimensions']
        if informed:
            hypotheses = hypothesisSpace[numRelevantDimensions - 1]
        else:
            hypotheses = [val for sublist in hypothesisSpace for val in sublist]
        numHypotheses = len(hypotheses)

        ER = np.zeros((gameLength, len(allChoices)))
        IG = np.zeros((gameLength, len(allChoices)))
        for iChoice, choice in enumerate(allChoices):
            ER[:, iChoice] = dataThisGame['ER' + str(iChoice)]
            IG[:, iChoice] = dataThisGame['IG' + str(iChoice)]

        # prior
        if ifFlatPrior:
            p = np.ones(numHypotheses) / numHypotheses
        else:
            p = (np.ones(numHypotheses) / numHypotheses) if informed else (
                    np.ones(numHypotheses) / numDimensions / flatten2Dlist(
                ([[len(hypothesisSpace[i])] * len(hypothesisSpace[i]) for i in np.arange(numDimensions)])))

        for iTrial in range(gameLength):

            if (realdata == True) & np.isnan(rt[iTrial]):
                if returnPrediction:
                    samplePList.append([np.nan]*len(allChoices))
                llh[iRow] = np.nan

            else:
                # choice phase
                DV = (ER[iTrial, :] - cost * (numDimensions - np.sum(np.isnan(allChoices), axis=1))) * weightReward + IG[iTrial, :] * (1 - weightReward)
                sampleP = np.exp(beta * DV) / np.sum(np.exp(beta * DV))
                if returnPrediction:
                    samplePList.append(sampleP)
                llh[iRow] = np.log(sampleP[int(choiceIndex[iTrial])])

                # learning phase: Bayesian update
                p = trialLikelihood(hypotheses, stimulus=allStimuli[int(stimulusIndex[iTrial])], reward=reward[iTrial]) * p
                p = p/np.sum(p)

            iRow += 1

    if not (returnPrediction | returnTrialLikelihood):
        results = -np.nansum(llh)
    else:
        results = []
        results.append(-np.nansum(llh))
        if returnPrediction:
            results.append(samplePList)
        if returnTrialLikelihood:
            results.append(llh)
    return results


def likelihood_myopicBayesianwIG_2beta(pars, dataFitting, ifFlatPrior, realdata=True, returnPrediction=False, returnTrialLikelihood=False):

    # parameters
    betaR, cost, betaIG = pars[0], pars[1], pars[2]

    # experiment info
    numGame = np.max(dataFitting['game'].values)
    gameLength = np.max(dataFitting['trial'].values)
    llh = np.zeros(dataFitting.shape[0])
    if returnPrediction:
        samplePList = []
        
    iRow = 0

    for iGame in dataFitting['game'].unique():

        # game data
        dataThisGame = dataFitting[dataFitting['game'] == iGame].reset_index(drop=True)
        rt = dataThisGame['rt'].values
        choiceIndex = dataThisGame['choiceIndex'].values
        stimulusIndex = dataThisGame['stimulusIndex'].values
        reward = dataThisGame['reward'].values

        # game info
        informed = dataThisGame.loc[0, 'informed']
        numRelevantDimensions = dataThisGame.loc[0, 'numRelevantDimensions']
        if informed:
            hypotheses = hypothesisSpace[numRelevantDimensions - 1]
        else:
            hypotheses = [val for sublist in hypothesisSpace for val in sublist]
        numHypotheses = len(hypotheses)

        ER = np.zeros((gameLength, len(allChoices)))
        IG = np.zeros((gameLength, len(allChoices)))
        for iChoice, choice in enumerate(allChoices):
            ER[:, iChoice] = dataThisGame['ER' + str(iChoice)]
            IG[:, iChoice] = dataThisGame['IG' + str(iChoice)]

        # prior
        if ifFlatPrior:
            p = np.ones(numHypotheses) / numHypotheses
        else:
            p = (np.ones(numHypotheses) / numHypotheses) if informed else (
                    np.ones(numHypotheses) / numDimensions / flatten2Dlist(
                ([[len(hypothesisSpace[i])] * len(hypothesisSpace[i]) for i in np.arange(numDimensions)])))

        for iTrial in range(gameLength):

            if (realdata == True) & np.isnan(rt[iTrial]):
                if returnPrediction:
                    samplePList.append([np.nan]*len(allChoices))
                llh[iRow] = np.nan

            else:
                # choice phase
                DV = betaR * (ER[iTrial, :] - cost * (numDimensions - np.sum(np.isnan(allChoices), axis=1))) + betaIG * IG[iTrial, :]
                sampleP = np.exp(DV) / np.sum(np.exp(DV))
                if returnPrediction:
                    samplePList.append(sampleP)
                llh[iRow] = np.log(sampleP[int(choiceIndex[iTrial])])

                # learning phase: Bayesian update
                p = trialLikelihood(hypotheses, stimulus=allStimuli[int(stimulusIndex[iTrial])], reward=reward[iTrial]) * p
                p = p/np.sum(p)

            iRow += 1

    if not (returnPrediction | returnTrialLikelihood):
        results = -np.nansum(llh)
    else:
        results = []
        results.append(-np.nansum(llh))
        if returnPrediction:
            results.append(samplePList)
        if returnTrialLikelihood:
            results.append(llh)
    return results


########## SIMULATION ##########

def model_myopicBayesian(modelName, pars, ifFlatPrior, gameLength, numGamePerType, ifSaveCSV):

    # default parameters
    epsilonGreedy = 0  # default: 0

    # parameters
    if modelName == 'myopicBayesian':
        beta, cost = pars[0], pars[1]
        weightReward = 1
        thisRewardSetting = rewardSetting
    elif modelName == 'myopicBayesianNoCost':
        beta, cost = pars[0], 0
        weightReward = 1
        thisRewardSetting = rewardSetting
    elif modelName == 'myopicBayesianProbDistortion':
        beta, cost, alpha = pars[0], pars[1], pars[2]
        weightReward = 1
        thisRewardSetting = [
            [np.exp(-(np.power(-np.log(rewardSetting[numRelevantDimensions][numRewardingFeatures]), alpha))) for
             numRewardingFeatures in range(numRelevantDimensions + 2)] for numRelevantDimensions in
            np.arange(len(DIMENSIONS))]
    elif modelName == 'myopicBayesianwIG':
        beta, cost, weightReward = pars[0], pars[1], pars[2]
        thisRewardSetting = rewardSetting

    # create the variables
    game, trial, informedList, numRelevantDimensionsList, reward, numSelectedFeatures, rt = \
        zerosLists(numList=7, lengthList=gameLength*numGamePerType*6)
    ifRelevantDimension, rewardingFeature, selectedFeature, randomlySelectedFeature, builtFeature = \
        emptyDicts(numDict=5, keys=DIMENSIONS, lengthList=gameLength*numGamePerType*6)
    # belief = np.full((gameLength*numGamePerType*6, (len(DIMENSIONS_TO_FEATURES[DIMENSIONS[0]])+1)**len(DIMENSIONS)), np.nan)

    # simulation
    iRow = 0
    iGame = 0
    for informed in [True, False]:
        for numRelevantDimensions in np.arange(3) + 1:

            print(informed, numRelevantDimensions, flush=True)

            # game settings that only depend on informed and numRelevantDimensions
            if informed:
                hypotheses = hypothesisSpace[numRelevantDimensions - 1]
            else:
                hypotheses = [val for sublist in hypothesisSpace for val in sublist]
            numHypotheses = len(hypotheses)

            numChoices = len(allChoices)

            pRewardMat = pRewardMatrix(hypotheses)  # a matrix of p(reward): #hypothesis * #all stimuli

            # save game info
            informedList[iRow:(iRow + gameLength * numGamePerType)] = [informed] * gameLength * numGamePerType
            numRelevantDimensionsList[iRow:(iRow + gameLength * numGamePerType)] = [numRelevantDimensions] * gameLength * numGamePerType

            for iRepeat in range(numGamePerType):
                # generate and save game-specific (reward) setting
                game[iRow:(iRow + gameLength)] = [iGame + 1] * gameLength
                relevantDimensions = np.random.choice(DIMENSIONS, size=numRelevantDimensions, replace=False)
                for dim in DIMENSIONS:
                    if dim in relevantDimensions:
                        ifRelevantDimension[dim][iRow:(iRow + gameLength)] = [True] * gameLength
                        rewardingFeature[dim][iRow:(iRow + gameLength)] = [np.random.choice(DIMENSIONS_TO_FEATURES[dim],
                                                                                            size=1)[0]] * gameLength
                    else:
                        ifRelevantDimension[dim][iRow:(iRow + gameLength)] = [False] * gameLength
                        rewardingFeature[dim][iRow:(iRow + gameLength)] = [np.nan] * gameLength

                # simulation
                if ifFlatPrior:
                    p = np.ones(numHypotheses) / numHypotheses
                else:
                    p = (np.ones(numHypotheses) / numHypotheses) if informed else (
                                np.ones(numHypotheses) / numDimensions / flatten2Dlist(
                            ([[len(hypothesisSpace[i])] * len(hypothesisSpace[i]) for i in np.arange(numDimensions)])))

                for iTrial in range(gameLength):

                    trial[iRow] = iTrial + 1

                    # choice phase (hard max; if more than one hypothesis have the highest posterior, pick randomly)
                    expectedR = np.sum(pRewardMat.T * p, axis=1) - cost * (
                                numDimensions - np.sum(np.isnan(allChoices), axis=1))

                    if weightReward == 1:
                        DV = expectedR
                    else:
                        IG = informationGain(p, hypotheses)
                        DV = expectedR * weightReward + IG * (1 - weightReward)

                    sampleP = np.zeros(numChoices)
                    if beta is None:  # hard max
                        sampleP[np.where(np.isclose(DV, np.max(DV)))[0]] = 1 / (
                            np.where(np.isclose(DV, np.max(DV)))[0].size)
                    else:  # softmax
                        sampleP = np.exp(beta * DV) / np.sum(np.exp(beta * DV))
                    sampleP = sampleP * (1 - epsilonGreedy) + epsilonGreedy / numChoices
                    choice = allChoices[np.random.choice(np.arange(numChoices), size=1, p=sampleP)[0]]

                    # generate stimulus and reward outcome
                    stimulus = deepcopy(choice)
                    for iDim, dim in enumerate(DIMENSIONS):
                        if (~np.isnan(choice[iDim])):
                            selectedFeature[dim][iRow] = DIMENSIONS_TO_FEATURES[dim][choice[iDim]]
                            randomlySelectedFeature[dim][iRow] = np.nan
                            builtFeature[dim][iRow] = selectedFeature[dim][iRow]
                        else:
                            selectedFeature[dim][iRow] = np.nan
                            stimulus[iDim] = np.random.choice(np.arange(len(DIMENSIONS_TO_FEATURES[dim])), 1)[0]
                            randomlySelectedFeature[dim][iRow] = DIMENSIONS_TO_FEATURES[dim][stimulus[iDim]]
                            builtFeature[dim][iRow] = randomlySelectedFeature[dim][iRow]
                    numSelectedFeatures[iRow] = np.array(
                        [(not pd.isnull(selectedFeature[dim][iRow])) for dim in DIMENSIONS]).sum()
                    numRewardingFeatureBuilt = np.array([((not pd.isnull(rewardingFeature[dim][iRow])) & \
                                                          (builtFeature[dim][iRow] == rewardingFeature[dim][iRow])) for
                                                         dim in DIMENSIONS]).sum()
                    reward[iRow] = (
                                np.random.random() < thisRewardSetting[numRelevantDimensions - 1][numRewardingFeatureBuilt])

                    # learning phase: Bayesian update
                    p = trialLikelihood(hypotheses, stimulus, reward[iRow], rewardSetting=thisRewardSetting) * p
                    p = p / np.sum(p)
                    # belief[iRow, :len(p)] = p

                    iRow += 1

                iGame += 1

    # save variables into dataframe
    simudata = pd.DataFrame(
        {'game': game, 'trial': trial, 'informed': informedList, 'numRelevantDimensions': numRelevantDimensionsList,
         'reward': reward, 'numSelectedFeatures': numSelectedFeatures, 'rt': rt,
         'ifRelevantDimension_color': ifRelevantDimension['color'], 'rewardingFeature_color': rewardingFeature['color'],
         'selectedFeature_color': selectedFeature['color'],
         'randomlySelectedFeature_color': randomlySelectedFeature['color'], 'builtFeature_color': builtFeature['color'],
         'ifRelevantDimension_shape': ifRelevantDimension['shape'], 'rewardingFeature_shape': rewardingFeature['shape'],
         'selectedFeature_shape': selectedFeature['shape'],
         'randomlySelectedFeature_shape': randomlySelectedFeature['shape'], 'builtFeature_shape': builtFeature['shape'],
         'ifRelevantDimension_pattern': ifRelevantDimension['pattern'],
         'rewardingFeature_pattern': rewardingFeature['pattern'], 'selectedFeature_pattern': selectedFeature['pattern'],
         'randomlySelectedFeature_pattern': randomlySelectedFeature['pattern'],
         'builtFeature_pattern': builtFeature['pattern']})
    # beliefDF = pd.DataFrame(belief, columns=['p'+str(i) for i in range(belief.shape[1])])
    # simudata = pd.concat([simudata, beliefDF], axis=1)

    if ifSaveCSV:
        simudata.to_csv('modelSimulation/' + getModelSimuCSVFileName_myopicBayesian(beta=beta, cost=cost, weightReward=weightReward,
                                                      ifFlatPrior=ifFlatPrior, epsilonGreedy=epsilonGreedy,
                                                      gameLength=gameLength, numGamePerType=numGamePerType), index=False)

    return simudata


def model_myopicBayesian_2beta(modelName, pars, ifFlatPrior, gameLength, numGamePerType, ifSaveCSV):

    # default parameters
    epsilonGreedy = 0  # default: 0

    # parameters
    if len(pars) == 2:
        betaR, cost = pars[0], pars[1]
        betaIG = 0
    else:
        betaR, cost, betaIG = pars[0], pars[1], pars[2]

    # create the variables
    game, trial, informedList, numRelevantDimensionsList, reward, numSelectedFeatures, rt = \
        zerosLists(numList=7, lengthList=gameLength*numGamePerType*6)
    ifRelevantDimension, rewardingFeature, selectedFeature, randomlySelectedFeature, builtFeature = \
        emptyDicts(numDict=5, keys=DIMENSIONS, lengthList=gameLength*numGamePerType*6)

    # simulation
    iRow = 0
    iGame = 0
    for informed in [True, False]:
        for numRelevantDimensions in np.arange(3) + 1:

            # game settings that only depend on informed and numRelevantDimensions
            if informed:
                hypotheses = hypothesisSpace[numRelevantDimensions - 1]
            else:
                hypotheses = [val for sublist in hypothesisSpace for val in sublist]
            numHypotheses = len(hypotheses)

            numChoices = len(allChoices)

            pRewardMat = pRewardMatrix(hypotheses)  # a matrix of p(reward): #hypothesis * #all stimuli

            # save game info
            informedList[iRow:(iRow + gameLength * numGamePerType)] = [informed] * gameLength * numGamePerType
            numRelevantDimensionsList[iRow:(iRow + gameLength * numGamePerType)] = [numRelevantDimensions] * gameLength * numGamePerType

            for iRepeat in range(numGamePerType):
                # generate and save game-specific (reward) setting
                game[iRow:(iRow + gameLength)] = [iGame + 1] * gameLength
                relevantDimensions = np.random.choice(DIMENSIONS, size=numRelevantDimensions, replace=False)
                for dim in DIMENSIONS:
                    if dim in relevantDimensions:
                        ifRelevantDimension[dim][iRow:(iRow + gameLength)] = [True] * gameLength
                        rewardingFeature[dim][iRow:(iRow + gameLength)] = [np.random.choice(DIMENSIONS_TO_FEATURES[dim],
                                                                                            size=1)[0]] * gameLength
                    else:
                        ifRelevantDimension[dim][iRow:(iRow + gameLength)] = [False] * gameLength
                        rewardingFeature[dim][iRow:(iRow + gameLength)] = [np.nan] * gameLength

                # simulation
                if ifFlatPrior:
                    p = np.ones(numHypotheses) / numHypotheses
                else:
                    p = (np.ones(numHypotheses) / numHypotheses) if informed else (
                                np.ones(numHypotheses) / numDimensions / flatten2Dlist(
                            ([[len(hypothesisSpace[i])] * len(hypothesisSpace[i]) for i in np.arange(numDimensions)])))

                for iTrial in range(gameLength):

                    trial[iRow] = iTrial + 1

                    sampleP = np.zeros(numChoices)

                    # choice phase (hard max; if more than one hypothesis have the highest posterior, pick randomly)
                    expectedR = np.sum(pRewardMat.T * p, axis=1) - cost * (
                                numDimensions - np.sum(np.isnan(allChoices), axis=1))

                    if betaIG != 0:
                        IG = informationGain(p, hypotheses)

                    # if (betaR is None) | (betaIG == 0):
                    #     DV = expectedR
                    # else:
                    #     IG = informationGain(p, hypotheses)
                    #     DV = betaR * expectedR + betaIG * IG

                    if betaR is None:  # hard max for R
                        sampleP[np.where(np.isclose(expectedR, np.max(expectedR)))[0]] = \
                            1 / (np.where(np.isclose(expectedR, np.max(expectedR)))[0].size)
                    elif betaIG is None:  # hard max for IG
                        sampleP[np.where(np.isclose(IG, np.max(IG)))[0]] = 1 / (np.where(np.isclose(IG, np.max(IG)))[0].size)
                    else:  # softmax
                        DV = betaR * expectedR + betaIG * IG
                        sampleP = np.exp(DV) / np.sum(np.exp(DV))
                    sampleP = sampleP * (1 - epsilonGreedy) + epsilonGreedy / numChoices
                    choice = allChoices[np.random.choice(np.arange(numChoices), size=1, p=sampleP)[0]]

                    # generate stimulus and reward outcome
                    stimulus = deepcopy(choice)
                    for iDim, dim in enumerate(DIMENSIONS):
                        if (~np.isnan(choice[iDim])):
                            selectedFeature[dim][iRow] = DIMENSIONS_TO_FEATURES[dim][choice[iDim]]
                            randomlySelectedFeature[dim][iRow] = np.nan
                            builtFeature[dim][iRow] = selectedFeature[dim][iRow]
                        else:
                            selectedFeature[dim][iRow] = np.nan
                            stimulus[iDim] = np.random.choice(np.arange(len(DIMENSIONS_TO_FEATURES[dim])), 1)[0]
                            randomlySelectedFeature[dim][iRow] = DIMENSIONS_TO_FEATURES[dim][stimulus[iDim]]
                            builtFeature[dim][iRow] = randomlySelectedFeature[dim][iRow]
                    numSelectedFeatures[iRow] = np.array(
                        [(not pd.isnull(selectedFeature[dim][iRow])) for dim in DIMENSIONS]).sum()
                    numRewardingFeatureBuilt = np.array([((not pd.isnull(rewardingFeature[dim][iRow])) & \
                                                          (builtFeature[dim][iRow] == rewardingFeature[dim][iRow])) for
                                                         dim in DIMENSIONS]).sum()
                    reward[iRow] = (
                                np.random.random() < rewardSetting[numRelevantDimensions - 1][numRewardingFeatureBuilt])

                    # learning phase: Bayesian update
                    p = trialLikelihood(hypotheses, stimulus, reward[iRow]) * p
                    p = p / np.sum(p)

                    iRow += 1

                iGame += 1

    # save variables into dataframe
    simudata = pd.DataFrame(
        {'game': game, 'trial': trial, 'informed': informedList, 'numRelevantDimensions': numRelevantDimensionsList,
         'reward': reward, 'numSelectedFeatures': numSelectedFeatures, 'rt': rt,
         'ifRelevantDimension_color': ifRelevantDimension['color'], 'rewardingFeature_color': rewardingFeature['color'],
         'selectedFeature_color': selectedFeature['color'],
         'randomlySelectedFeature_color': randomlySelectedFeature['color'], 'builtFeature_color': builtFeature['color'],
         'ifRelevantDimension_shape': ifRelevantDimension['shape'], 'rewardingFeature_shape': rewardingFeature['shape'],
         'selectedFeature_shape': selectedFeature['shape'],
         'randomlySelectedFeature_shape': randomlySelectedFeature['shape'], 'builtFeature_shape': builtFeature['shape'],
         'ifRelevantDimension_pattern': ifRelevantDimension['pattern'],
         'rewardingFeature_pattern': rewardingFeature['pattern'], 'selectedFeature_pattern': selectedFeature['pattern'],
         'randomlySelectedFeature_pattern': randomlySelectedFeature['pattern'],
         'builtFeature_pattern': builtFeature['pattern']})

    if ifSaveCSV:
        simudata.to_csv('modelSimulation/' + getModelSimuCSVFileName_myopicBayesian_2beta(betaR, cost, betaIG, ifFlatPrior, epsilonGreedy, gameLength, numGamePerType), index=False)

    return simudata

### Uncertainty measures

# expected information gain of a choice
def informationGain(p, hypotheses, choices = allChoices, allStimuli = allStimuli, rewardSetting = rewardSetting):
    KLList = np.zeros(len(allStimuli))
    for iSti, stimulus in enumerate(allStimuli):
        pReward = np.mean([rewardSetting[np.sum(~np.isnan(h))-1][np.sum(np.equal(h, stimulus))] for h in hypotheses])
        posteriorRewarded = trialLikelihood(hypotheses, stimulus, reward = True)*p
        KLRewarded = scipy.stats.entropy(posteriorRewarded/np.sum(posteriorRewarded), p)
        posteriorUnrewarded = trialLikelihood(hypotheses, stimulus, reward = False)*p
        KLUnrewarded = scipy.stats.entropy(posteriorUnrewarded/np.sum(posteriorUnrewarded), p)
        KLList[iSti] = KLRewarded*pReward + KLUnrewarded*(1-pReward)
    IG = np.zeros(len(choices))
    for iChoice, choice in enumerate(choices):
        _, idxStimuli = choiceToStimuli(choice)
        IG[iChoice] = np.mean(KLList[idxStimuli])
    return IG

# expected entropy of the posterior
def posteriorEntropy(p, hypotheses, choices = allChoices, allStimuli = allStimuli, rewardSetting = rewardSetting):
    entropyList = np.zeros(len(allStimuli))
    for iSti, stimulus in enumerate(allStimuli):
        pReward = np.mean([rewardSetting[np.sum(~np.isnan(h))-1][np.sum(np.equal(h, stimulus))] for h in hypotheses])
        posteriorRewarded = trialLikelihood(hypotheses, stimulus, reward = True)*p
        entropyRewarded = scipy.stats.entropy(posteriorRewarded/np.sum(posteriorRewarded))
        posteriorUnrewarded = trialLikelihood(hypotheses, stimulus, reward = False)*p
        entropyUnrewarded = scipy.stats.entropy(posteriorUnrewarded/np.sum(posteriorUnrewarded))
        entropyList[iSti] = entropyRewarded*pReward + entropyUnrewarded*(1-pReward)
    entropy = np.zeros(len(choices))
    for iChoice, choice in enumerate(choices):
        _, idxStimuli = choiceToStimuli(choice)
        entropy[iChoice] = np.mean(entropyList[idxStimuli])
    return entropy

# max probability (peak) of the posterior
def posteriorMax(p, hypotheses, choices = allChoices, allStimuli = allStimuli, rewardSetting = rewardSetting):
    maxPList = np.zeros(len(allStimuli))
    for iSti, stimulus in enumerate(allStimuli):
        pReward = np.mean([rewardSetting[np.sum(~np.isnan(h))-1][np.sum(np.equal(h, stimulus))] for h in hypotheses])
        posteriorRewarded = trialLikelihood(hypotheses, stimulus, reward = True)*p
        maxPRewarded = np.max(posteriorRewarded/np.sum(posteriorRewarded))
        posteriorUnrewarded = trialLikelihood(hypotheses, stimulus, reward = False)*p
        maxPUnrewarded = np.max(posteriorUnrewarded/np.sum(posteriorUnrewarded))
        maxPList[iSti] = maxPRewarded*pReward + maxPUnrewarded*(1-pReward)
    maxP = np.zeros(len(choices))
    for iChoice, choice in enumerate(choices):
        _, idxStimuli = choiceToStimuli(choice)
        maxP[iChoice] = np.mean(maxPList[idxStimuli])
    return maxP

# variance of the posterior
def posteriorVar(p, hypotheses, choices = allChoices, allStimuli = allStimuli, rewardSetting = rewardSetting):
    varList = np.zeros(len(allStimuli))
    for iSti, stimulus in enumerate(allStimuli):
        pReward = np.mean([rewardSetting[np.sum(~np.isnan(h))-1][np.sum(np.equal(h, stimulus))] for h in hypotheses])
        posteriorRewarded = trialLikelihood(hypotheses, stimulus, reward = True)*p
        varRewarded = np.var(posteriorRewarded/np.sum(posteriorRewarded))
        posteriorUnrewarded = trialLikelihood(hypotheses, stimulus, reward = False)*p
        varUnrewarded = np.var(posteriorUnrewarded/np.sum(posteriorUnrewarded))
        varList[iSti] = varRewarded*pReward + varUnrewarded*(1-pReward)
    var = np.zeros(len(choices))
    for iChoice, choice in enumerate(choices):
        _, idxStimuli = choiceToStimuli(choice)
        var[iChoice] = np.mean(varList[idxStimuli])
    return var
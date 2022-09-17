import numpy as np
import pandas

from taskSetting import *
from utilities import *

numDimensions = len(DIMENSIONS)
numFeaturesPerDimension = len(DIMENSIONS_TO_FEATURES[DIMENSIONS[0]])


def prepForFitting_conjunctionRL(data):

    allFeaturesList = flatten2Dlist([DIMENSIONS_TO_FEATURES[dim] for dim in DIMENSIONS])

    data = data.reset_index(drop=True)

    dataFitting = data[['game', 'trial', 'informed', 'numRelevantDimensions', 'rt', 'reward']].copy()

    for iRow in range(data.shape[0]):

        if pandas.isnull(dataFitting.loc[iRow, 'rt']):
            dataFitting.loc[iRow, 'choiceIndex'] = np.nan
            dataFitting.loc[iRow, 'stimulusIndex'] = np.nan
            for dim in DIMENSIONS:
                dataFitting.loc[iRow, 'choiceFeatureIndex_'+dim] = np.nan
                dataFitting.loc[iRow, 'stimulusFeatureIndex_'+dim] = np.nan

        else:
            # code choice as index according to allChoices
            choice = [(np.nan if pandas.isnull(data.loc[iRow, 'selectedFeature_' + dim]) else
                       DIMENSIONS_TO_FEATURES[dim].index(data.loc[iRow, 'selectedFeature_' + dim])) for dim in
                      DIMENSIONS]
            dataFitting.loc[iRow, 'choiceIndex'] = allChoices.index(choice)

            # code stimulus as index also according to allChoices
            stimulus = [DIMENSIONS_TO_FEATURES[dim].index(data.loc[iRow, 'builtFeature_' + dim]) for dim in DIMENSIONS]
            dataFitting.loc[iRow, 'stimulusIndex'] = allStimuli.index(stimulus)

    return dataFitting


def likelihood_conjuctionRLwDecayNoCost(pars, dataFitting, realdata=True, returnPrediction=False, returnTrialLikelihood=False, returnEstimates=False):

    # parameters
    if len(pars) == 3:
        beta, eta, cost, decay = pars[0], pars[1], 0, pars[2]
    elif len(pars) == 2:
        beta, eta, cost, decay = pars[0], pars[1], 0, 1

    # experiment info
    numGame = np.max(dataFitting['game'].values)
    gameLength = np.max(dataFitting['trial'].values)
    llh = np.zeros(dataFitting.shape[0])
    if returnPrediction:
        samplePList = []
    
    iRow = 0

    for iGame in dataFitting['game'].unique():

        dataThisGame = dataFitting[dataFitting['game'] == iGame]
        rt = dataThisGame['rt'].values
        choiceIndex = dataThisGame['choiceIndex'].values
        stimulusIndex = dataThisGame['stimulusIndex'].values
        reward = dataThisGame['reward'].values

        # initializing Q values
        Qconj = np.zeros(len(allChoices))

        for iTrial in range(gameLength):

            if (realdata == True) & np.isnan(rt[iTrial]):
                if returnPrediction:
                    samplePList.append([np.nan]*len(allChoices))
                llh[iRow] = np.nan

            else:
                # choice phase
                expectedR = Qconj
                expectedRwCost = expectedR - cost * (numDimensions - np.sum(np.isnan(allChoices), axis=1))
                sampleP = np.exp(beta * expectedRwCost) / np.sum(np.exp(beta * expectedRwCost))
                if returnPrediction:
                    samplePList.append(sampleP)
                llh[iRow] = np.log(sampleP[int(choiceIndex[iTrial])])

                # learning phase: RW learning
                stimulus = allStimuli[int(stimulusIndex[iTrial])]
                for iChoice in range(len(allChoices)):
                    if np.sum([np.isnan(allChoices[iChoice][iDim]) or (allChoices[iChoice][iDim]==stimulus[iDim]) for iDim in range(numDimensions)]) == numFeaturesPerDimension:
                        Qconj[iChoice] = Qconj[iChoice] + eta * (reward[iTrial] - Qconj[iChoice])
                    else:
                        Qconj[iChoice] = decay * Qconj[iChoice]
                
            iRow += 1
    
    if not (returnPrediction | returnTrialLikelihood | returnEstimates):
        results = -np.nansum(llh)
    else:
        results = []
        results.append(-np.nansum(llh))
        if returnPrediction:
            results.append(samplePList)
        if returnTrialLikelihood:
            results.append(llh)
    return results


def getExpectedR(Qconj):
    expectedR = np.zeros(len(allChoices))
    for iChoice, choice in enumerate(allChoices):
        dims = np.where(~np.isnan(choice))[0]
        if dims.shape[0] == 2: # sum of two 1D features
            f1 = deepcopy(choice)
            f1[dims[1]] = np.nan
            f2 = deepcopy(choice)
            f2[dims[0]] = np.nan
            Vconj = Qconj[idx_allChoices[tuple(f1)]] + Qconj[idx_allChoices[tuple(f2)]]
            expectedR[iChoice] = (Qconj[iChoice] + Vconj) / 2
        elif dims.shape[0] == 3: # sum of three 1D features, or sum of 1x1D+1x2D features
            v_f1 = Qconj[idx_allChoices[(choice[0], np.nan, np.nan)]]
            v_f2 = Qconj[idx_allChoices[(np.nan, choice[1], np.nan)]]
            v_f3 = Qconj[idx_allChoices[(np.nan, np.nan, choice[2])]]
            v_cf1 = Qconj[idx_allChoices[(np.nan, choice[1], choice[2])]]
            v_cf2 = Qconj[idx_allChoices[(choice[0], np.nan, choice[2])]]
            v_cf3 = Qconj[idx_allChoices[(choice[0], choice[1], np.nan)]]
            Vconj = [v_f1 + v_f2 + v_f3, v_f1 + v_cf1, v_f2 + v_cf2, v_f3 + v_cf3]
            expectedR[iChoice] = np.mean(Vconj + [Qconj[iChoice]])
        else:
            expectedR[iChoice] = Qconj[iChoice]
    return expectedR


def getExpectedRNew(Qconj, stimulus):
    v_f1 = Qconj[idx_allChoices[(stimulus[0], np.nan, np.nan)]]
    v_f2 = Qconj[idx_allChoices[(np.nan, stimulus[1], np.nan)]]
    v_f3 = Qconj[idx_allChoices[(np.nan, np.nan, stimulus[2])]]
    v_cf1 = Qconj[idx_allChoices[(np.nan, stimulus[1], stimulus[2])]]
    v_cf2 = Qconj[idx_allChoices[(stimulus[0], np.nan, stimulus[2])]]
    v_cf3 = Qconj[idx_allChoices[(stimulus[0], stimulus[1], np.nan)]]
    return np.mean([v_f1 + v_f2 + v_f3, v_f1 + v_cf1, v_f2 + v_cf2, v_f3 + v_cf3, Qconj[allChoices.index(stimulus)]])


def likelihood_conjuctionRLwDecayNoCostFlexChoice(pars, dataFitting, realdata=True, returnPrediction=False, returnTrialLikelihood=False, returnEstimates=False):

    # parameters
    if len(pars) == 3:
        beta, eta, cost, decay = pars[0], pars[1], 0, pars[2]
    elif len(pars) == 2:
        beta, eta, cost, decay = pars[0], pars[1], 0, 1

    # experiment info
    numGame = np.max(dataFitting['game'].values)
    gameLength = np.max(dataFitting['trial'].values)
    llh = np.zeros(dataFitting.shape[0])
    if returnPrediction:
        samplePList = []
    if returnEstimates:
        expectedROldList = []
        expectedRNewList = []
        expectedRAllList = []
    
    iRow = 0

    for iGame in dataFitting['game'].unique():

        dataThisGame = dataFitting[dataFitting['game'] == iGame]
        rt = dataThisGame['rt'].values
        choiceIndex = dataThisGame['choiceIndex'].values
        stimulusIndex = dataThisGame['stimulusIndex'].values
        reward = dataThisGame['reward'].values

        # initializing Q values
        Qconj = np.zeros(len(allChoices))

        for iTrial in range(gameLength):

            if (realdata == True) & np.isnan(rt[iTrial]):
                if returnPrediction:
                    samplePList.append([np.nan]*len(allChoices))
                if returnEstimates:
                    expectedROldList.append(np.nan)
                    expectedRNewList.append(np.nan)
                    expectedRAllList.append([np.nan] * len(allChoices))
                llh[iRow] = np.nan

            else:
                # choice phase
                expectedR = getExpectedR(Qconj)
                expectedRwCost = expectedR - cost * (numDimensions - np.sum(np.isnan(allChoices), axis=1))
                sampleP = np.exp(beta * expectedRwCost) / np.sum(np.exp(beta * expectedRwCost))
                if returnPrediction:
                    samplePList.append(sampleP)
                llh[iRow] = np.log(sampleP[int(choiceIndex[iTrial])])

                # learning phase: RW learning
                stimulus = allStimuli[int(stimulusIndex[iTrial])]
                expectedRNew = getExpectedRNew(Qconj, stimulus)
                for iChoice in range(len(allChoices)):
                    if np.sum([np.isnan(allChoices[iChoice][iDim]) or (allChoices[iChoice][iDim]==stimulus[iDim]) for iDim in range(numDimensions)]) == numFeaturesPerDimension:
                        Qconj[iChoice] = Qconj[iChoice] + eta * (reward[iTrial] - expectedRNew)
                    else:
                        Qconj[iChoice] = decay * Qconj[iChoice]
        
                if returnEstimates:
                    expectedROldList.append(expectedR[int(choiceIndex[iTrial])])
                    expectedRNewList.append(expectedRNew)
                    expectedRAllList.append(expectedR)
                
            iRow += 1
    
    if not (returnPrediction | returnTrialLikelihood | returnEstimates):
        results = -np.nansum(llh)
    else:
        results = []
        results.append(-np.nansum(llh))
        if returnPrediction:
            results.append(samplePList)
        if returnTrialLikelihood:
            results.append(llh)
        if returnEstimates:
            results.append(expectedROldList)
            results.append(expectedRNewList)
            results.append(expectedRAllList)
    return results
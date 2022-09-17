import numpy as np
import pandas

from taskSetting import *
from utilities import *

numDimensions = len(DIMENSIONS)
numFeaturesPerDimension = len(DIMENSIONS_TO_FEATURES[DIMENSIONS[0]])


def prepForFitting_featureRLwDecaySameEta(data):
    allFeaturesList = flatten2Dlist([DIMENSIONS_TO_FEATURES[dim] for dim in DIMENSIONS])

    data = data.reset_index(drop=True)

    dataFitting = data[['game', 'trial', 'informed', 'numRelevantDimensions', 'rt', 'reward']].copy()

    for iRow in range(data.shape[0]):

        if pandas.isnull(dataFitting.loc[iRow, 'rt']):
            dataFitting.loc[iRow, 'choiceIndex'] = np.nan
            dataFitting.loc[iRow, 'stimulusIndex'] = np.nan
            for dim in DIMENSIONS:
                dataFitting.loc[iRow, 'choiceFeatureIndex_' + dim] = np.nan
                dataFitting.loc[iRow, 'stimulusFeatureIndex_' + dim] = np.nan

        else:
            # code choice as index according to allChoices
            choice = [(np.nan if pandas.isnull(data.loc[iRow, 'selectedFeature_' + dim]) else
                       DIMENSIONS_TO_FEATURES[dim].index(data.loc[iRow, 'selectedFeature_' + dim])) for dim in
                      DIMENSIONS]
            dataFitting.loc[iRow, 'choiceIndex'] = allChoices.index(choice)

            # code stimulus as index also according to allChoices
            stimulus = [DIMENSIONS_TO_FEATURES[dim].index(data.loc[iRow, 'builtFeature_' + dim]) for dim in DIMENSIONS]
            dataFitting.loc[iRow, 'stimulusIndex'] = allStimuli.index(stimulus)

            # code features in choice and stimulus from 0 to 8
            for dim in DIMENSIONS:
                if pandas.isnull(data.loc[iRow, 'selectedFeature_' + dim]):
                    dataFitting.loc[iRow, 'choiceFeatureIndex_' + dim] = np.nan
                else:
                    dataFitting.loc[iRow, 'choiceFeatureIndex_' + dim] = allFeaturesList.index(
                        data.loc[iRow, 'selectedFeature_' + dim])
                dataFitting.loc[iRow, 'stimulusFeatureIndex_' + dim] = allFeaturesList.index(
                    data.loc[iRow, 'builtFeature_' + dim])

    return dataFitting


def likelihood_featureRLwDecaySameEtaNoCost(pars, dataFitting, realdata=True, returnPrediction=False,
                                     returnTrialLikelihood=False, returnEstimates=False):
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

    featureMatAllChoices = choiceToFeatureMat(allChoices)
    featureMatAllStimuli = stimulusToFeatureMat(allStimuli)

    iRow = 0

    for iGame in dataFitting['game'].unique():

        dataThisGame = dataFitting[dataFitting['game'] == iGame]
        rt = dataThisGame['rt'].values
        choiceIndex = dataThisGame['choiceIndex'].values
        stimulusIndex = dataThisGame['stimulusIndex'].values
        choiceFeatureIndexes = dataThisGame[['choiceFeatureIndex_' + dim for dim in DIMENSIONS]].values
        stimulusFeatureIndexes = dataThisGame[['stimulusFeatureIndex_' + dim for dim in DIMENSIONS]].values
        reward = dataThisGame['reward'].values

        # initializing Q values
        Qfeat = np.zeros(numDimensions * numFeaturesPerDimension)

        for iTrial in range(gameLength):

            if (realdata == True) & np.isnan(rt[iTrial]):
                if returnPrediction:
                    samplePList.append([np.nan] * len(allChoices))
                if returnEstimates:
                    expectedROldList.append(np.nan)
                    expectedRNewList.append(np.nan)
                    expectedRAllList.append([np.nan] * len(allChoices))
                llh[iRow] = np.nan

            else:
                # choice phase
                expectedR = np.dot(featureMatAllChoices, Qfeat)
                expectedRwCost = expectedR - cost * (numDimensions - np.sum(np.isnan(allChoices), axis=1))
                sampleP = np.exp(beta * expectedRwCost) / np.sum(np.exp(beta * expectedRwCost))
                if returnPrediction:
                    samplePList.append(sampleP)
                llh[iRow] = np.log(sampleP[int(choiceIndex[iTrial])])

                # learning phase: RW learning with new estimates of expected reward
                expectedRNew = np.dot(featureMatAllStimuli[int(stimulusIndex[iTrial])], Qfeat)
                for iDim in range(numDimensions):
                    iFeat_stimulus = int(stimulusFeatureIndexes[iTrial, iDim])
                    for iFeat in range(iDim*numFeaturesPerDimension, (iDim+1)*numFeaturesPerDimension):
                        if iFeat == iFeat_stimulus:
                            Qfeat[iFeat] = Qfeat[iFeat] + eta * (reward[iTrial] - expectedRNew)
                        else:
                            Qfeat[iFeat] = decay * Qfeat[iFeat]

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
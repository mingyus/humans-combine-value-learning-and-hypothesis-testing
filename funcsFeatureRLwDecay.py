import numpy as np
import pandas

from taskSetting import *
from utilities import *

numDimensions = len(DIMENSIONS)
numFeaturesPerDimension = len(DIMENSIONS_TO_FEATURES[DIMENSIONS[0]])


def prepForFitting_featureRLwDecay(data):
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


def likelihood_featureRLwDecayNoCost(pars, dataFitting, realdata=True, returnPrediction=False,
                                     returnTrialLikelihood=False, returnEstimates=False):
    # parameters
    beta, eta_s, eta_r, cost, decay = pars[0], pars[1], pars[2], 0, pars[3]

    # experiment info
    numGame = np.max(dataFitting['game'].values)
    gameLength = np.max(dataFitting['trial'].values)
    llh = np.zeros(dataFitting.shape[0])
    if returnPrediction:
        samplePList = []
    if returnEstimates:
        QfeatList = []
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
                    QfeatList.append(np.nan)
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
                            if ~np.isnan(choiceFeatureIndexes[iTrial, iDim]):  # selected feature
                                Qfeat[iFeat] = Qfeat[iFeat] + eta_s * (reward[iTrial] - expectedRNew)
                            else:  # randomly selected feature
                                Qfeat[iFeat] = Qfeat[iFeat] + eta_r * (reward[iTrial] - expectedRNew)
                        else:
                            Qfeat[iFeat] = decay * Qfeat[iFeat]

                if returnEstimates:
                    QfeatList.append(deepcopy(Qfeat))
                    expectedROldList.append(deepcopy(expectedR[int(choiceIndex[iTrial])]))
                    expectedRNewList.append(deepcopy(expectedRNew))
                    expectedRAllList.append(deepcopy(expectedR))

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
            results.append(QfeatList)
            results.append(expectedROldList)
            results.append(expectedRNewList)
            results.append(expectedRAllList)
    return results


def getModelSimuCSVFileName_featureRLwDecay(beta, eta_s, eta_r, cost, decay, gameLength, numGamePerType):
    betaStr = ('_hardmax' if beta is None else ('_softmaxBeta' + str(beta)))
    costStr = '_cost' + str(cost)
    etasStr = '_etaS' + str(eta_s)
    etarStr = '_etaR' + str(eta_r)
    decayStr = '_decay' + str(decay)

    return 'featureRLwDecay' + betaStr + etasStr + etarStr + costStr + decayStr + '_gameLength' + str(
        gameLength) + '_numGamePerType' + str(numGamePerType) + '.csv'


def model_featureRLwDecay(modelName, pars, gameLength, numGamePerType, ifSaveCSV, ifFlatPrior=None):
    # parameters
    if modelName == 'featureRLwDecay':
        beta, eta_s, eta_r, cost, decay = pars[0], pars[1], pars[2], pars[3], pars[4]
    elif modelName == 'featureRLwDecayNoCost':
        beta, eta_s, eta_r, cost, decay = pars[0], pars[1], pars[2], 0, pars[3]

    # create the variables
    game, trial, informedList, numRelevantDimensionsList, reward, numSelectedFeatures, rt = \
        zerosLists(numList=7, lengthList=gameLength * numGamePerType * 6)
    ifRelevantDimension, rewardingFeature, selectedFeature, randomlySelectedFeature, builtFeature = \
        emptyDicts(numDict=5, keys=DIMENSIONS, lengthList=gameLength * numGamePerType * 6)

    featureMatAllChoices = choiceToFeatureMat(allChoices)

    # simulation
    iRow = 0
    iGame = 0
    for informed in [True, False]:
        for numRelevantDimensions in np.arange(3) + 1:

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
                Qfeat = np.zeros(numDimensions * numFeaturesPerDimension)

                for iTrial in range(gameLength):

                    trial[iRow] = iTrial + 1

                    # choice phase (hard max; if more than one hypothesis have the highest posterior, pick randomly)
                    expectedRwCost = np.dot(featureMatAllChoices, Qfeat) - cost * (
                            numDimensions - np.sum(np.isnan(allChoices), axis=1))
                    sampleP = np.exp(beta * expectedRwCost) / np.sum(np.exp(beta * expectedRwCost))
                    indChoice = np.random.choice(np.arange(len(allChoices)), size=1, p=sampleP)[0]
                    choice = allChoices[indChoice]

                    # generate stimulus and reward outcome
                    stimulus = deepcopy(choice)
                    for iDim, dim in enumerate(DIMENSIONS):
                        if ~np.isnan(choice[iDim]):
                            selectedFeature[dim][iRow] = DIMENSIONS_TO_FEATURES[dim][choice[iDim]]
                            randomlySelectedFeature[dim][iRow] = np.nan
                            builtFeature[dim][iRow] = selectedFeature[dim][iRow]
                        else:
                            selectedFeature[dim][iRow] = np.nan
                            stimulus[iDim] = np.random.choice(np.arange(len(DIMENSIONS_TO_FEATURES[dim])), 1)[0]
                            randomlySelectedFeature[dim][iRow] = DIMENSIONS_TO_FEATURES[dim][stimulus[iDim]]
                            builtFeature[dim][iRow] = randomlySelectedFeature[dim][iRow]
                    numSelectedFeatures[iRow] = np.array(
                        [(not pandas.isnull(selectedFeature[dim][iRow])) for dim in DIMENSIONS]).sum()
                    numRewardingFeatureBuilt = np.array([((not pandas.isnull(rewardingFeature[dim][iRow])) &
                                                          (builtFeature[dim][iRow] == rewardingFeature[dim][iRow]))
                                                         for dim in DIMENSIONS]).sum()
                    reward[iRow] = (
                            np.random.random() < rewardSetting[numRelevantDimensions - 1][numRewardingFeatureBuilt])

                    # learning phase: RW learning
                    expectedRNew = np.dot(choiceToFeatureMat(choices=[stimulus]), Qfeat)[0]
                    for iDim in range(numDimensions):
                        iFeat_stimulus = int(iDim * numFeaturesPerDimension + stimulus[iDim])
                        for iFeat in range(iDim*numFeaturesPerDimension, (iDim+1)*numFeaturesPerDimension):
                            if iFeat == iFeat_stimulus:
                                if ~np.isnan(choice[iDim]):  # selected feature
                                    Qfeat[iFeat] = Qfeat[iFeat] + eta_s * (reward[iRow] - expectedRNew)
                                else:  # randomly selected feature
                                    Qfeat[iFeat] = Qfeat[iFeat] + eta_r * (reward[iRow] - expectedRNew)
                            else:
                                Qfeat[iFeat] = decay * Qfeat[iFeat]

                    iRow += 1

                iGame += 1

    # save variables into dataframe
    simudata = pandas.DataFrame(
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
        simudata.to_csv(
            'modelSimulation/' + getModelSimuCSVFileName_featureRLwDecay(beta, eta_s, eta_r, cost, decay, gameLength,
                                                                               numGamePerType), index=False)

    return simudata

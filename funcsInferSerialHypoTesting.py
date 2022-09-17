from utilities import *
from taskSetting import *
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from copy import deepcopy


numDimensions = len(DIMENSIONS)
numFeaturesPerDimension = len(DIMENSIONS_TO_FEATURES[DIMENSIONS[0]])

taskCondKeys = [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2), (False, 3)]

def prepForFitting_inferSerialHypoTesting(model, data):
    # initialize the output variable
    keys = ['dataFitting', 'logPhSwitchmodelAll']
    if 'LRTest' in model:
        keys = keys + ['posterior']
    elif 'Counting' in model:
        keys = keys + ['estimatedPReward']
    if 'ValueBasedSwitch' in model:
        keys = keys + ['featureMatChoices', 'featureMatStimuli', 'featureMatAllHypothesesAll']
    if 'Epsilon' in model:
        keys = keys + ['consistentCHAll']
    elif 'SelectMore' in model:
        keys = keys + ['numMoreDimAll']
    output = dict.fromkeys(keys)

    # prepare the data for fitting
    allFeaturesList = flatten2Dlist([DIMENSIONS_TO_FEATURES[dim] for dim in DIMENSIONS])
    data = data.reset_index(drop=True)
    dataFitting = data[['game', 'trial', 'informed', 'numRelevantDimensions', 'rt', 'reward']].copy()
    choiceIndex, stimulusIndex = zerosLists(numList=2, lengthList=data.shape[0])
    choiceFeatureIndex, stimulusFeatureIndex, selectedFeature, builtFeature = emptyDicts(numDict=4, keys=DIMENSIONS, lengthList=data.shape[0])
    for dim in DIMENSIONS:
        selectedFeature[dim] = data['selectedFeature_' + dim]
        builtFeature[dim] = data['builtFeature_' + dim]

    for iRow in range(data.shape[0]):

        if pd.isnull(dataFitting.loc[iRow, 'rt']):
            choiceIndex[iRow] = np.nan
            stimulusIndex[iRow] = np.nan
            for dim in DIMENSIONS:
                choiceFeatureIndex[dim][iRow] = np.nan
                stimulusFeatureIndex[dim][iRow] = np.nan

        else:
            # code choice as index according to allChoices
            choice = [(np.nan if pd.isnull(selectedFeature[dim][iRow]) else DIMENSIONS_TO_FEATURES[dim].index(selectedFeature[dim][iRow])) for dim in DIMENSIONS]
            choiceIndex[iRow] = allChoices.index(choice)

            # code stimulus as index also according to allChoices
            stimulus = [DIMENSIONS_TO_FEATURES[dim].index(builtFeature[dim][iRow]) for dim in DIMENSIONS]
            stimulusIndex[iRow] = allStimuli.index(stimulus)

            # code features in choice and stimulus from 0 to 8
            for dim in DIMENSIONS:
                if pd.isnull(selectedFeature[dim][iRow]):
                    choiceFeatureIndex[dim][iRow] = np.nan
                else:
                    choiceFeatureIndex[dim][iRow] = allFeaturesList.index(selectedFeature[dim][iRow])
                stimulusFeatureIndex[dim][iRow] = allFeaturesList.index(builtFeature[dim][iRow])

    dataFitting['choiceIndex'], dataFitting['stimulusIndex'] = choiceIndex, stimulusIndex
    for dim in DIMENSIONS:
        dataFitting['choiceFeatureIndex_'+dim], dataFitting['stimulusFeatureIndex_'+dim] = choiceFeatureIndex[dim], stimulusFeatureIndex[dim]
    output['dataFitting'] = dataFitting

    # prepare for the hypothesis testing model:
    # the posterior of bayesian inference for the LR test; or the estimated reward probability for the counting policy
    gameLength = np.max(dataFitting['trial'].values)
    if 'DiffThres' not in model:
        allHypothesesAll, _, hPriorAll = getGamesHypoInfo(model)
    else:
        allHypothesesAll, _, hPriorAll, _ = getGamesHypoInfo(model)

    keys = [(iGame, iTrial, lOld) for iGame in dataFitting['game'].unique() for iTrial in range(gameLength) for lOld in range(iTrial)]
    if 'LRTest' in model:
        posterior = dict(zip(keys, [None for _ in range(len(keys))]))
    elif 'Counting' in model:
        estimatedPReward = dict(zip(keys, [None for _ in range(len(keys))]))

    for iGame in dataFitting['game'].unique():
        # get the data
        dataThisGame = dataFitting[dataFitting['game'] == iGame].reset_index(drop=True)
        rt = dataThisGame['rt'].values
        stimulusIndex = dataThisGame['stimulusIndex'].values
        reward = dataThisGame['reward'].values

        # data in only the valid trials
        stimulusIndex_valid = stimulusIndex[~np.isnan(rt)]
        reward_valid = reward[~np.isnan(rt)]

        # hypothesis space
        allHypotheses = allHypothesesAll[dataThisGame.loc[0, 'informed'], dataThisGame.loc[0, 'numRelevantDimensions']]
        hPrior = hPriorAll[dataThisGame.loc[0, 'informed'], dataThisGame.loc[0, 'numRelevantDimensions']]
        NHypos = len(allHypotheses)

        t = 0
        for iTrial in range(gameLength):
            if not np.isnan(rt[iTrial]):
                if t > 0:
                    if 'LRTest' in model:
                        if 'FlexibleHypo' in model:
                            raise Exception('LRTest and FlexibleHypo are not supported together...')
                        else:
                            logp = np.log(hPrior)
                            for lOld in range(t):
                                loglik = np.zeros(NHypos)
                                for iH, h in enumerate(allHypotheses):
                                    pReward = rewardSetting[np.sum(~np.isnan(h)) - 1][np.sum(np.equal(h, allStimuli[int(stimulusIndex_valid[t - 1 - lOld])]))]
                                    loglik[iH] = np.log(pReward) if reward_valid[t - 1 - lOld] else np.log(1 - pReward)
                                logp = loglik + logp
                                logp = logp - logsumexp(logp)
                                posterior[iGame, iTrial, lOld] = np.exp(logp)
                    elif 'Counting' in model:
                        if not 'CountingAll' in model:
                            for lOld in range(t):
                                rewardCount = np.zeros(NHypos)
                                trialCount = np.zeros(NHypos)
                                for iH, h in enumerate(allHypotheses):
                                    for i in range(t - 1 - lOld, t):
                                        if np.sum([np.isnan(h[iDim]) | (h[iDim] == allStimuli[int(stimulusIndex_valid[i])][iDim]) for iDim in range(numDimensions)]) == numDimensions:  # compatible trial
                                            rewardCount[iH] += reward_valid[i]
                                            trialCount[iH] += 1
                                estimatedPReward[iGame, iTrial, lOld] = (rewardCount + 1) / (trialCount + 2)
                        else:  # TODO: remove this after done with testing
                            for lOld in range(t):
                                rewardCount = np.ones(NHypos) * np.sum(reward_valid[t - 1 - lOld:t])
                                trialCount = np.ones(NHypos) * (lOld + 1)
                                estimatedPReward[iGame, iTrial, lOld] = (rewardCount + 1) / (trialCount + 2)
                t += 1

    if 'LRTest' in model:
        output['posterior'] = posterior
    elif 'Counting' in model:
        output['estimatedPReward'] = estimatedPReward


    # prepare for the value-based switch policy
    if 'ValueBasedSwitch' in model:
        # feature mat for choices and stimuli (used for feature values)
        featureMatChoices, featureMatStimuli = [np.empty((np.max(dataFitting['game'].unique()) + 1, gameLength, numDimensions * numFeaturesPerDimension)) for _ in range(2)]
        featureMatChoices[:] = None
        featureMatStimuli[:] = None
        for iGame in dataFitting['game'].unique():
            dataThisGame = dataFitting[dataFitting['game'] == iGame].reset_index(drop=True)
            for i in range(dataThisGame.shape[0]):
                if not pd.isnull(dataThisGame.loc[i, 'rt']):
                    featureMatChoices[iGame, i, :] = choiceToFeatureMat(choices=[allChoices[int(dataThisGame.loc[i, 'choiceIndex'])]])
                    featureMatStimuli[iGame, i, :] = stimulusToFeatureMat(stimuli=[allStimuli[int(dataThisGame.loc[i, 'stimulusIndex'])]])
        output['featureMatChoices'] = featureMatChoices
        output['featureMatStimuli'] = featureMatStimuli

        # feature mat for hypotheses (used to calculate expected reward for each hypothesis)
        featureMatAllHypothesesAll = dict.fromkeys(taskCondKeys)
        for informed in [True, False]:
            for numRD in np.arange(numDimensions) + 1:
                allHypotheses = allHypothesesAll[informed, numRD]
                featureMatAllHypothesesAll[informed, numRD] = hypothesisToFeatureMat(hypotheses=allHypotheses)
        output['featureMatAllHypothesesAll'] = featureMatAllHypothesesAll


    # prepare for the choice policy
    if 'Epsilon' in model:
        consistentCHAll = dict.fromkeys(taskCondKeys)
    elif 'SelectMore' in model:
        numMoreDimAll = dict.fromkeys(taskCondKeys)
    for informed in [True, False]:
        for numRD in np.arange(numDimensions) + 1:
            allHypotheses = allHypothesesAll[informed, numRD]
            if 'Epsilon' in model:
                consistentCHAll[informed, numRD] = getConsistentCH(allHypotheses)
            elif 'SelectMore' in model:
                numMoreDimAll[informed, numRD] = getNumMoreDim(allHypotheses)
    if 'Epsilon' in model:
        output['consistentCHAll'] = consistentCHAll
    elif 'SelectMore' in model:
        output['numMoreDimAll'] = numMoreDimAll

    return output


def getModelParNames_inferSerialHypoTesting(model):
    parNames = ['betaStay']
    if 'DiffThres' in model:
        parNames = parNames + ['deltaStay']
    elif 'SeparateThres' in model:
        parNames = parNames + ['thetaStay1D', 'thetaStay2D', 'thetaStay3D', 'thetaStayUnknown']
    else:
        parNames = parNames + ['thetaStay']
    if 'RandomSwitch' in model:
        if 'ThresonTest' in model:
            parNames = parNames + ['pTest']
    elif 'ValueBasedSwitch' in model:
        if ('DiffEta' in model) and ('Decay' in model):
            parNames = parNames + ['eta_s', 'eta_r', 'decay', 'betaSwitch']
        elif ('DiffEta' in model) and ('Decay' not in model):
            parNames = parNames + ['eta_s', 'eta_r', 'betaSwitch']
        elif ('DiffEta' not in model) and ('Decay' in model):
            parNames = parNames + ['eta', 'decay', 'betaSwitch']
        elif ('DiffEta' not in model) and ('Decay' not in model):
            parNames = parNames + ['eta', 'betaSwitch']
        if 'ThresonTest' in model:
            parNames = parNames + ['betaTest', 'thetaTest']
    parNames = parNames + ['epsilon']
    if 'SelectMore' in model:
        parNames = parNames + ['kChoice']
    if 'Cost' in model and 'NoCost' not in model:
        parNames = parNames + ['cost']
    if 'FlexibleHypo' in model:
        if 'PerDim' not in model:
            parNames = parNames + ['wl', 'wh']
        else:
            parNames = parNames + ['w1D2', 'w1D3', 'w2D1', 'w2D3', 'w3D1', 'w3D2']
    if 'Separate' in model:
        parNames = parNames + ['w2', 'w3']
    return parNames


def getModelSimuCSVFileName_inferSerialHypoTesting(model, pars, gameLength, numGamePerType):
    parNames = getModelParNames_inferSerialHypoTesting(model)
    parStr = ''
    for iPar in range(len(pars)):
        parStr = parStr + '_' + parNames[iPar] + str(pars[iPar])
    return model + parStr + '_gameLength' + str(gameLength) + '_numGamePerType' + str(numGamePerType) + '.csv'


def likelihood_inferSerialHypoTesting(pars, model, dataFitting, realdata=True, returnPrediction=False, returnTrialLikelihood=False, returnQvalues=False, returnPh=False):
    # get the data and pre-calculated variables
    if 'LRTest' in model:
        posterior = dataFitting['posterior']
    elif 'Counting' in model:
        estimatedPReward = dataFitting['estimatedPReward']
    if 'ValueBasedSwitch' in model:
        featureMatChoices = dataFitting['featureMatChoices']
        featureMatStimuli = dataFitting['featureMatStimuli']
        featureMatAllHypothesesAll = dataFitting['featureMatAllHypothesesAll']
    if 'Epsilon' in model:
        consistentCHAll = dataFitting['consistentCHAll']
    elif 'SelectMore' in model:
        numMoreDimAll = dataFitting['numMoreDimAll']
    dataFitting = dataFitting['dataFitting']

    # parameters
    betaStay = pars[0]
    i = 0
    if 'DiffThres' in model:
        deltaStay = pars[i+1]
        i += 1
    elif 'SeparateThres' in model:
        thetaStay1D, thetaStay2D, thetaStay3D, thetaStayUnknown = pars[i+1], pars[i+2], pars[i+3], pars[i+4]
        thetaStayAll = dict()
        thetaStayAll[True, 1] = thetaStay1D
        thetaStayAll[True, 2] = thetaStay2D
        thetaStayAll[True, 3] = thetaStay3D
        for numRD in np.arange(numDimensions) + 1:
            thetaStayAll[False, numRD] = thetaStayUnknown
        i += 4
    else:
        thetaStay = pars[i+1]
        i += 1
    if 'RandomSwitch' in model:
        if 'ThresonTest' in model:
            pTest = pars[i+1]
            i += 1
        else:
            pTest = None
    elif 'ValueBasedSwitch' in model:
        if ('DiffEta' in model) and ('Decay' in model):
            eta_s, eta_r, decay, betaSwitch = pars[i+1], pars[i+2], pars[i+3], pars[i+4]
            i += 4
        elif ('DiffEta' in model) and ('Decay' not in model):
            eta_s, eta_r, betaSwitch = pars[i+1], pars[i+2], pars[i+3]
            decay = 1
            i += 3
        elif ('DiffEta' not in model) and ('Decay' in model):
            eta, decay, betaSwitch = pars[i+1], pars[i+2], pars[i+3]
            eta_s, eta_r = eta, eta
            i += 3
        elif ('DiffEta' not in model) and ('Decay' not in model):
            eta, betaSwitch = pars[i+1], pars[i+2]
            eta_s, eta_r = eta, eta
            decay = 1
            i += 2
        if 'ThresonTest' in model:
            betaTest, thetaTest = pars[i+1], pars[i+2]
            i += 2
        else:
            betaTest, thetaTest = None, None
    epsilon = pars[i+1]
    i += 1
    if 'SelectMore' in model:
        kChoice = pars[i+1]
        i += 1
    if 'Cost' in model and 'NoCost' not in model:
        cost = pars[i+1]
        i += 1
    else:
        cost = 0
    if 'FlexibleHypo' in model:
        if 'PerDim' not in model:
            wl, wh = pars[i+1], pars[i+2]
            w = [wl ,wh]
            i += 2
        else:
            w1D2, w1D3, w2D1, w2D3, w3D1, w3D2 = pars[i+1:i+7]
            w = [w1D2, w1D3, w2D1, w2D3, w3D1, w3D2]
            i += 6
    if 'Separate' in model:
        w2, w3 = pars[i+1], pars[i+2]

    # experiment info
    gameLength = np.max(dataFitting['trial'].values)
    NChoices = len(allChoices)
    if 'DiffThres' not in model:
        allHypothesesAll, NHyposAll, hPriorAll = getGamesHypoInfo(model)
    else:
        allHypothesesAll, NHyposAll, hPriorAll, maxRPAll = getGamesHypoInfo(model)
    if 'FlexibleHypo' in model:
        hWeightAll = hypothesisWeightKnown_flexibleHypoSpace(model, NHyposAll, w)
        if 'Separate' in model:
            hWeightAll = hypothesisWeightUnknown_separate(model, hWeightAll, w2, w3)
        loghPriorWAll = loghPriorWAll_flexibleHypoSpace(model, hPriorAll, hWeightAll)
    else:
        loghPriorWAll = dict(zip(taskCondKeys, [np.log(hPriorAll[key]) for key in taskCondKeys]))

    # prepare for the switch policy (calculate logPhSwitchmodel_randomSwitch)
    if 'RandomSwitch' in model:
        logPhSwitchmodelAll = emptyDicts(numDict=1, keys=taskCondKeys, lengthList=0)
        for informed in [True, False]:
            for numRD in np.arange(numDimensions) + 1:
                NHypos, loghPriorW = NHyposAll[informed, numRD], loghPriorWAll[informed, numRD]
                for t in range(gameLength):
                    logPhSwitchmodelAll[NHypos, t] = calculate_logPhSwitchmodel_randomSwitch(NHypos, t, loghPriorW, pTest)
    
    # calculate cost
    costAll = getCost(cost, allHypothesesAll)

    llh = np.zeros(dataFitting.shape[0])
    if returnPrediction:
        samplePList = []
    if returnQvalues:
        QfeatList = []
    if returnPh:
        PhList = []

    iRow = 0

    for iGame in dataFitting['game'].unique():

        dataThisGame = dataFitting[dataFitting['game'] == iGame].reset_index(drop=True)
        rt = dataThisGame['rt'].values
        choiceIndex = dataThisGame['choiceIndex'].values
        stimulusIndex = dataThisGame['stimulusIndex'].values
        reward = dataThisGame['reward'].values

        # data in only the valid trials
        choiceIndex_valid = choiceIndex[~np.isnan(rt)]
        stimulusIndex_valid = stimulusIndex[~np.isnan(rt)]
        reward_valid = reward[~np.isnan(rt)]
        if 'ValueBasedSwitch' in model:
            featureMatChoices_valid = featureMatChoices[iGame, ~np.isnan(rt), :]
            featureMatStimuli_valid = featureMatStimuli[iGame, ~np.isnan(rt), :]

        # game info
        informed = dataThisGame.loc[0, 'informed']
        numRD = dataThisGame.loc[0, 'numRelevantDimensions']
        if 'DiffThres' in model:
            maxRP = maxRPAll[informed, numRD]

        # hypothesis space
        NHypos, loghPriorW = NHyposAll[informed, numRD], loghPriorWAll[informed, numRD]
        if 'ValueBasedSwitch' in model:
            featureMatAllHypotheses = featureMatAllHypothesesAll[informed, numRD]
        if 'Epsilon' in model:
            consistentCH = consistentCHAll[informed, numRD]
        elif 'SelectMore' in model:
            numMoreDim = numMoreDimAll[informed, numRD]

        # feature value learning
        if 'ValueBasedSwitchNoReset' in model:
            Qfeat_valid = QfeatNoReset(featureMatChoices_valid, featureMatStimuli_valid, reward=reward_valid, eta_s=eta_s, eta_r=eta_r, decay=decay)
        elif 'ValueBasedSwitchReset' in model:
            Qfeat_valid = QfeatReset(featureMatChoices_valid, featureMatStimuli_valid, reward=reward_valid, eta_s=eta_s, eta_r=eta_r, decay=decay)

        # cost
        costThis = costAll[informed, numRD]

        # initialization
        t = 0

        for iTrial in range(gameLength):

            if (realdata is True) & np.isnan(rt[iTrial]):
                if returnPrediction:
                    samplePList.append([np.nan] * len(allChoices))
                if returnPh:
                    PhList.append(np.nan)
                llh[iRow] = np.nan

            else:

                # the switch policy (first term)
                if 'RandomSwitch' in model:
                    logPhSwitchmodel = logPhSwitchmodelAll[NHypos, t]
                elif 'ValueBasedSwitchNoReset' in model:
                    logPhSwitchmodel = calculate_logPhSwitchmodel_valueBasedNoReset(NHypos, t, featureMatAllHypotheses, Qfeat_valid, loghPriorW, betaSwitch, betaTest, thetaTest, costThis)
                elif 'ValueBasedSwitchReset' in model:
                    logPhSwitchmodel = calculate_logPhSwitchmodel_valueBasedReset(NHypos, t, featureMatAllHypotheses, Qfeat_valid, loghPriorW, betaSwitch, betaTest, thetaTest, costThis)

                if t == 0:  # the first trial (with response)
                    logPh = logPhSwitchmodel

                    # save for use on next trial - part1
                    logPhOldLast = logPhSwitchmodel
                    logPlOldLast = None
                    logPhSwitchmodelLast = None
                    logPlRunlengthmodelLast = None

                else:

                    # recursive calculation (second and fourth terms)
                    logPhOld, logPlOld = calculate_24terms(t, logPhSwitchmodelLast, logPhOldLast, logPlOldLast, logPlRunlengthmodelLast, logPchLast)

                    # the hypothesis testing policy/the run-length model (third term)
                    if 'DiffThres' in model:
                        thetaStay = maxRP + deltaStay
                    if 'SeparateThres' in model:
                        thetaStay = thetaStayAll[informed, numRD]
                    if 'LRTest' in model:
                        logPlRunlengthmodel = hypothesisTestingPolicy_LRTest(t, iGame, iTrial, posterior, logPhOld, betaStay, thetaStay)
                    elif 'Counting' in model:
                        logPlRunlengthmodel = hypothesisTestingPolicy_Counting(t, iGame, iTrial, estimatedPReward, logPhOld, betaStay, thetaStay)

                    # posterior over hypotheses
                    logPh = calculate_logPh(logPhSwitchmodel, logPhOld, logPlRunlengthmodel, logPlOld)

                    # save for use on next trial - part1
                    logPhOldLast = logPhOld
                    logPlOldLast = logPlOld
                    logPhSwitchmodelLast = logPhSwitchmodel
                    logPlRunlengthmodelLast = logPlRunlengthmodel

                # the choice policy: choice probability of the current trial, given all possible hypotheses
                if 'Epsilon' in model:
                    logPchFull, logpChoice = choicePolicy_epsilon(NChoices, NHypos, logPh, consistentCH, epsilon)
                elif 'SelectMore' in model:
                    logPchFull, logpChoice = choicePolicy_selectMore(NChoices, NHypos, logPh, numMoreDim, epsilon, kChoice)

                # save for use on next trial - part2
                logPchLast = logPchFull[int(choiceIndex_valid[t]), :]

                # likelihood of the trial
                llh[iRow] = logpChoice[int(choiceIndex_valid[t])]

                if returnPrediction:
                    samplePList.append(np.exp(logpChoice))
                if returnQvalues:
                    QfeatList.append(Qfeat_valid[t])
                if returnPh:
                    PhList.append(np.exp(logPh))

                # count of "normal" trials: trials w/ a response
                t += 1

            iRow += 1

    i_valid = np.invert(np.isnan(dataFitting['rt'].values))
    if not (returnPrediction | returnTrialLikelihood | returnQvalues | returnPh):
        results = -np.sum(llh[i_valid])
    else:
        results = []
        results.append(-np.sum(llh[i_valid]))
        if returnPrediction:
            results.append(samplePList)
        if returnTrialLikelihood:
            results.append(llh)
        if returnQvalues:
            results.append(QfeatList)
        if returnPh:
            results.append(PhList)
    return results


def model_inferSerialHypoTesting(model, pars, gameLength, numGamePerType, ifSaveCSV, ifReturnLlh=False):
    # parameters
    betaStay = pars[0]
    i = 0
    if 'DiffThres' in model:
        deltaStay = pars[i+1]
        i += 1
    elif 'SeparateThres' in model:
        thetaStay1D, thetaStay2D, thetaStay3D, thetaStayUnknown = pars[i+1], pars[i+2], pars[i+3], pars[i+4]
        thetaStayAll = dict()
        thetaStayAll[True, 1] = thetaStay1D
        thetaStayAll[True, 2] = thetaStay2D
        thetaStayAll[True, 3] = thetaStay3D
        for numRD in np.arange(numDimensions) + 1:
            thetaStayAll[False, numRD] = thetaStayUnknown
        i += 4
    else:
        thetaStay = pars[i+1]
        i += 1
    if 'RandomSwitch' in model:
        if 'ThresonTest' in model:
            pTest = pars[i+1]
            i += 1
        else:
            pTest = None
    elif 'ValueBasedSwitch' in model:
        if ('DiffEta' in model) and ('Decay' in model):
            eta_s, eta_r, decay, betaSwitch = pars[i+1], pars[i+2], pars[i+3], pars[i+4]
            i += 4
        elif ('DiffEta' in model) and ('Decay' not in model):
            eta_s, eta_r, betaSwitch = pars[i+1], pars[i+2], pars[i+3]
            decay = 1
            i += 3
        elif ('DiffEta' not in model) and ('Decay' in model):
            eta, decay, betaSwitch = pars[i+1], pars[i+2], pars[i+3]
            eta_s, eta_r = eta, eta
            i += 3
        elif ('DiffEta' not in model) and ('Decay' not in model):
            eta, betaSwitch = pars[i+1], pars[i+2]
            eta_s, eta_r = eta, eta
            decay = 1
            i += 2
        if 'ThresonTest' in model:
            betaTest, thetaTest = pars[i+1], pars[i+2]
            i += 2
        else:
            betaTest, thetaTest = None, None
    epsilon = pars[i+1]
    i += 1
    if 'SelectMore' in model:
        kChoice = pars[i+1]
        i += 1
    if 'Cost' in model and 'NoCost' not in model:
        cost = pars[i+1]
        i += 1
    else:
        cost = 0
    if 'FlexibleHypo' in model:
        if 'PerDim' not in model:
            wl, wh = pars[i+1], pars[i+2]
            w = [wl ,wh]
            i += 2
        else:
            w1D2, w1D3, w2D1, w2D3, w3D1, w3D2 = pars[i+1:i+7]
            w = [w1D2, w1D3, w2D1, w2D3, w3D1, w3D2]
            i += 6
    if 'Separate' in model:
        w2, w3 = pars[i+1], pars[i+2]

    # experiment info
    NChoices = len(allChoices)
    if 'DiffThres' not in model:
        allHypothesesAll, NHyposAll, hPriorAll = getGamesHypoInfo(model)
    else:
        allHypothesesAll, NHyposAll, hPriorAll, maxRPAll = getGamesHypoInfo(model)

    # prepare variables for reuse
    # hypothesis space
    if 'FlexibleHypo' in model:
        hWeightAll = hypothesisWeightKnown_flexibleHypoSpace(model, NHyposAll, w)
        if 'Separate' in model:
            hWeightAll = hypothesisWeightUnknown_separate(model, hWeightAll, w2, w3)
        loghPriorWAll = loghPriorWAll_flexibleHypoSpace(model, hPriorAll, hWeightAll)
    else:
        loghPriorWAll = dict(zip(taskCondKeys, [np.log(hPriorAll[key]) for key in taskCondKeys]))
    # switch policy
    if 'ValueBasedSwitch' in model:
        featureMatAllHypothesesAll = dict.fromkeys(taskCondKeys)
        for informed in [True, False]:
            for numRD in np.arange(numDimensions) + 1:
                featureMatAllHypothesesAll[informed, numRD] = hypothesisToFeatureMat(hypotheses=allHypothesesAll[informed, numRD])
    # calculate cost
    costAll = getCost(cost, allHypothesesAll)
    # choice policy
    if 'Epsilon' in model:
        consistentCHAll = dict.fromkeys(taskCondKeys)
    elif 'SelectMore' in model:
        numMoreDimAll = dict.fromkeys(taskCondKeys)
    for informed in [True, False]:
        for numRD in np.arange(numDimensions) + 1:
            if 'Epsilon' in model:
                consistentCHAll[informed, numRD] = getConsistentCH(allHypothesesAll[informed, numRD])
            elif 'SelectMore' in model:
                numMoreDimAll[informed, numRD] = getNumMoreDim(allHypothesesAll[informed, numRD])

    # create the variables
    game, trial, informedList, numRelevantDimensionsList, reward, numSelectedFeatures, rt, llh = \
        zerosLists(numList=8, lengthList=gameLength * numGamePerType * 6)
    ifRelevantDimension, rewardingFeature, selectedFeature, randomlySelectedFeature, builtFeature = \
        emptyDicts(numDict=5, keys=DIMENSIONS, lengthList=gameLength * numGamePerType * 6)

    # simulation
    iRow = 0
    iGame = 0
    for informed in [True, False]:
        for numRelevantDimensions in np.arange(3) + 1:

            # save game info
            informedList[iRow:(iRow + gameLength * numGamePerType)] = [informed] * gameLength * numGamePerType
            numRelevantDimensionsList[iRow:(iRow + gameLength * numGamePerType)] = [numRelevantDimensions] * gameLength * numGamePerType
            
            if 'DiffThres' in model:
                maxRP = maxRPAll[informed, numRelevantDimensions]

            # hypothesis space
            allHypotheses, NHypos, hPrior, loghPriorW = allHypothesesAll[informed, numRelevantDimensions], NHyposAll[informed, numRelevantDimensions], hPriorAll[informed, numRelevantDimensions], loghPriorWAll[informed, numRelevantDimensions]
            if 'ValueBasedSwitch' in model:
                featureMatAllHypotheses = featureMatAllHypothesesAll[informed, numRelevantDimensions]
            if 'Epsilon' in model:
                consistentCH = consistentCHAll[informed, numRelevantDimensions]
            elif 'SelectMore' in model:
                numMoreDim = numMoreDimAll[informed, numRelevantDimensions]

            # cost
            costThis = costAll[informed, numRD]

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

                # initialization
                choicesThisGame = []
                stimuliThisGame = []
                rewardThisGame = np.zeros(gameLength)
                rewardThisGame[:] = None
                if 'ValueBasedSwitchNoReset' in model:
                    keys = [iTrial for iTrial in range(gameLength)]
                    Q0 = np.zeros(numDimensions * numFeaturesPerDimension)
                    Qfeat = dict(zip(keys, [Q0 for _ in range(len(keys))]))
                elif 'ValueBasedSwitchReset' in model:
                    keys = [(iTrial, lOld) for iTrial in range(gameLength) for lOld in range(iTrial)]
                    Q0 = np.zeros(numDimensions * numFeaturesPerDimension)
                    Qfeat = dict(zip(keys, [Q0 for _ in range(len(keys))]))

                for iTrial in range(gameLength):

                    trial[iRow] = iTrial + 1

                    # the switch model (first term)
                    if 'RandomSwitch' in model:
                        logPhSwitchmodel = calculate_logPhSwitchmodel_randomSwitch(NHypos, iTrial, loghPriorW, pTest)
                    if 'ValueBasedSwitch' in model:
                        if iTrial > 0:
                            featureMatChoice = choiceToFeatureMat([choicesThisGame[iTrial - 1]])[0]
                            featureMatStimulus = stimulusToFeatureMat([stimuliThisGame[iTrial - 1]])[0]
                            if 'NoReset' in model:
                                Qfeat[iTrial] = ((1 - decay) * featureMatStimulus + decay) * Qfeat[iTrial - 1] + (featureMatStimulus * eta_r + featureMatChoice * (eta_s - eta_r)) * (rewardThisGame[iTrial - 1] - np.dot(featureMatStimulus, Qfeat[iTrial - 1]))
                            else:  # 'Reset'
                                for lOld in range(iTrial):
                                    if lOld == 0:
                                        Qfeat[iTrial, lOld] = ((1 - decay) * featureMatStimulus + decay) * Q0 + (featureMatStimulus * eta_r + featureMatChoice * (eta_s - eta_r)) * (rewardThisGame[iTrial - 1] - np.dot(featureMatStimulus, Q0))
                                    else:
                                        Qfeat[iTrial, lOld] = ((1 - decay) * featureMatStimulus + decay) * Qfeat[iTrial - 1, lOld - 1] + (featureMatStimulus * eta_r + featureMatChoice * (eta_s - eta_r)) * (rewardThisGame[iTrial - 1] - np.dot(featureMatStimulus, Qfeat[iTrial - 1, lOld - 1]))
                        if 'NoReset' in model:
                            logPhSwitchmodel = calculate_logPhSwitchmodel_valueBasedNoReset(NHypos, iTrial, featureMatAllHypotheses, Qfeat, loghPriorW, betaSwitch, betaTest, thetaTest, costThis)
                        else:  # 'Reset'
                            logPhSwitchmodel = calculate_logPhSwitchmodel_valueBasedReset(NHypos, iTrial, featureMatAllHypotheses, Qfeat, loghPriorW, betaSwitch, betaTest, thetaTest, costThis)
                        
                    if iTrial == 0:  # the first trial
                        logPh = logPhSwitchmodel

                        # save for use on next trial - part1
                        logPhOldLast = logPhSwitchmodel
                        logPlOldLast = None
                        logPhSwitchmodelLast = None
                        logPlRunlengthmodelLast = None

                    else:

                        # recursive calculation (second and fourth term)
                        logPhOld, logPlOld = calculate_24terms(iTrial, logPhSwitchmodelLast, logPhOldLast, logPlOldLast, logPlRunlengthmodelLast, logPchLast)

                        # the run-length model (third term)
                        if 'DiffThres' in model:
                            thetaStay = maxRP + deltaStay
                        if 'SeparateThres' in model:
                            thetaStay = thetaStayAll[informed, numRD]
                        if 'LRTest' in model:
                            posterior = dict.fromkeys([(iGame, iTrial, lOld) for lOld in range(iTrial)])
                            logp = np.log(hPrior)
                            for lOld in range(iTrial):
                                loglik = np.zeros(NHypos)
                                for iH, h in enumerate(allHypotheses):
                                    pReward = rewardSetting[np.sum(~np.isnan(h)) - 1][np.sum(np.equal(h, stimuliThisGame[iTrial - 1 - lOld]))]
                                    loglik[iH] = np.log(pReward) if rewardThisGame[iTrial - 1 - lOld] else np.log(1 - pReward)
                                logp = loglik + logp
                                logp = logp - logsumexp(logp)
                                posterior[iGame, iTrial, lOld] = np.exp(logp)
                            logPlRunlengthmodel = hypothesisTestingPolicy_LRTest(iTrial, iGame, iTrial, posterior, logPhOld, betaStay, thetaStay)
                        elif 'Counting' in model:
                            estimatedPReward = dict.fromkeys([(iGame, iTrial, lOld) for lOld in range(iTrial)])
                            if not 'CountingAll' in model:
                                for lOld in range(iTrial):
                                    rewardCount = np.zeros(NHypos)
                                    trialCount = np.zeros(NHypos)
                                    for iH, h in enumerate(allHypotheses):
                                        for i in range(iTrial - 1 - lOld, iTrial):
                                            if np.sum([np.isnan(h[iDim]) | (h[iDim] == stimuliThisGame[i][iDim]) for iDim in range(numDimensions)]) == numDimensions:  # compatible trial
                                                rewardCount[iH] += rewardThisGame[i]
                                                trialCount[iH] += 1
                                    estimatedPReward[iGame, iTrial, lOld] = (rewardCount + 1) / (trialCount + 2)
                            else:
                                for lOld in range(iTrial):
                                    rewardCount = np.ones(NHypos) * np.sum(rewardThisGame[iTrial - 1 - lOld:iTrial])
                                    trialCount = np.ones(NHypos) * (lOld + 1)
                                    estimatedPReward[iGame, iTrial, lOld] = (rewardCount + 1) / (trialCount + 2)
                            logPlRunlengthmodel = hypothesisTestingPolicy_Counting(iTrial, iGame, iTrial, estimatedPReward, logPhOld, betaStay, thetaStay)

                        # posterior over hypotheses
                        logPh = logsumexp(logsumexp(logsumexp(logPhSwitchmodel + logPhOld[np.newaxis, np.newaxis, :, :], axis=2) + logPlRunlengthmodel[np.newaxis, :] + logPlOld[np.newaxis, np.newaxis, :], axis=2), axis=1)
                        logPh = logPh - logsumexp(logPh)  # normalize to solve potential numerical deviation from sum to 1

                        # save for use on next trial - part1
                        logPhOldLast = logPhOld
                        logPlOldLast = logPlOld
                        logPhSwitchmodelLast = logPhSwitchmodel
                        logPlRunlengthmodelLast = logPlRunlengthmodel

                    # the choice model: choice probability of the current trial, given all possible hypotheses
                    if 'Epsilon' in model:
                        logPchFull, logpChoice = choicePolicy_epsilon(NChoices, NHypos, logPh, consistentCH, epsilon)
                    elif 'SelectMore' in model:
                        logPchFull, logpChoice = choicePolicy_selectMore(NChoices, NHypos, logPh, numMoreDim, epsilon, kChoice)

                    indChoice = np.random.choice(np.arange(NChoices), size=1, p=np.exp(logpChoice))[0]
                    choice = allChoices[indChoice]
                    choicesThisGame.append(choice)

                    llh[iRow] = logpChoice[indChoice]

                    # save for use on next trial - part2
                    logPchLast = logPchFull[indChoice, :]

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
                    stimuliThisGame.append(stimulus)
                    numSelectedFeatures[iRow] = np.array(
                        [(not pd.isnull(selectedFeature[dim][iRow])) for dim in DIMENSIONS]).sum()
                    numRewardingFeatureBuilt = np.array([((not pd.isnull(rewardingFeature[dim][iRow])) &
                                                          (builtFeature[dim][iRow] == rewardingFeature[dim][iRow]))
                                                         for dim in DIMENSIONS]).sum()
                    reward[iRow] = (np.random.random() < rewardSetting[numRelevantDimensions - 1][numRewardingFeatureBuilt])
                    rewardThisGame[iTrial] = reward[iRow]

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
    
    if ifReturnLlh:
        simudata['llh_cogModel'] = llh

    if ifSaveCSV:
        simudata.to_csv(
            'modelSimulation/' + getModelSimuCSVFileName_inferSerialHypoTesting(model, pars, gameLength, numGamePerType), index=False)

    return simudata



########## individual functions ##########

def getGamesHypoInfo(model):
    allHypothesesAll, NHyposAll, hPriorAll = emptyDicts(numDict=3,
                                                        keys=[(True, 1), (True, 2), (True, 3), (False, 1), (False, 2),
                                                              (False, 3)], lengthList=0)
    for informed in [True, False]:
        for numRD in np.arange(numDimensions) + 1:

            if informed:  # known games
                if 'FullHypo' in model:
                    allHypotheses = [hypothesisSpace[iD][iH] for iD in range(len(hypothesisSpace)) for iH in
                                     range(len(hypothesisSpace[iD]))]
                elif 'ExactHypo' in model:
                    allHypotheses = hypothesisSpace[numRD - 1]
                elif 'SubsetHypo' in model:
                    allHypotheses = [hypothesisSpace[iD][iH] for iD in range(numRD) for iH in
                                     range(len(hypothesisSpace[iD]))]
                elif 'FlexibleHypo' in model:
                    allHypotheses = [hypothesisSpace[iD][iH] for iD in range(len(hypothesisSpace)) for iH in
                                     range(len(hypothesisSpace[iD]))]
                if '+' in model:
                    allHypotheses.insert(0, allChoices[0])
                NHypos = len(allHypotheses)
                hPrior = np.ones(NHypos) / NHypos

            elif not informed:  # unknown games
                allHypotheses = [hypothesisSpace[iD][iH] for iD in range(len(hypothesisSpace)) for iH in
                                 range(len(hypothesisSpace[iD]))]
                if '+' in model:
                    allHypotheses.insert(0, allChoices[0])
                NHypos = len(allHypotheses)
                if 'PriorFlat' in model:
                    hPrior = np.ones(NHypos) / NHypos
                elif 'PriorbyDim' in model:
                    hPrior = np.ones(NHypos) / numDimensions / flatten2Dlist(([
                        [len(hypothesisSpace[i])] * len(hypothesisSpace[i]) for i in np.arange(numDimensions)]))
                else:  # Avg, Separate; will be overwritten later
                    hPrior = np.ones(NHypos) / NHypos

            allHypothesesAll[informed, numRD] = allHypotheses
            NHyposAll[informed, numRD] = NHypos
            hPriorAll[informed, numRD] = hPrior
    
    if 'ThresonTest' in model:
        for informed in [True, False]:
            for numRD in np.arange(numDimensions) + 1:
                allHypothesesAll[informed, numRD].insert(0, allChoices[0])
                NHyposAll[informed, numRD] += 1
                hPriorAll[informed, numRD] = np.concatenate((np.array([0]), hPriorAll[informed, numRD]))
    
    if 'DiffThres' in model:
        maxRPAll = emptyDicts(numDict=1, keys=[(True, 1), (True, 2), (True, 3), (False, 1), (False, 2), (False, 3)], lengthList=0)
        for informed in [True, False]:
            for numRD in np.arange(numDimensions) + 1:
                if not 'ER' in model:  # max reward probability
                    if 'PriorFlat' in model:
                        maxRPAll[informed, numRD] = [maxRewardProb['PriorFlat'][informed, numRD][numDimensions - np.sum(np.isnan(h))] for h in allHypotheses]
                    elif ('PriorbyDim' in model) or ('Avg' in model):
                        maxRPAll[informed, numRD] = [maxRewardProb['PriorbyDim'][informed, numRD][numDimensions - np.sum(np.isnan(h))] for h in allHypotheses]
                else:  # max expected reward probability
                    if ('PriorbyDim' in model) or ('Avg' in model):
                        maxRPAll[informed, numRD] = [maxExpectedRewardProb['PriorbyDim'][informed, numRD][numDimensions - np.sum(np.isnan(h))] for h in allHypotheses]

    if 'DiffThres' not in model:
        return allHypothesesAll, NHyposAll, hPriorAll
    else:
        return allHypothesesAll, NHyposAll, hPriorAll, maxRPAll


def hypothesisWeightKnown_flexibleHypoSpace(model, NHyposAll, w):
    if 'PerDim' not in model:
        wl, wh = w
        wMat = [[1, wh, wh], [wl, 1, wh], [wl, wl, 1]]
    else:
        w1D2, w1D3, w2D1, w2D3, w3D1, w3D2 = w
        wMat = [[1, w1D2, w1D3], [w2D1, 1, w2D3], [w3D1, w3D2, 1]]
    hWeightAll = dict.fromkeys([(True, 1), (True, 2), (True, 3)])
    informed = True  # known games
    for numRD in np.arange(numDimensions) + 1:
        if '+' in model:  # TODO: think about this...
            hWeight = np.array([wl])  # the [nan,nan,nan] hypothesis
        elif 'ThresonTest' in model:
            hWeight = np.array([0])
        else:
            hWeight = np.array([])
        for d in np.arange(numDimensions) + 1:
            hWeight = np.concatenate((hWeight, np.ones(len(hypothesisSpace[d - 1])) * wMat[numRD - 1][d - 1]), axis=None)
        hWeightAll[informed, numRD] = hWeight
    return hWeightAll


def hypothesisWeightUnknown_separate(model, hWeightAll, w2, w3):
    weights = np.array([1, w2, w3]) / np.sum([1, w2 ,w3])
    informed = False
    if 'ThresonTest' in model:
        hWeight = np.array([0])
    else:
        hWeight = np.array([])
    hWeight = np.concatenate((hWeight, flatten2Dlist(([[weights[i] / len(hypothesisSpace[i])] * len(hypothesisSpace[i]) for i in np.arange(numDimensions)]))), axis=None)
    for numRD in np.arange(numDimensions) + 1:
        hWeightAll[informed, numRD] = hWeight
    return hWeightAll


def loghPriorWAll_flexibleHypoSpace(model, hPriorAll, hWeightAll):
    loghPriorWAll = emptyDicts(numDict=1, keys=taskCondKeys, lengthList=0)
    for informed in [True, False]:
        for numRD in np.arange(numDimensions) + 1:
            if informed == True:
                loghPriorW = np.log(hPriorAll[informed, numRD]) + np.log(hWeightAll[informed, numRD])
            else:
                if 'Avg' in model:
                    loghPriorW = np.log(np.mean(np.stack([np.exp(loghPriorWAll[True, d]) for d in np.arange(numDimensions) + 1]), axis=0))
                elif 'Separate' in model:
                    loghPriorW = np.log(hWeightAll[informed, numRD])
                else:  # PriorFlat or PriorbyDim
                    loghPriorW = np.log(hPriorAll[informed, numRD])
            loghPriorW = loghPriorW - logsumexp(loghPriorW)  # normalization
            loghPriorWAll[informed, numRD] = loghPriorW
    return loghPriorWAll


def getCost(cost, allHypothesesAll):
    costAll = dict.fromkeys(taskCondKeys)
    for informed in [True, False]:
        for numRD in np.arange(numDimensions) + 1:
            costAll[informed, numRD] = np.array([(numDimensions - np.sum(np.isnan(h))) * cost for h in allHypothesesAll[informed, numRD]])
    return costAll


def calculate_logPhSwitchmodel_randomSwitch(NHypos, t, loghPriorW, pTest):
    if t == 0:  # the first trial
        logpSwitch = deepcopy(loghPriorW)
        if pTest is not None: # determine whether to test or not
            logpSwitch[0] = np.log(1 - pTest)  # log softmax
            logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
        logPhSwitchmodel = logpSwitch
    else:
        logPhSwitchmodel = np.full((NHypos, t + 1, NHypos, t), np.log(0))
        # stay
        for lOld in range(t):
            lNew = lOld + 1
            for iH in range(NHypos):
                logPhSwitchmodel[iH, lNew, iH, lOld] = np.log(1)
        # switch
        for iHOld in range(NHypos):
            logpSwitch = deepcopy(loghPriorW)  # multiplying probability by weight in the probability space is the same as adding log weight in the log space
            if pTest is None: # models that always test
                logpSwitch[iHOld] = np.log(0)
                logpSwitch = logpSwitch - logsumexp(logpSwitch)  # normalize
            else: # determine whether to test or not
                logpSwitch[0] = np.log(1 - pTest)  # log softmax
                if iHOld > 0: # can't switch to the old hypothesis, but only if it's not [np.nan, np.nan, np.nan]; otherwise, allow keep not testing
                    logpSwitch[iHOld] = np.log(0)
                logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
            logPhSwitchmodel[:, 0, iHOld, :t] = logpSwitch[:, np.newaxis]
    return logPhSwitchmodel


def QfeatNoReset(featureMatChoices, featureMatStimuli, reward, eta_s, eta_r, decay):
    keys = [t for t in range(len(reward))]
    Qfeat = dict(zip(keys, [np.zeros(numDimensions*numFeaturesPerDimension) for _ in range(len(keys))]))
    for t in np.arange(1, len(reward)):
        Qfeat[t] = ((1 - decay) * featureMatStimuli[t - 1, :] + decay) * Qfeat[t - 1] + (featureMatStimuli[t - 1, :] * eta_r + featureMatChoices[t - 1, :] * (eta_s - eta_r)) * (reward[t - 1] - np.dot(featureMatStimuli[t - 1, :], Qfeat[t - 1]))
    return Qfeat


def QfeatReset(featureMatChoices, featureMatStimuli, reward, eta_s, eta_r, decay):
    keys = [(0, 0)] + [(t, lOld) for t in range(len(reward)) for lOld in range(t)]
    Q0 = np.zeros(numDimensions*numFeaturesPerDimension)
    Qfeat = dict(zip(keys, [Q0 for _ in range(len(keys))]))
    for t_start in range(len(reward) - 1):
        for lOld in range(len(reward) - t_start - 1):
            t = t_start + lOld + 1
            if lOld == 0:
                Qfeat[t, lOld] = ((1 - decay) * featureMatStimuli[t - 1, :] + decay) * Q0 + (featureMatStimuli[t - 1, :] * eta_r + featureMatChoices[t - 1, :] * (eta_s - eta_r)) * (reward[t - 1] - np.dot(featureMatStimuli[t - 1, :], Q0))
            else:
                Qfeat[t, lOld] = ((1 - decay) * featureMatStimuli[t - 1, :] + decay) * Qfeat[t - 1, lOld - 1] + (featureMatStimuli[t - 1, :] * eta_r + featureMatChoices[t - 1, :] * (eta_s - eta_r)) * (reward[t - 1] - np.dot(featureMatStimuli[t - 1, :], Qfeat[t - 1, lOld - 1]))
    return Qfeat


def calculate_logPhSwitchmodel_valueBasedNoReset(NHypos, t, featureMatAllHypotheses, Qfeat, loghPriorW, betaSwitch, betaTest, thetaTest, costThis):
    # calculate expected reward for all hypotheses based on Qfeat and costThis
    ExpectedRHypo = np.dot(featureMatAllHypotheses, Qfeat[t]) - costThis
    # consider cost per dimension
    
    # determine p(switch) for all hypotheses except for the currently tested one
    logpSwitchCached = betaSwitch * ExpectedRHypo + loghPriorW  # multiplying probability by weight in the probability space is the same as adding log weight in the log space
    if t == 0:  # the first trial
        logpSwitch = deepcopy(logpSwitchCached)
        if betaTest is None: # models that always test
            logpSwitch = logpSwitch - logsumexp(logpSwitch)  # normalize to 1
        else:
            pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
            logpSwitch[0] = np.log(1 - pTest)  # log softmax
            logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
        logPhSwitchmodel = logpSwitch
    else:
        logPhSwitchmodel = np.full((NHypos, t + 1, NHypos, t), np.log(0))
        # stay
        for lOld in range(t):
            lNew = lOld + 1
            for iH in range(NHypos):
                logPhSwitchmodel[iH, lNew, iH, lOld] = np.log(1)
        # switch
        lNew = 0
        for iHOld in range(NHypos):
            logpSwitch = deepcopy(logpSwitchCached)
            if betaTest is None: # models that always test
                logpSwitch[iHOld] = np.log(0)
                logpSwitch = logpSwitch - logsumexp(logpSwitch)  # normalize to 1
            else: # determine whether to test or not
                pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
                logpSwitch[0] = np.log(1 - pTest)  # log softmax
                if iHOld > 0: # can't switch to the old hypothesis, but only if it's not [np.nan, np.nan, np.nan]; otherwise, allow keep not testing
                    logpSwitch[iHOld] = np.log(0)
                logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
            logPhSwitchmodel[:, lNew, iHOld, :t] = logpSwitch[:, np.newaxis]  # independent of run length
    return logPhSwitchmodel


def calculate_logPhSwitchmodel_valueBasedReset(NHypos, t, featureMatAllHypotheses, Qfeat, loghPriorW, betaSwitch, betaTest, thetaTest, costThis):
    if t == 0:  # the first trial
        ExpectedRHypo = np.dot(featureMatAllHypotheses, Qfeat[0, 0]) - costThis # calculate expected reward for all hypotheses based on Qfeat and costThis
        logpSwitch = betaSwitch * ExpectedRHypo + loghPriorW  # multiplying probability by weight in the probability space is the same as adding log weight in the log space
        if betaTest is None: # models that always test
            logpSwitch = logpSwitch - logsumexp(logpSwitch)  # normalize to 1
        else:
            pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
            logpSwitch[0] = np.log(1 - pTest)  # log softmax
            logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
        logPhSwitchmodel = logpSwitch
    else:
        logPhSwitchmodel = np.full((NHypos, t + 1, NHypos, t), np.log(0))
        # stay
        for lOld in range(t):
            lNew = lOld + 1
            for iH in range(NHypos):
                logPhSwitchmodel[iH, lNew, iH, lOld] = np.log(1)
        # switch
        lNew = 0
        for iHOld in range(NHypos):
            for lOld in range(t):
                # determine p(switch) for all hypotheses except for the currently tested one
                ExpectedRHypo = np.dot(featureMatAllHypotheses, Qfeat[t, lOld]) # calculate expected reward for all hypotheses based on Qfeat
                logpSwitch = betaSwitch * ExpectedRHypo + loghPriorW  # multiplying probability by weight in the probability space is the same as adding log weight in the log space
                if betaTest is None: # models that always test
                    logpSwitch[iHOld] = np.log(0)
                    logpSwitch = logpSwitch - logsumexp(logpSwitch)  # normalize to 1
                else: # determine whether to test or not
                    pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
                    logpSwitch[0] = np.log(1 - pTest)  # log softmax
                    if iHOld > 0: # can't switch to the old hypothesis, but only if it's not [np.nan, np.nan, np.nan]; otherwise, allow keep not testing
                        logpSwitch[iHOld] = np.log(0)
                    logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
                logPhSwitchmodel[:, lNew, iHOld, lOld] = logpSwitch
    return logPhSwitchmodel


def calculate_24terms(t, logPhSwitchmodelLast=None, logPhOldLast=None, logPlOldLast=None, logPlRunlengthmodelLast=None, logPchLast=None):
    if t == 1:

        logPhOld = logPchLast + logPhOldLast
        norm = logsumexp(logPhOld)
        if not np.isinf(norm):
            logPhOld = logPhOld - norm
        logPhOld = logPhOld[:, np.newaxis]

        logPlOld = np.array([0])

    else:

        logtmp = logsumexp(logsumexp(logPhSwitchmodelLast + logPhOldLast[np.newaxis, np.newaxis, :, :], axis=2) + logPlOldLast[np.newaxis, np.newaxis, :], axis=2)

        logPhOld = logPchLast[:, np.newaxis] + logtmp
        norm = logsumexp(logPhOld, axis=0)
        if np.sum(np.isinf(norm)) == 0:
            logPhOld = logPhOld - norm[np.newaxis, :]
        else:
            for i in range(norm.shape[0]):
                if not np.isinf(norm[i]):
                    logPhOld[:, i] = logPhOld[:, i] - norm[np.newaxis, i]

        logPlOld = logsumexp(logPchLast[:, np.newaxis] + logtmp, axis=0) + logsumexp(logPlRunlengthmodelLast + logPlOldLast[np.newaxis, :], axis=1)
        norm = logsumexp(logPlOld)
        if not np.isinf(norm):
            logPlOld = logPlOld - norm
        logPlOld = np.atleast_1d(logPlOld)  # deal with the situation when PlOld turns out to be a scalar

    return logPhOld, logPlOld


# hypothesis tesing policy: likelihood ratio test
def hypothesisTestingPolicy_LRTest(t, iGame, iTrial, posterior, logPhOld, betaStay, thetaStay):
    PlRunlengthmodel = np.zeros((t + 1, t))
    logPlRunlengthmodel = np.log(PlRunlengthmodel)
    for lOld in range(t):
        Ph_m = posterior[iGame, iTrial, lOld]
        LR = np.log(Ph_m / (1 - Ph_m))
        pStayCondH = 1 / (1 + np.exp(- betaStay * (LR - thetaStay)))
        logpStay = logsumexp(np.log(pStayCondH) + logPhOld[:, lOld])
        logpStay = 0 if logpStay > 0 else logpStay  # solve numerical issue
        logpSwitch = np.log(1 - np.exp(logpStay))
        [logpStayNormed, logpSwitchNormed] = [logpStay, logpSwitch] - logsumexp([logpStay, logpSwitch])
        logPlRunlengthmodel[lOld + 1, lOld] = logpStayNormed
        logPlRunlengthmodel[0, lOld] = logpSwitchNormed
    return logPlRunlengthmodel

# hypothesis tesing policy: counting and estimating the probability that the current hypothesis generates rewards
def hypothesisTestingPolicy_Counting(t, iGame, iTrial, estimatedPReward, logPhOld, betaStay, thetaStay):
    PlRunlengthmodel = np.zeros((t + 1, t))
    logPlRunlengthmodel = np.log(PlRunlengthmodel)
    for lOld in range(t):
        pStayCondH = 1 / (1 + np.exp(- betaStay * (estimatedPReward[iGame, iTrial, lOld] - thetaStay)))
        logpStay = logsumexp(np.log(pStayCondH) + logPhOld[:, lOld])
        logpStay = 0 if logpStay > 0 else logpStay  # solve numerical issue
        logpSwitch = np.log(1 - np.exp(logpStay))
        [logpStayNormed, logpSwitchNormed] = [logpStay, logpSwitch] - logsumexp([logpStay, logpSwitch])
        logPlRunlengthmodel[lOld + 1, lOld] = logpStayNormed
        logPlRunlengthmodel[0, lOld] = logpSwitchNormed
    return logPlRunlengthmodel


def calculate_logPh(logPhSwitchmodel, logPhOld, logPlRunlengthmodel, logPlOld):
    logPh = logsumexp(logsumexp(logsumexp(logPhSwitchmodel + logPhOld[np.newaxis, np.newaxis, :, :], axis=2) + logPlRunlengthmodel[np.newaxis, :] + logPlOld[np.newaxis, np.newaxis, :], axis=2), axis=1)
    logPh = logPh - logsumexp(logPh)  # normalize to solve potential numerical deviation from sum to 1
    return logPh


# choice policy: (1) epsilon greedy: a special case of (2) with kChoice=0
def choicePolicy_epsilon(NChoices, NHypos, logPh, consistentCH, epsilon):
    logPchFull = np.log(epsilon / NChoices) * np.ones((NChoices, NHypos))
    logPchFull[consistentCH] = np.log(1 - epsilon + epsilon / NChoices)

    logpChoice = logsumexp(logPchFull + logPh[np.newaxis, :], axis=1)
    logpChoice = logpChoice - logsumexp(logpChoice)

    return logPchFull, logpChoice

# choice policy: (2) allowing for selecting more features than the hypothesis
def choicePolicy_selectMore(NChoices, NHypos, logPh, numMoreDim, epsilon, kChoice):
    kernel = np.exp(kChoice * numMoreDim)
    kernel[np.isnan(numMoreDim)] = 0
    logPchFull = np.log( kernel / np.nansum(kernel, axis=0) * (1 - epsilon) + epsilon / NChoices ) # probability for all the "compatible" choices sum to 1 - epsilon

    logpChoice = logsumexp(logPchFull + logPh[np.newaxis, :], axis=1)
    logpChoice = logpChoice - logsumexp(logpChoice)

    return logPchFull, logpChoice


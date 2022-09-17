import numpy as np
import pandas
from scipy.special import logsumexp

from taskSetting import *
from utilities import *

numDimensions = len(DIMENSIONS)
numFeaturesPerDimension = len(DIMENSIONS_TO_FEATURES[DIMENSIONS[0]])


def prepForFitting_expertRL(data):

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


def contains(x, x_sub):
    return np.sum([np.isnan(x_sub[iDim]) or (x_sub[iDim] == x[iDim]) for iDim in range(numDimensions)]) == 3


def likelihood_expertRL(pars, dataFitting, realdata=True, returnPrediction=False, returnTrialLikelihood=False, returnEstimates=False):

    # parameters
    if len(pars) == 4: # expertRL
        beta, eta, decay, gamma, nu_rev = pars[0], pars[1], 1, pars[2], pars[3]
    elif len(pars) == 5: # expertRLwDecay
        beta, eta, decay, gamma, nu_rev = pars[0], pars[1], pars[2], pars[3]

    # experiment info
    numGame = np.max(dataFitting['game'].values)
    gameLength = np.max(dataFitting['trial'].values)
    llh = np.zeros(dataFitting.shape[0])
    if returnPrediction:
        samplePList = []
    
    # experts
    idx = [[0], [1,2,3], [4,5,6], [7,8,9], [10,11,12,16,17,18,22,23,24], [13,14,15,19,20,21,25,26,27], list(range(28,37)), list(range(37,64))]

    iRow = 0

    for iGame in dataFitting['game'].unique():

        dataThisGame = dataFitting[dataFitting['game'] == iGame]
        rt = dataThisGame['rt'].values
        choiceIndex = dataThisGame['choiceIndex'].values
        stimulusIndex = dataThisGame['stimulusIndex'].values
        reward = dataThisGame['reward'].values

        # initializing Q values
        Q = np.zeros(len(allChoices))
        Q[0] = 0.4
        weight = np.ones(len(idx)) / len(idx)
        RPE = np.zeros(len(idx))
        avg_RPE = np.zeros(len(idx))

        for iTrial in range(gameLength):

            if (realdata == True) & np.isnan(rt[iTrial]):
                if returnPrediction:
                    samplePList.append([np.nan]*len(allChoices))
                llh[iRow] = np.nan

            else:
                # choice phase
                sampleP = np.exp(beta * Q) / np.sum(np.exp(beta * Q))
                for iExpert in range(len(idx)):
                    sampleP[idx[iExpert]] = weight[iExpert] * sampleP[idx[iExpert]]
                sampleP = sampleP / np.sum(sampleP)
                
                if returnPrediction:
                    samplePList.append(sampleP)
                llh[iRow] = np.log(sampleP[int(choiceIndex[iTrial])])

                # learning phase: RW learning                
                stimulus = allStimuli[int(stimulusIndex[iTrial])]
                for iExpert in range(len(idx)):
                    for iChoice in idx[iExpert]:
                        if contains(stimulus, allChoices[iChoice]):
                            RPE[iExpert] = reward[iTrial] - Q[iChoice]
                            if iExpert > 0:
                                Q[iChoice] = Q[iChoice] + weight[iExpert] * eta * RPE[iExpert]
                        else:
                            Q[iChoice] = decay * Q[iChoice]

                # compbine
                avg_RPE = gamma * avg_RPE + (1 - gamma) * RPE**2
                weight = np.exp(- avg_RPE * nu_rev) / np.sum(np.exp(- avg_RPE * nu_rev))
                
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


def getExpectedR(Q, weight, idx_weight):
    Q_weighted = [Q[i] * weight[idx_weight[i]] for i in range(len(idx_weight))]
    expectedR = np.zeros(len(allChoices))
    for iChoice, choice in enumerate(allChoices):
        dims = np.where(~np.isnan(choice))[0]
        if dims.shape[0] == 2: # sum of two 1D features
            f1 = deepcopy(choice)
            f1[dims[1]] = np.nan
            f2 = deepcopy(choice)
            f2[dims[0]] = np.nan
            Vconj = Q_weighted[idx_allChoices[tuple(f1)]] + Q_weighted[idx_allChoices[tuple(f2)]]
            expectedR[iChoice] = (Q_weighted[iChoice] + Vconj) / 2
        elif dims.shape[0] == 3: # sum of three 1D features, or sum of 1x1D+1x2D features
            v_f1 = Q_weighted[idx_allChoices[(choice[0], np.nan, np.nan)]]
            v_f2 = Q_weighted[idx_allChoices[(np.nan, choice[1], np.nan)]]
            v_f3 = Q_weighted[idx_allChoices[(np.nan, np.nan, choice[2])]]
            v_cf1 = Q_weighted[idx_allChoices[(np.nan, choice[1], choice[2])]]
            v_cf2 = Q_weighted[idx_allChoices[(choice[0], np.nan, choice[2])]]
            v_cf3 = Q_weighted[idx_allChoices[(choice[0], choice[1], np.nan)]]
            Vconj = [v_f1 + v_f2 + v_f3, v_f1 + v_cf1, v_f2 + v_cf2, v_f3 + v_cf3]
            expectedR[iChoice] = np.mean(Vconj + [Q_weighted[iChoice]])
        else:
            expectedR[iChoice] = Q_weighted[iChoice]
    return expectedR


def getModelSimuCSVFileName_expertRL(model, beta, eta, decay, gamma, nu_rev, gameLength, numGamePerType):

    betaStr = '_beta' + str(beta)
    etaStr = '_eta' + str(eta)
    decayStr = '_decay' + str(decay)
    gammaStr = '_gamma' + str(gamma)
    nuStr = '_nuRev' + str(nu_rev)

    return model + betaStr + etaStr + decayStr + gammaStr + nuStr + '_gameLength' + str(gameLength) + '_numGamePerType' + str(numGamePerType) + '.csv'

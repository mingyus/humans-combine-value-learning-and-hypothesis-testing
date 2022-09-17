from funcsMyopicBayesian import *
from funcsFeatureRLwDecay import *
from funcsFeatureRLwDecaySameEta import *
from funcsInferSerialHypoTesting import *
from funcsConjunctionRL import *
from funcsExpertRL import *

# model fitting setting
fittingAlgorithm = 'L-BFGS-B'
ifFlatPrior = False
numRep = 10

inferModels = [
    'inferSerialHypoTesting_CountingValueBasedSwitchNoResetDecayEpsilonNoCost_FlexibleHypoAvg',
    'inferSerialHypoTesting_CountingDiffThresValueBasedSwitchNoResetDecayEpsilonNoCost_FlexibleHypoAvg',
    'inferSerialHypoTesting_CountingValueBasedSwitchNoResetDecayThresonTestEpsilonNoCost_FlexibleHypoAvg',
    'inferSerialHypoTesting_CountingRandomSwitchEpsilonNoCost_FlexibleHypoAvg',
    'inferSerialHypoTesting_CountingValueBasedSwitchNoResetDecaySelectMoreNoCost_FlexibleHypoAvg',
    'inferSerialHypoTesting_CountingValueBasedSwitchNoResetDecayThresonTestSelectMoreNoCost_FlexibleHypoAvg',
]

parameterName = {
    'myopicBayesianNoCost': ['beta'],
    'featureRLwDecayNoCost': ['beta', 'eta_s', 'eta_r', 'decay'],
    'featureRLSameEtaNoCost': ['beta', 'eta'],
    'conjunctionRLNoCost': ['beta', 'eta'],
    'conjunctionRLNoCostFlexChoice': ['beta', 'eta'],
    'expertRL': ['beta', 'eta', 'gamma', 'nu_rev'],
}


def getModelSimuCSVFileName(model, p, ifFlatPrior, gameLength, numGamePerType):
    if model == 'myopicBayesianNoCost':
        beta = p[0]
        weightReward = 1
        fileName = getModelSimuCSVFileName_myopicBayesian(beta=beta, cost=0, weightReward=weightReward, ifFlatPrior=ifFlatPrior,
                                                          gameLength=gameLength, numGamePerType=numGamePerType)
    elif model == 'featureRLwDecayNoCost':
        beta, eta_s, eta_r, decay = p[0], p[1], p[2], p[3]
        fileName = getModelSimuCSVFileName_featureRLwDecay(beta, eta_s, eta_r, 0, decay, gameLength, numGamePerType)
    elif 'inferSerialHypoTesting' in model:
        fileName = getModelSimuCSVFileName_inferSerialHypoTesting(model, p, gameLength, numGamePerType)
    return fileName


bounds = {
    'myopicBayesianNoCost': [(0, np.inf)],
    'featureRLwDecayNoCost': [(0, np.inf), (0, 1), (0, 1), (0, 1)],
    'featureRLSameEtaNoCost': [(0, np.inf), (0, 1)],
    'conjunctionRLNoCost': [(0, np.inf), (0, 1)],
    'conjunctionRLNoCostFlexChoice': [(0, np.inf), (0, 1)],
    'expertRL': [(0, np.inf), (0, 1), (0, 1), (0, np.inf)],  # beta, eta, gamma, nu_rev
}


def getP0(model):
    if model == 'myopicBayesianNoCost':
        return np.array([np.random.random() * 5])
    elif model == 'featureRLwDecayNoCost':
        return np.array([np.random.random() * 5, np.random.random(), np.random.random(), np.random.random()])
    elif model == 'featureRLSameEtaNoCost':
        return np.array([np.random.random() * 5, np.random.random()])
    elif model in ['conjunctionRLNoCost',  'conjunctionRLNoCostFlexChoice']:
        return np.array([np.random.random() * 5, np.random.random()])
    elif model == 'expertRL':
        return np.array([np.random.random() * 5, np.random.random(), np.random.random(), np.random.random()]) # beta, eta, gamma, nu_rev
    elif 'inferSerialHypoTesting' in model:
        p0 = np.array([])
        # betaStay, thetaStay
        if 'LRTest' in model:
            p0 = np.concatenate((p0, np.array([np.random.random() + 1, -5 + np.random.random()])))
        elif 'Counting' in model:
            if 'DiffThres' in model:
                p0 = np.concatenate((p0, np.array([np.random.random() * 5 + 5, np.random.random() * 0.5 - 0.25])))
            elif 'SeparateThres' in model:
                p0 = np.concatenate((p0, np.array([np.random.random() * 5 + 5, np.random.random() * 0.5 + 0.25, np.random.random() * 0.5 + 0.25, np.random.random() * 0.5 + 0.25, np.random.random() * 0.5 + 0.25])))
            else:
                p0 = np.concatenate((p0, np.array([np.random.random() * 5 + 5, np.random.random() * 0.5 + 0.25])))
        # eta(s), (decay), betaSwitch for value-based switch
        if 'ValueBasedSwitch' in model:
            if 'DiffEta' not in model:
                p0 = np.concatenate((p0, np.array([np.random.random()]))) # eta
            else:
                p0 = np.concatenate((p0, np.array([np.random.random(), np.random.random()]))) # eta_s, eta_r
            if 'Decay' in model:
                p0 = np.concatenate((p0, np.array([np.random.random()]))) # decay
            p0 = np.concatenate((p0, np.array([np.random.random() * 5 + 5]))) # betaSwitch
        # threshold on testing
        if 'ThresonTest' in model:
            if 'RandomSwitch' in model:
                p0 = np.concatenate((p0, np.array([1 - np.random.random() * 0.3]))) # pTest
            elif 'ValueBasedSwitch' in model:
                p0 = np.concatenate((p0, np.array([np.random.random() * 5 + 5, np.random.random() * 0.3]))) # betaTest, thetaTest
        # epsilon
        p0 = np.concatenate((p0, np.array([np.random.random() * 0.3])))
        # kChoice for select-more
        if 'SelectMore' in model:
            p0 = np.concatenate((p0, np.array([np.random.random() * 5])))
        # cost per dimension
        if 'Cost' in model and 'NoCost' not in model:
            p0 = np.concatenate((p0, np.array([np.random.random() * 0.2])))
        # hypothesis space parameters
        if 'FlexibleHypo' in model:
            if 'PerDim' not in model:
                p0 = np.concatenate((p0, np.array([np.random.random(), np.random.random()])))
            else:
                p0 = np.concatenate((p0, np.array([np.random.random(), np.random.random(), np.random.random(), np.random.random(), np.random.random(), np.random.random()])))
        if 'Separate' in model:
            p0 = np.concatenate((p0, np.array([np.random.random()*2+1, np.random.random()*2+1])))
        return p0
    elif model == 'WSLS':
        return np.array([np.random.random(), np.random.random()])


prepDataFunction = {
    'myopicBayesianNoCost': prepForFitting_myopicBayesian,
    'featureRLwDecayNoCost': prepForFitting_featureRLwDecay,
    'featureRLSameEtaNoCost': prepForFitting_featureRLwDecaySameEta,
    'conjunctionRLNoCost': prepForFitting_conjunctionRL,
    'conjunctionRLNoCostFlexChoice': prepForFitting_conjunctionRL,
    'expertRL': prepForFitting_expertRL,
}

likelihoodFunction = {
    'myopicBayesianNoCost': likelihood_myopicBayesianNoCost,
    'featureRLwDecayNoCost': likelihood_featureRLwDecayNoCost,
    'featureRLSameEtaNoCost': likelihood_featureRLwDecaySameEtaNoCost,
    'conjunctionRLNoCost': likelihood_conjuctionRLwDecayNoCost,
    'conjunctionRLNoCostFlexChoice': likelihood_conjuctionRLwDecayNoCostFlexChoice,
    'expertRL': likelihood_expertRL,
}

modelFunction = {
    'myopicBayesianNoCost': model_myopicBayesian,
    'featureRLwDecayNoCost': model_featureRLwDecay,
}


for model in inferModels:
    eps = 1e-6

    prepDataFunction[model] = prepForFitting_inferSerialHypoTesting
    likelihoodFunction[model] = likelihood_inferSerialHypoTesting
    modelFunction[model] = model_inferSerialHypoTesting

    # parameterName and bounds
    parNames = []
    parBounds = []
    # betaStay, theta
    parNames = parNames + ['betaStay']
    if 'DiffThres' in model:
        parNames = parNames + ['deltaStay']
    elif 'SeparateThres' in model:
        parNames = parNames + ['thetaStay1D'] + ['thetaStay2D'] + ['thetaStay3D'] + ['thetaStayUnknown']
    else:
        parNames = parNames + ['thetaStay']
    if 'LRTest' in model:
        parBounds = parBounds + [(0, np.inf), (-np.inf, np.inf)]
    elif 'Counting' in model:
        if 'DiffThres' in model:
            parBounds = parBounds + [(0, np.inf), (-0.8, 0.6)]
        elif 'SeparateThres' in model:
            parBounds = parBounds + [(0, np.inf), (0, 1), (0, 1), (0, 1), (0, 1)]
        else:
            parBounds = parBounds + [(0, np.inf), (0, 1)]
    # eta, betaSwitch for value-based switch
    if 'ValueBasedSwitch' in model:
        if 'DiffEta' not in model:
            parNames = parNames + ['eta']
            parBounds = parBounds + [(0, 1)]
        else:
            parNames = parNames + ['eta_s', 'eta_r']
            parBounds = parBounds + [(0, 1), (0, 1)]
        if 'Decay' in model:
            parNames = parNames + ['decay']
            parBounds = parBounds + [(0, 1)]
        parNames = parNames + ['betaSwitch']
        parBounds = parBounds + [(0, np.inf)]
    # thetaTest
    if 'ThresonTest' in model:
        if 'RandomSwitch' in model:
            parNames = parNames + ['pTest']
            parBounds = parBounds + [(0, 1)]
        elif 'ValueBasedSwitch' in model:
            parNames = parNames + ['betaTest', 'thetaTest']
            parBounds = parBounds + [(0, np.inf), (-np.inf, np.inf)]
    # epsilon
    parNames = parNames + ['epsilon']
    parBounds = parBounds + [(eps, 1)]
    # kChoice for select-more
    if 'SelectMore' in model:
        parNames = parNames + ['kChoice']
        parBounds = parBounds + [(-np.inf, np.inf)]
    # cost per dimension
    if 'Cost' in model and 'NoCost' not in model:
        parNames = parNames + ['cost']
        parBounds = parBounds + [(-np.inf, np.inf)]
    # hypothesis space parameters
    if 'FlexibleHypo' in model:
        if 'PerDim' not in model:
            parNames = parNames + ['wl', 'wh']
            parBounds = parBounds + [(0, np.inf), (0, np.inf)]
        else:
            parNames = parNames + ['w1D2', 'w1D3', 'w2D1', 'w2D3', 'w3D1', 'w3D2']
            parBounds = parBounds + [(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)]
    if 'Separate' in model:
        parNames = parNames + ['w2', 'w3']
        parBounds = parBounds + [(0, np.inf), (0, np.inf)]
    parameterName[model] = parNames
    bounds[model] = parBounds


def getFittingFileName(ifCV, iCVgame=None, iRep=None, model=None, ifFlatPrior=None, workerId=None, fittingAlgorithm=None):
    fileNameFitResults = ''

    if 'Bayesian' not in model:
        fileNameFitResults += model + '_' + str(workerId)
    else:
        priorStr = 'priorFlat' if ifFlatPrior else 'priorNormalizedForDim'
        fileNameFitResults += model + '_' + priorStr + '_' + str(workerId)

    if ifCV:
        if iRep is None:
            fileNameFitResults += '_iCVgame' + str(iCVgame) + '_' + fittingAlgorithm + '.csv'
        else:
            fileNameFitResults += '_iCVgame' + str(iCVgame) +  '_iRep' + str(iRep) + '_' + fittingAlgorithm + '.csv'
    else:
        if iRep is None:
            fileNameFitResults += '_' + fittingAlgorithm + '.csv'
        else:
            fileNameFitResults += '_iRep' + str(iRep) + '_' + fittingAlgorithm + '.csv'

    return fileNameFitResults


def getFittingCollectFileName(model=None, expVersion=None, ifFlatPrior=None, fittingAlgorithm=None, CV=None, workerId=None):
    fileNameCollectResults = ('' if (CV is None) else ('CVTotal_' if CV == 'total' else 'CV_')) + model + '_' + expVersion + ('' if workerId is None else ('_' + str(workerId)))

    if 'Bayesian' not in model:
        fileNameCollectResults += ''
    else:
        priorStr = 'priorFlat' if ifFlatPrior else 'priorNormalizedForDim'
        fileNameCollectResults += '_' + priorStr

    fileNameCollectResults += '_' + fittingAlgorithm + '.csv'

    return fileNameCollectResults


def getSimulationwParticiantsParFileName(model=None, ifFlatPrior=None, workerId=None, fittingAlgorithm=None):
    fileName = ''

    if 'Bayesian' not in model:
        fileName += model
    else:
        priorStr = 'priorFlat' if ifFlatPrior else 'priorNormalizedForDim'
        fileName += model + '_' + priorStr

    fileName += '_' + str(workerId) + '_' + fittingAlgorithm + '.csv'

    return fileName

import numpy as np
from itertools import compress
from copy import deepcopy
from utilities import *

DIMENSIONS = ('color', 'shape', 'pattern')
DIMENSIONS_TO_FEATURES = {
    'color': ['red','blue','green'],
    'shape': ['square','circle','triangle'],
    'pattern': ['plaid','dots','waves']
}
FEATURE_TO_DIMENSION = {
    'red': 'color',
    'blue': 'color',
    'green': 'color',
    'square': 'shape',
    'circle': 'shape',
    'triangle': 'shape',
    'plaid': 'pattern',
    'dots': 'pattern',
    'waves': 'pattern',
}

# task setting
numDimensions = len(DIMENSIONS)
numFeaturesPerDimension = len(DIMENSIONS_TO_FEATURES[DIMENSIONS[0]])
rewardSetting = [[0.2,0.8],[0.2,0.5,0.8],[0.2,0.4,0.6,0.8]]

maxRewardProb = {  # the highest possible reward probability, conditioned on the number of features selected
    'PriorFlat': {
        (True, 1): [0, 0.8, 0.8, 0.8],
        (True, 2): [0, 0.5, 0.8, 0.8],
        (True, 3): [0, 0.4, 0.6, 0.8],
        (False, 1): [0, 0.5, 5/7, 0.8],
        (False, 2): [0, 0.5, 5/7, 0.8],
        (False, 3): [0, 0.5, 5/7, 0.8],
    },
    'PriorbyDim':{
        (True, 1): [0, 0.8, 0.8, 0.8],
        (True, 2): [0, 0.5, 0.8, 0.8],
        (True, 3): [0, 0.4, 0.6, 0.8],
        (False, 1): [0, 17/30, 11/15, 0.8],
        (False, 2): [0, 17/30, 11/15, 0.8],
        (False, 3): [0, 17/30, 11/15, 0.8],
    }   
}

maxExpectedRewardProb = {  # the highest possible reward probability, conditioned on the number of features selected
    'PriorbyDim':{
        (True, 1): [0.4, 0.8, 0.8, 0.8],
        (True, 2): [0.4, 0.6, 0.8, 0.8],
        (True, 3): [0.4, 8/15, 2/3, 0.8],
        (False, 1): [0.4, 29/45, 34/45, 0.8],
        (False, 2): [0.4, 29/45, 34/45, 0.8],
        (False, 3): [0.4, 29/45, 34/45, 0.8],
    }
}

hypothesisSpace = [
    # 1D-relevant
    [[0,np.nan,np.nan],[1,np.nan,np.nan],[2,np.nan,np.nan],[np.nan,0,np.nan],[np.nan,1,np.nan],[np.nan,2,np.nan],[np.nan,np.nan,0],[np.nan,np.nan,1],[np.nan,np.nan,2]],
    # 2D-relevant
    [[0,0,np.nan],[0,1,np.nan],[0,2,np.nan],[0,np.nan,0],[0,np.nan,1],[0,np.nan,2],
     [1,0,np.nan],[1,1,np.nan],[1,2,np.nan],[1,np.nan,0],[1,np.nan,1],[1,np.nan,2],
     [2,0,np.nan],[2,1,np.nan],[2,2,np.nan],[2,np.nan,0],[2,np.nan,1],[2,np.nan,2],
     [np.nan,0,0],[np.nan,0,1],[np.nan,0,2],[np.nan,1,0],[np.nan,1,1],[np.nan,1,2],[np.nan,2,0],[np.nan,2,1],[np.nan,2,2],
    ],
    # 3D-relevant
    [[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,1,2],[0,2,0],[0,2,1],[0,2,2],
     [1,0,0],[1,0,1],[1,0,2],[1,1,0],[1,1,1],[1,1,2],[1,2,0],[1,2,1],[1,2,2],
     [2,0,0],[2,0,1],[2,0,2],[2,1,0],[2,1,1],[2,1,2],[2,2,0],[2,2,1],[2,2,2],
    ]
]

# we assume that the model always have the option of all possible choices
allChoices = [
    # doesn't select anything
    [np.nan,np.nan,np.nan],
    # select on 1 dimension
    [0,np.nan,np.nan],[1,np.nan,np.nan],[2,np.nan,np.nan],[np.nan,0,np.nan],[np.nan,1,np.nan],[np.nan,2,np.nan],[np.nan,np.nan,0],[np.nan,np.nan,1],[np.nan,np.nan,2],
    # select on 2 dimensions
    [0,0,np.nan],[0,1,np.nan],[0,2,np.nan],[0,np.nan,0],[0,np.nan,1],[0,np.nan,2],
    [1,0,np.nan],[1,1,np.nan],[1,2,np.nan],[1,np.nan,0],[1,np.nan,1],[1,np.nan,2],
    [2,0,np.nan],[2,1,np.nan],[2,2,np.nan],[2,np.nan,0],[2,np.nan,1],[2,np.nan,2],
    [np.nan,0,0],[np.nan,0,1],[np.nan,0,2],[np.nan,1,0],[np.nan,1,1],[np.nan,1,2],[np.nan,2,0],[np.nan,2,1],[np.nan,2,2],
    # select on 3 dimensions
    [0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,1,2],[0,2,0],[0,2,1],[0,2,2],
    [1,0,0],[1,0,1],[1,0,2],[1,1,0],[1,1,1],[1,1,2],[1,2,0],[1,2,1],[1,2,2],
    [2,0,0],[2,0,1],[2,0,2],[2,1,0],[2,1,1],[2,1,2],[2,2,0],[2,2,1],[2,2,2],
]

idx_allChoices = dict(zip([tuple(choice) for choice in allChoices], list(range(len(allChoices)))))

allChoices_fullSelection = [
    [0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,1,2],[0,2,0],[0,2,1],[0,2,2],
    [1,0,0],[1,0,1],[1,0,2],[1,1,0],[1,1,1],[1,1,2],[1,2,0],[1,2,1],[1,2,2],
    [2,0,0],[2,0,1],[2,0,2],[2,1,0],[2,1,1],[2,1,2],[2,2,0],[2,2,1],[2,2,2],
]

allStimuli = [[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,1,2],[0,2,0],[0,2,1],[0,2,2],
     [1,0,0],[1,0,1],[1,0,2],[1,1,0],[1,1,1],[1,1,2],[1,2,0],[1,2,1],[1,2,2],
     [2,0,0],[2,0,1],[2,0,2],[2,1,0],[2,1,1],[2,1,2],[2,2,0],[2,2,1],[2,2,2],
]

def choiceToStimuli(choice, allStimuli = allStimuli):
    dimSelected = np.where(~np.isnan(choice))[0]
    boolStimuli = [np.sum(np.equal(np.array(stimulus)[dimSelected],np.array(choice)[dimSelected]))==len(dimSelected) for stimulus in allStimuli]
    idxStimuli = np.where(boolStimuli)[0]
    stimuli = list(compress(deepcopy(allStimuli), boolStimuli))
    return stimuli, idxStimuli


def choiceToFeatureMat(choices=allChoices):
    featureMat = np.zeros([len(choices), numDimensions * numFeaturesPerDimension])
    for iChoice, choice in enumerate(choices):
        for iDim in range(numDimensions):
            if np.isnan(choice[iDim]):  # code nan as 3
                featureMat[iChoice, (iDim * numFeaturesPerDimension):((iDim + 1) * numFeaturesPerDimension)] = 1 / numFeaturesPerDimension
            else:
                featureMat[iChoice, iDim * numFeaturesPerDimension + choice[iDim]] = 1
    return featureMat


def stimulusToFeatureMat(stimuli=allStimuli):
    featureMat = np.zeros([len(stimuli), numDimensions * numFeaturesPerDimension])
    for iStimulus, stimulus in enumerate(stimuli):
        for iDim in range(numDimensions):
            featureMat[iStimulus, iDim * numFeaturesPerDimension + stimulus[iDim]] = 1
    return featureMat


def hypothesisToFeatureMat(hypotheses):
    featureMat = np.zeros([len(hypotheses), numDimensions * numFeaturesPerDimension])
    for iHypothesis, hypothesis in enumerate(hypotheses):
        for iDim in range(numDimensions):
            if np.isnan(hypothesis[iDim]):  # code nan as 3
                featureMat[iHypothesis, (iDim * numFeaturesPerDimension):((iDim + 1) * numFeaturesPerDimension)] = 1 / numFeaturesPerDimension
            else:
                featureMat[iHypothesis, iDim * numFeaturesPerDimension + hypothesis[iDim]] = 1
    return featureMat


def getConsistentCH(allHypotheses):
    return np.array([[np.sum([(np.isnan(c[i]) & np.isnan(h[i])) | (c[i] == h[i]) for i in range(numDimensions)]) == numDimensions for h in allHypotheses] for c in allChoices])


def getNumMoreDim(allHypotheses):
    compatibleChoice = np.array([[np.sum([np.isnan(h[i]) | (h[i] == c[i]) for i in range(numDimensions)]) == numDimensions for h in allHypotheses] for c in allChoices])
    numDiffDim = np.array([[len(c) - np.sum([(np.isnan(c[i]) & np.isnan(h[i])) | (c[i] == h[i]) for i in range(numDimensions)]) for h in allHypotheses] for c in allChoices])
    numMoreDim = np.empty(compatibleChoice.shape)
    numMoreDim[:] = np.nan
    numMoreDim[compatibleChoice] = numDiffDim[compatibleChoice]
    return numMoreDim

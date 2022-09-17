import numpy as np
import pandas as pd
from taskSetting import *
from expsInfo import *
from scipy.stats import binom

expVersion = 'all'

def load_data(expVersion, getWorkerIds=False, acc=0.46):
    # load data
    data = pd.read_csv('data/data_' + expVersion + '_wClickInfo.csv')
    
    # calcualte numRewardingFeaturesSelected and expectedReward
    numRewardingFeaturesSelected = pd.concat([(~data['rewardingFeature_'+dim].isnull() & \
        (data['selectedFeature_'+dim] == data['rewardingFeature_'+dim])).astype(int) for dim in DIMENSIONS], axis = 1, keys = DIMENSIONS).sum(axis = 1).values
    numRelevantDimensionsUnselected = pd.concat([(~data['rewardingFeature_'+dim].isnull() & data['selectedFeature_'+dim].isnull()).astype(int) for dim in DIMENSIONS], axis = 1, keys = DIMENSIONS).sum(axis = 1).values
    rt = data['rt'].values
    numRelevantDimensions = data['numRelevantDimensions'].values
    p = np.zeros((numDimensions+1,numDimensions+1)) * np.nan
    for n in range(numDimensions+1):
        for k in range(n+1):
            p[n,k] = binom.pmf(k, n, p=1/3)
    data['add_expectedReward'] = [0.4 if np.isnan(rt[iRow]) else np.sum([p[int(numRelevantDimensionsUnselected[iRow]), i] *
                            rewardSetting[int(numRelevantDimensions[iRow]) - 1][numRewardingFeaturesSelected[iRow] + i]
                            for i in range(int(numRelevantDimensionsUnselected[iRow])+1)]) for iRow in range(data.shape[0])]
    data['add_numRewardingFeaturesSelected'] = numRewardingFeaturesSelected
    data.loc[np.isnan(rt), 'add_numRewardingFeaturesSelected'] = np.nan
    
    # exclude participants based on accuracy criterion
    if acc is not None:
        avgReward = data.groupby(['workerId']).mean()['add_expectedReward']
        workerIds = avgReward.index[avgReward >= acc].tolist()
        data = data[data['workerId'].isin(workerIds)]
    else:
        workerIds = data['workerId'].unique().tolist()

    data = data.reset_index(drop=True)

    # return data and/or workerIds
    if getWorkerIds:
        return data, workerIds
    else:
        return data
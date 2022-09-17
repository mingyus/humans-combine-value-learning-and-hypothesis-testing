import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

from taskSetting import *

changeTypeStr = ['add','drop','switch_within','switch_across','mixed']
changeTypeLabels = ['add','drop','within','across','mixed']

def mark_choiceChange(data):
    # exclude no-response trials
    data_valid = data[~pd.isnull(data['rt'])].copy()
    data_valid = data_valid.reset_index(drop=True)
    # find change points of choices
    choiceChange = (np.sum([(data_valid['selectedFeature_' + dim].iloc[1:].values == data_valid['selectedFeature_' + dim].iloc[:-1].values) 
                            | (pd.isnull(data_valid['selectedFeature_' + dim].iloc[1:]).values & pd.isnull(data_valid['selectedFeature_' + dim].iloc[:-1]).values) for dim in DIMENSIONS],axis=0)<3).astype(np.float)
    choiceChange[data_valid['trial'].iloc[1:].values-data_valid['trial'].iloc[:-1].values<0] = np.nan  # the first trial of a game (not including the first game)
    # put it back to data
    data.loc[~pd.isnull(data['rt']),'add_choiceChange'] = np.concatenate((np.nan, choiceChange), axis = None)
    data.loc[pd.isnull(data['rt']),'add_choiceChange'] = None
    data.loc[~pd.isnull(data['rt']),'add_beforeChoiceChange'] = np.concatenate((choiceChange, np.nan), axis = None)
    data.loc[pd.isnull(data['rt']),'add_beforeChoiceChange'] = None
    return data


def get_sameChoiceLength(data):
    # exclude no-response trials
    data_valid = data[~pd.isnull(data['rt'])].copy()
    data_valid = data_valid.reset_index(drop=True)
    # find change points of choices
    data_valid['add_choiceChange'] = np.concatenate((1, (np.sum([(data_valid['selectedFeature_' + dim].iloc[1:].values == data_valid['selectedFeature_' + dim].iloc[:-1].values) 
                            | (pd.isnull(data_valid['selectedFeature_' + dim].iloc[1:]).values & pd.isnull(data_valid['selectedFeature_' + dim].iloc[:-1]).values) 
                            for dim in DIMENSIONS],axis=0)<3).astype(np.float)), axis = None)
    data_valid.loc[np.where(data_valid['trial'].values[1:]-data_valid['trial'].values[:-1]<0)[0]+1, 'add_choiceChange'] = 1
    data_valid['add_sameChoiceLength'] = np.nan
    data_valid.loc[data_valid['add_choiceChange']==1, 'add_sameChoiceLength'] = np.concatenate((np.where(data_valid['add_choiceChange'])[0][1:] - np.where(data_valid['add_choiceChange'])[0][:-1], len(data_valid) - np.where(data_valid['add_choiceChange'])[0][-1]), axis = None)
    # the first trial of a game
    data_valid.loc[0, 'add_choiceChange'] = np.nan
    data_valid.loc[np.where(data_valid['trial'].values[1:]-data_valid['trial'].values[:-1]<0)[0]+1, 'add_choiceChange'] = np.nan
    # put it back to data
    data['add_sameChoiceLength'] = np.nan
    data.loc[~pd.isnull(data['rt']), 'add_sameChoiceLength'] = data_valid['add_sameChoiceLength'].values
    return data

        
def get_runLength_rewardRate(data):
    if 'add_choiceChange' not in data.columns:
        data = mark_choiceChange(data)
    rt = data['rt'].values
    trial = data['trial'].values
    reward = data['reward'].values
    choiceChange = data['add_choiceChange'].values
    rewardRate = np.empty(len(data)) * np.nan
    runLength = np.empty(len(data)) * np.nan
    for i in range(len(data)):
        if trial[i] == 1:
            reward_count = 0
            trial_count = 0
        if not np.isnan(rt[i]):
            rewardRate[i] = reward_count/trial_count if trial_count > 0 else np.nan
            runLength[i] = trial_count
            if choiceChange[i] == 1:
                reward_count = 0
                trial_count = 0
            reward_count += reward[i]
            trial_count += 1
    data['add_rewardRate'] = rewardRate
    data['add_runLength'] = runLength
    return data


def get_choiceChange_info(data):
    # exclude no-response trials
    data_valid = data[~pd.isnull(data['rt'])].copy().reset_index(drop=True)
    # find change points of choices
    choiceChange = (np.sum([(data_valid['selectedFeature_' + dim].iloc[1:].values == data_valid['selectedFeature_' + dim].iloc[:-1].values) 
                            | (pd.isnull(data_valid['selectedFeature_' + dim].iloc[1:]).values & pd.isnull(data_valid['selectedFeature_' + dim].iloc[:-1]).values) for dim in DIMENSIONS],axis=0)<3).astype(np.float)
    choiceChange[data_valid['trial'].iloc[1:].values-data_valid['trial'].iloc[:-1].values<0] = np.nan  # the first trial of a game (not including the first game)
    data_valid.loc[:,'add_choiceChange'] = np.concatenate((np.nan, choiceChange), axis = None)
    data_valid.loc[:,'add_beforeChoiceChange'] = np.concatenate((choiceChange, np.nan), axis = None)
    ## record last choice
    iLastChoice = [allChoices.index([np.nan if pd.isnull(data_valid.loc[iLast, 'selectedFeature_' + dim]) else DIMENSIONS_TO_FEATURES[dim].index(data_valid.loc[iLast, 'selectedFeature_' + dim]) for dim in DIMENSIONS]) for iLast in range(len(data_valid)-1)]
    data_valid.loc[:,'add_iLastChoice'] = np.concatenate((np.nan, iLastChoice), axis = None)

    ## count the number of features changed in each choice
    numDimensionsChange = 3.0 - np.sum([(data_valid['selectedFeature_' + dim].iloc[1:].values == data_valid['selectedFeature_' + dim].iloc[:-1].values)
                                        | (pd.isnull(data_valid['selectedFeature_' + dim].iloc[1:]).values & pd.isnull(data_valid['selectedFeature_' + dim].iloc[:-1]).values) for dim in DIMENSIONS], axis=0)
    numDimensionsChange[data_valid['trial'].iloc[1:].values-data_valid['trial'].iloc[:-1].values<0] = np.nan  # the first trial of a game (not including the first game)
    data_valid.loc[:,'add_numDimensionsChange'] = np.concatenate((np.nan, numDimensionsChange), axis = None)

    ## label the type of choice change
    # if there is no choice change, mark as nan
    for dim in DIMENSIONS:
        data_valid.loc[:,'add_choiceChangeType_'+dim] = np.nan
    data_valid.loc[:,'add_choiceChangeType_overall'] = np.nan
    data_valid.loc[:,'add_choiceChangeTypeCount'] = np.nan
    data_valid.loc[:,'add_choiceChangeTypeList'] = np.nan
    data_valid.loc[:,'add_choiceChangewSame'] = 0
    data_valid.loc[data_valid['add_choiceChange']==False,'add_choiceChangewSame'] = np.nan
    data_valid['add_choiceChangeTypeList'] = data_valid['add_choiceChangeTypeList'].astype(object)

    for irow in tqdm(np.where(data_valid['add_choiceChange']==True)[0]):
        choiceChangeTypeList = []
        for dim in DIMENSIONS:
            if (pd.isnull(data_valid.loc[irow-1,'selectedFeature_'+dim])) & (pd.isnull(data_valid.loc[irow,'selectedFeature_'+dim])):
                data_valid.loc[irow,'add_choiceChangeType_'+dim] = 'no_choice'
            elif data_valid.loc[irow-1,'selectedFeature_'+dim] == data_valid.loc[irow,'selectedFeature_'+dim]:
                data_valid.loc[irow,'add_choiceChangeType_'+dim] = 'same'
                data_valid.loc[irow,'add_choiceChangewSame'] = 1
            elif (pd.isnull(data_valid.loc[irow-1,'selectedFeature_'+dim])) & (~pd.isnull(data_valid.loc[irow,'selectedFeature_'+dim])):
                data_valid.loc[irow,'add_choiceChangeType_'+dim] = 'add'
                choiceChangeTypeList.append('add')
            elif (~pd.isnull(data_valid.loc[irow-1,'selectedFeature_'+dim])) & (pd.isnull(data_valid.loc[irow,'selectedFeature_'+dim])):
                data_valid.loc[irow,'add_choiceChangeType_'+dim] = 'drop'
                choiceChangeTypeList.append('drop')
            elif data_valid.loc[irow-1,'selectedFeature_'+dim] != data_valid.loc[irow,'selectedFeature_'+dim]:
                data_valid.loc[irow,'add_choiceChangeType_'+dim] = 'switch_within'
                choiceChangeTypeList.append('switch_within')
        data_valid.loc[irow,'add_choiceChangeTypeCount'] = len(choiceChangeTypeList)

        if len(set(choiceChangeTypeList)) == 1: # only one type of changes
            data_valid.loc[irow,'add_choiceChangeType_overall'] = choiceChangeTypeList[0]
        elif (data_valid.loc[irow,'add_choiceChangeTypeCount'] == 2) & np.isin('add',choiceChangeTypeList) & np.isin('drop',choiceChangeTypeList):
            data_valid.loc[irow,'add_choiceChangeType_overall'] = 'switch_across'
        else:
            data_valid.loc[irow,'add_choiceChangeType_overall'] = 'mixed'

        choiceChangeTypeList.sort()
        data_valid.at[irow,'add_choiceChangeTypeList'] = '-'.join(choiceChangeTypeList)
        
        # change in # of rewarding features
        data_valid.loc[irow,'add_numRewardingFeaturesSelectedChange'] = data_valid.loc[irow,'add_numRewardingFeaturesSelected'] - data_valid.loc[irow-1,'add_numRewardingFeaturesSelected']
    
    ## assign values to original dataframe
    colNames = ['add_choiceChange', 'add_beforeChoiceChange', 'add_iLastChoice', 'add_numDimensionsChange'] + ['add_choiceChangeType_'+dim for dim in DIMENSIONS] + ['add_choiceChangeType_overall', 'add_choiceChangeTypeCount', 'add_choiceChangeTypeList', 'add_choiceChangewSame', 'add_numRewardingFeaturesSelectedChange']
    for col in colNames:
        data.loc[~pd.isnull(data['rt']), col] = data_valid[col].values
        data.loc[pd.isnull(data['rt']), col] = np.nan
        
    return data


def get_choiceChangeTypeMat():
    choiceChangeType = pd.DataFrame(np.zeros((len(allChoices),len(allChoices))))
    choiceChangewSame = np.zeros((len(allChoices),len(allChoices)))
    for iChoice1, choice1 in enumerate(allChoices):
        for iChoice2, choice2 in enumerate(allChoices):
            if choice2 == choice1:
                choiceChangeType.iloc[iChoice1, iChoice2] = np.nan
            else:
                choiceChangeTypeList = []
                for iDim, dim in enumerate(DIMENSIONS):
                    if (np.isnan(choice1[iDim])) & (np.isnan(choice2[iDim])):
                        continue
                    elif choice1[iDim] == choice2[iDim]:
                        choiceChangewSame[iChoice1, iChoice2] = 1
                    elif (np.isnan(choice1[iDim])) & (not np.isnan(choice2[iDim])):
                        choiceChangeTypeList.append('add')
                    elif (not np.isnan(choice1[iDim])) & (np.isnan(choice2[iDim])):
                        choiceChangeTypeList.append('drop')
                    elif choice1[iDim] != choice2[iDim]:
                        choiceChangeTypeList.append('switch_within')
                if len(set(choiceChangeTypeList)) == 1: # only one type of changes
                    choiceChangeType.iloc[iChoice1, iChoice2] = choiceChangeTypeList[0]
                elif (len(choiceChangeTypeList) == 2) & np.isin('add',choiceChangeTypeList) & np.isin('drop',choiceChangeTypeList):
                    choiceChangeType.iloc[iChoice1, iChoice2] = 'switch_across'
                else:
                    choiceChangeType.iloc[iChoice1, iChoice2] = 'mixed'
    return choiceChangeType, choiceChangewSame
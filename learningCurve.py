import numpy as np
import pandas
from pandas import DataFrame, concat, read_csv
from statsmodels.stats.anova import AnovaRM
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import comb
from scipy.stats import binom
from scipy.stats import ttest_rel

from taskSetting import *
from expsInfo import *


def plot_learning_curves(ax, varName, learning_curves, wIndividualCurves, plotColor, reference_curves, reference_legend, reference_color, game_length, numRelevantDimension, fontsize=16, linewidth=5):

    trial_index = np.arange(game_length)+1
    
    # individual learning curves
    if wIndividualCurves:
        for learning_curve in learning_curves:
            ax.plot(trial_index, learning_curve, alpha=0.2, color=plotColor, label='_nolegend_')

    # average (and sem) learning curves
    if all([len(curve) == 0 for curve in learning_curves]):
        return None, None
    elif np.isscalar(learning_curves[0]):
        varPlot = ax.plot(trial_index, learning_curves, color=plotColor, lw=linewidth)[0]
    else:
        average_values = np.squeeze(np.nanmean(np.stack(learning_curves, axis=1),axis=1))
        sem_values = np.squeeze(np.nanstd(np.stack(learning_curves, axis=1),axis=1)/np.sqrt(np.stack(learning_curves, axis=1).shape[1]))
        varPlot = ax.plot(trial_index, average_values, color=plotColor, lw=linewidth)[0]
        ax.fill_between(trial_index, average_values - sem_values, average_values + sem_values, lw=0, alpha=0.3, color=plotColor)
    
    # reference curves (mean and sem)
    if reference_curves is None:
        refPlot = None
    elif np.isscalar(reference_curves):
        refPlot = ax.plot(trial_index, [reference_curves]*game_length, color=reference_color, ls='--')[0] #, lw=linewidth
    elif np.isscalar(reference_curves[0]):
        refPlot = ax.plot(trial_index, reference_curves, color=reference_color, ls='--')[0] #, lw=linewidth
    else:
        average_reference = np.squeeze(np.nanmean(np.stack(reference_curves, axis=1),axis=1))
        sem_reference = np.squeeze(np.nanstd(np.stack(reference_curves, axis=1),axis=1)/np.sqrt(np.stack(reference_curves, axis=1).shape[1]))
        refPlot = ax.plot(trial_index, average_reference, color=reference_color, ls='--')[0] #, lw=linewidth
        ax.fill_between(trial_index, average_reference - sem_reference, average_reference + sem_reference, lw=0, alpha=0.3, color=reference_color)
    
    # ax settings
    ylimValues = {
        'NumSelected': [0, 3],
        'ExpectedReward': [0.35, 0.85],
        'numDimensionsChange': [0, 3],
        'numDimensionsChangeCondChoiceChange': [0, 3],
        'typeChoiceChange': [0, 0.5],
        'typeFeaturesLearningCurve': [0, 2],
    }
    ax.set_ylim(ylimValues[varName][0],ylimValues[varName][1])
    ax.tick_params(axis='both', labelsize=fontsize, pad=10)
    
    return varPlot, refPlot


def get_chance_curve(empiricalChance, current_df, varName, numRelevantDimension, game_length, ifdata = True):
    
    if varName in ['NumSelected', 'numDimensionsChange', 'numDimensionsChangeCondChoiceChange']:
        return [np.nan]*game_length
    
    elif varName in ['ExpectedReward']:
        return [0.4] * game_length


def get_learning_curve(current_df, varName, numRelevantDimension, ifdata = True):

    if varName in ['ExpectedReward']:
        current_df['add_numFeatureSelected'] = concat([(~current_df['selectedFeature_'+dim].isnull()).astype(int) for dim in DIMENSIONS], axis = 1, keys = DIMENSIONS).sum(axis = 1)
        current_df['add_numRewardingFeaturesSelected'] = concat([(~current_df['rewardingFeature_'+dim].isnull() & \
            (current_df['selectedFeature_'+dim] == current_df['rewardingFeature_'+dim])).astype(int) \
            for dim in DIMENSIONS], axis = 1, keys = DIMENSIONS).sum(axis = 1)
        current_df['add_expectedNumRewardingFeaturesBuilt'] = current_df['add_numRewardingFeaturesSelected'] + concat([(~current_df['rewardingFeature_'+dim].isnull() & \
            current_df['selectedFeature_'+dim].isnull()).astype(int) for dim in DIMENSIONS], axis = 1, keys = DIMENSIONS).sum(axis = 1)/3
        tmp_df = current_df.reset_index(drop=True)
        tmp_df['numRelevantDimensionsUnselected'] = concat([(~tmp_df['rewardingFeature_'+dim].isnull() & tmp_df['selectedFeature_'+dim].isnull()).astype(int) for dim in DIMENSIONS], axis = 1, keys = DIMENSIONS).sum(axis = 1)
        current_df['add_expectedReward'] = [np.sum([binom.pmf(k=i, n=tmp_df.loc[iRow, 'numRelevantDimensionsUnselected'], p=1/3) *
                                                    rewardSetting[int(tmp_df.loc[iRow, 'numRelevantDimensions']-1)][int(tmp_df.loc[iRow, 'add_numRewardingFeaturesSelected']) + i]
                                                    for i in range(int(tmp_df.loc[iRow, 'numRelevantDimensionsUnselected']+1))]) for iRow in range(tmp_df.shape[0])]

    elif varName == 'numDimensionsChange':
        tmp_df = current_df[~current_df['rt'].isnull()].copy().reset_index(drop=True)
        numDimensionsChange = 3.0 - np.sum([(tmp_df['selectedFeature_' + dim].iloc[1:].values == tmp_df['selectedFeature_' + dim].iloc[:-1].values)
                                         | (pandas.isnull(tmp_df['selectedFeature_' + dim].iloc[1:]).values & pandas.isnull(tmp_df['selectedFeature_' + dim].iloc[:-1]).values)
                                         for dim in DIMENSIONS], axis=0)
        numDimensionsChange[tmp_df['trial'].iloc[1:].values - tmp_df['trial'].iloc[:-1].values < 0] = np.nan  # the first trial of a game (not including the first game)
        current_df.loc[~current_df['rt'].isnull(), 'add_numDimensionsChange'] = np.concatenate((np.nan, numDimensionsChange), axis=None)

    elif varName == 'numDimensionsChangeCondChoiceChange':
        tmp_df = current_df[~current_df['rt'].isnull()].copy().reset_index(drop=True)
        choiceChange = (np.sum([(tmp_df['selectedFeature_' + dim].iloc[1:].values == tmp_df['selectedFeature_' + dim].iloc[:-1].values)
                        | (pandas.isnull(tmp_df['selectedFeature_' + dim].iloc[1:]).values & pandas.isnull(tmp_df['selectedFeature_' + dim].iloc[:-1]).values)
                        for dim in DIMENSIONS], axis=0) < 3).astype(np.float)
        choiceChange[tmp_df['trial'].iloc[1:].values - tmp_df['trial'].iloc[:-1].values < 0] = np.nan  # the first trial of a game (not including the first game)
        current_df.loc[~current_df['rt'].isnull(), 'add_choiceChange'] = np.concatenate((np.nan, choiceChange), axis=None)
        current_df.loc[current_df['rt'].isnull(), 'add_choiceChange'] = np.nan
        numDimensionsChange = 3.0 - np.sum([tmp_df['selectedFeature_' + dim].iloc[1:].values ==
                                         tmp_df['selectedFeature_' + dim].iloc[:-1].values for dim in DIMENSIONS],axis=0)
        numDimensionsChange[tmp_df['trial'].iloc[1:].values - tmp_df['trial'].iloc[:-1].values < 0] = np.nan  # the first trial of a game (not including the first game)
        current_df.loc[~current_df['rt'].isnull(), 'add_numDimensionsChange'] = np.concatenate((np.nan, numDimensionsChange), axis=None)
        current_df.loc[current_df['rt'].isnull(), 'add_numDimensionsChange'] = np.nan
        current_df.loc[(~current_df['rt'].isnull()) & (current_df['add_choiceChange'] == True), 'add_numDimensionsChangeCondChoiceChange'] = current_df.loc[(~current_df['rt'].isnull()) & (current_df['add_choiceChange'] == True), 'add_numDimensionsChange']
        current_df.loc[(~current_df['rt'].isnull()) & (current_df['add_choiceChange'] == False), 'add_numDimensionsChangeCondChoiceChange'] = np.nan

    returnVarName = {
        'NumSelected': 'numSelectedFeatures',
        'ExpectedReward': 'add_expectedReward',
        'numDimensionsChange': 'add_numDimensionsChange',
        'numDimensionsChangeCondChoiceChange': 'add_numDimensionsChangeCondChoiceChange'
    }
    if ifdata:
        current_df.loc[current_df['rt'].isnull(),returnVarName[varName]] = np.nan
    return current_df.groupby('trial').agg({returnVarName[varName]:np.nanmean})[returnVarName[varName]].values, current_df[returnVarName[varName]]


def selectedFeatureTypeCounts(df, period):
    # selecting correct rewarding feature
    df['add_numRewardingFeatureSelected'] = concat([(~df['rewardingFeature_'+dim].isnull() & \
        (df['selectedFeature_'+dim] == df['rewardingFeature_'+dim])).astype(int) \
        for dim in DIMENSIONS], axis=1, keys = DIMENSIONS).sum(axis=1)
    # selecting wrong feature on the relevant dimension
    df['add_numWrongFeatureSelected'] = concat([(~df['rewardingFeature_'+dim].isnull() & \
        ~df['selectedFeature_'+dim].isnull() & \
        (df['selectedFeature_'+dim] != df['rewardingFeature_'+dim])).astype(int) \
        for dim in DIMENSIONS], axis=1, keys = DIMENSIONS).sum(axis=1)
    # selecting a feature on the irrelevant dimension
    df['add_numIrrelevantFeatureSelected'] = concat([(df['rewardingFeature_'+dim].isnull() & \
        ~df['selectedFeature_'+dim].isnull() ).astype(int) \
        for dim in DIMENSIONS], axis=1, keys = DIMENSIONS).sum(axis=1)
    if period == 'gameEnd':
        return [df.mean()[col].values for col in ['add_numRewardingFeatureSelected','add_numWrongFeatureSelected','add_numIrrelevantFeatureSelected']]
    else: # 'learning curve'
        return [df.groupby('trial').agg({col:np.nanmean})[col].values for col in
                ['add_numRewardingFeatureSelected','add_numWrongFeatureSelected','add_numIrrelevantFeatureSelected']]


def choiceChangeTypeCounts(df):
    # exclude no-response trials
    data_valid = df[~df['rt'].isnull()].copy()
    data_valid = data_valid.reset_index(drop=True)

    # find change points of choices
    choiceChange = (np.sum(
        [(data_valid['selectedFeature_' + dim].iloc[1:].values == data_valid['selectedFeature_' + dim].iloc[:-1].values)
         | (pandas.isnull(data_valid['selectedFeature_' + dim].iloc[1:]).values & pandas.isnull(
            data_valid['selectedFeature_' + dim].iloc[:-1]).values)
         for dim in DIMENSIONS], axis=0) < 3).astype(np.float)
    choiceChange[data_valid['trial'].iloc[1:].values - data_valid['trial'].iloc[
                                                       :-1].values < 0] = np.nan  # the first trial of a game (not including the first game)
    data_valid.loc[:, 'add_choiceChange'] = np.concatenate((np.nan, choiceChange), axis=None)

    # count the number of features changed in each choice
    numDimensionsChange = 3.0 - np.sum(
        [(data_valid['selectedFeature_' + dim].iloc[1:].values == data_valid['selectedFeature_' + dim].iloc[:-1].values)
         | (pandas.isnull(data_valid['selectedFeature_' + dim].iloc[1:]).values & pandas.isnull(
            data_valid['selectedFeature_' + dim].iloc[:-1]).values)
         for dim in DIMENSIONS], axis=0)
    data_valid.loc[:, 'add_numDimensionsChange'] = np.concatenate((np.nan, numDimensionsChange), axis=None)

    # label the type of choice change
    # if there is no choice change, mark as nan
    for dim in DIMENSIONS:
        data_valid.loc[:, 'add_choiceChangeType_' + dim] = np.nan
    data_valid.loc[:, 'add_choiceChangeType_overall'] = np.nan
    data_valid.loc[:, 'add_choiceChangeTypeCount'] = np.nan
    data_valid.loc[:, 'add_choiceChangeTypeList'] = np.nan
    data_valid['add_choiceChangeTypeList'] = data_valid['add_choiceChangeTypeList'].astype(object)

    for irow in np.where(data_valid['add_choiceChange'] == True)[0]:
        choiceChangeTypeList = []
        for dim in DIMENSIONS:
            if (data_valid.loc[irow - 1, 'selectedFeature_' + dim] == data_valid.loc[
                irow, 'selectedFeature_' + dim]) | (
                    (pandas.isnull(data_valid.loc[irow - 1, 'selectedFeature_' + dim])) & (
            pandas.isnull(data_valid.loc[irow, 'selectedFeature_' + dim]))):
                data_valid.loc[irow, 'add_choiceChangeType_' + dim] = np.nan
            elif (pandas.isnull(data_valid.loc[irow - 1, 'selectedFeature_' + dim])) & (
            ~pandas.isnull(data_valid.loc[irow, 'selectedFeature_' + dim])):
                data_valid.loc[irow, 'add_choiceChangeType_' + dim] = 'add'
                choiceChangeTypeList.append('add')
            elif (~pandas.isnull(data_valid.loc[irow - 1, 'selectedFeature_' + dim])) & (
            pandas.isnull(data_valid.loc[irow, 'selectedFeature_' + dim])):
                data_valid.loc[irow, 'add_choiceChangeType_' + dim] = 'drop'
                choiceChangeTypeList.append('drop')
            elif data_valid.loc[irow - 1, 'selectedFeature_' + dim] != data_valid.loc[irow, 'selectedFeature_' + dim]:
                data_valid.loc[irow, 'add_choiceChangeType_' + dim] = 'switch_within'
                choiceChangeTypeList.append('switch_within')
        data_valid.loc[irow, 'add_choiceChangeTypeCount'] = len(choiceChangeTypeList)

        if len(set(choiceChangeTypeList)) == 1:  # only one type of changes
            data_valid.loc[irow, 'add_choiceChangeType_overall'] = choiceChangeTypeList[0]
        elif (data_valid.loc[irow, 'add_choiceChangeTypeCount'] == 2) & np.isin('add', choiceChangeTypeList) & np.isin(
                'drop', choiceChangeTypeList):
            data_valid.loc[irow, 'add_choiceChangeType_overall'] = 'switch_across'
        else:
            data_valid.loc[irow, 'add_choiceChangeType_overall'] = 'mixed'

        choiceChangeTypeList.sort()
        data_valid.at[irow, 'add_choiceChangeTypeList'] = '-'.join(choiceChangeTypeList)

    # assign the values to the original df
    df.loc[~df['rt'].isnull(), 'add_choiceChangeType_overall'] = data_valid['add_choiceChangeType_overall'].values
    df.loc[df['rt'].isnull(), 'add_choiceChangeType_overall'] = np.nan
    df.loc[~df['rt'].isnull(), 'add_choiceChangeTypeList'] = data_valid['add_choiceChangeTypeList'].values
    df.loc[df['rt'].isnull(), 'add_choiceChangeTypeList'] = np.nan

    typeChoiceChangeList = ['add', 'drop', 'switch_within', 'switch_across', 'mixed']
    for typeChoiceChange in typeChoiceChangeList:
        df.loc[~((df['rt'].isnull()) | (df['trial'] == 1)), 'add_choiceChangeType_overall_' + typeChoiceChange] \
            = (df.loc[~((df['rt'].isnull()) | (df['trial'] == 1)), 'add_choiceChangeType_overall'] == typeChoiceChange).astype(float)
        df.loc[(df['rt'].isnull()) | (df['trial'] == 1), 'add_choiceChangeType_overall_' + typeChoiceChange] = np.nan

    return [df.groupby('trial').agg({col: np.nanmean})[col].values for col in
            ['add_choiceChangeType_overall_' + typesChoiceChange for typesChoiceChange in typeChoiceChangeList]]


def plotLearningCurves(varName, data, plotType='seperate', wIndividualCurves=False, empiricalChance=True, wLegend=True, showFigure=False, printcsv=False, runANOVA=False, expVersion=None, wTitleAxLabel=True, fontsize=20, xticklabels=[0, 10, 20, 30], ifPublish=True, plotChance=True, wTitle=False):

    plt.rcParams.update({'font.size': fontsize})
    linewidth = 2

    ## Get exp info
    # Get the list of participants and their worker IDs
    if 'workerId' in data.keys():
        workerIds = data['workerId'].unique()
    else:  # simulated data
        workerIds = [0]
        data['workerId'] = 0

    # Get game variables
    gameLength = DataFrame.max(data['trial'])
    numGamePerType = int(DataFrame.max(data['game']) / 3 / 2)
    trialIndex = np.arange(gameLength)

    ## Plotting
    if varName in ['NumSelected', 'ExpectedReward', 'numDimensionsChange', 'numDimensionsChangeCondChoiceChange']:

        title = {
            'NumSelected': '# features selected',
            'ExpectedReward': 'Reward probability',
            'numDimensionsChange': '# features changed (all trials)',
            'numDimensionsChangeCondChoiceChange': '# features changed (choice change trials only)',
        }
        
        if ifPublish:
            fig, axes = plt.subplots(1, 3, sharex=True, figsize=(8, 2.5))
            fontsize = 12.5
            plt.rcParams.update({'font.size': fontsize})
            linewidth = 2
            axes_linewidth = 1 #1.3
        else:
            fig, axes = plt.subplots(1, 3, sharex=True, figsize=(20, 12 / 3) if wLegend else (18, 12 / 3))
            linewidth = 5
            
        dfANOVAList_all = []
        for numRelevantDimensions in np.arange(3) + 1:

            if plotType == 'collapsed':
                if varName == 'pTrueHypo':
                    Warning('Not supported')
                    return
                # collapsed over informed and uninformed games
                learning_curves_collapsed = []
                chance_curves_collapsed = []
                for participant, workerId in enumerate(workerIds):
                    current_df = data[
                        (data['workerId'] == workerId) & (data['numRelevantDimensions'] == numRelevantDimensions)].copy()
                    learning_curves_collapsed.append(get_learning_curve(current_df, varName, numRelevantDimensions))
                    chance_curves_collapsed.append(
                        get_chance_curve(empiricalChance, current_df, varName, numRelevantDimensions, gameLength))
                ax = axes[numRelevantDimensions - 1]
                varPlot, refPlot = plot_learning_curves(ax, varName, learning_curves_collapsed, wIndividualCurves, 'purple',
                                                        chance_curves_collapsed if plotChance else None, 'Chance', 'black', gameLength,
                                                        numRelevantDimensions, fontsize=fontsize, linewidth=linewidth)
            else:
                varPlots = []
                refPlots = []
                dfANOVAList = []
                # for informed and uninformed games separately
                for idx, informed in enumerate([True, False]):
                    learning_curves = []
                    chance_curves = []
                    individualDataList = []
                    for participant, workerId in enumerate(workerIds):
                        current_df = data[(data['workerId'] == workerId) & (data['numRelevantDimensions'] == numRelevantDimensions) & (
                                        data['informed'] == informed)].copy()
                        if current_df.shape[0] > 0:
                            learning_curve, individualData = get_learning_curve(current_df, varName, numRelevantDimensions)
                            learning_curves.append(learning_curve)
                            individualDataList.append(individualData)
                            chance_curves.append(
                                get_chance_curve(empiricalChance, current_df, varName, numRelevantDimensions, gameLength))
                    ax = axes[numRelevantDimensions - 1]
                    if varName not in ['NumSelected', 'ifChoiceChange', 'numDimensionsChange',
                                   'numDimensionsChangeCondChoiceChange', 'pTrueHypo', 'Reward', 'Built', 'ExpectedBuilt',  'ExpectedReward']:
                        varPlot, refPlot = plot_learning_curves(ax, varName, learning_curves, wIndividualCurves, 'red' if informed else 'blue',
                                                            chance_curves if plotChance else None, 'Chance', 'maroon' if informed else 'darkblue',
                                                            gameLength, numRelevantDimensions, fontsize=fontsize, linewidth=linewidth)
                    else:
                        varPlot, refPlot = plot_learning_curves(ax, varName, learning_curves, wIndividualCurves,
                                                                'red' if informed else 'blue',
                                                                chance_curves if plotChance else None, 'Chance', 'black',
                                                                gameLength, numRelevantDimensions, fontsize=fontsize, linewidth=linewidth)
                    varPlots.append(varPlot)
                    refPlots.append(refPlot)

                    if runANOVA:
                        df = DataFrame({'dv': flatten2Dlist(individualDataList), 
                                        'participant': flatten2Dlist([[workerId] * gameLength * numGamePerType for workerId in workerIds]),
                                        'trial': flatten2Dlist([np.arange(gameLength) + 1] * len(workerIds) * numGamePerType),
                                        'informed': [informed] * gameLength * len(workerIds) * numGamePerType,
                                        'numRelevantDimensions': [numRelevantDimensions] * gameLength * len(workerIds) * numGamePerType
                                       })
                        dfANOVAList.append(df)
                        dfANOVAList_all.append(df)

                if runANOVA:
                    # note that ANOVA is run on the mean of each participant (only one value, i.e. the average over observations, per participant per condition in fed into ANOVA)
                    # with statsmodel.AnovaRM, you can either use the mean directly or provide individual data points and define aggregate_func as mean/np.nanmean
                    dfANOVA = pandas.concat(dfANOVAList)
                    aovrm = AnovaRM(data=dfANOVA, depvar='dv', subject='participant', within=['informed', 'trial'], aggregate_func=np.nanmean)
                    result = aovrm.fit()
                    print('Repeated measures ANOVA: ' + str(numRelevantDimensions) + 'D\n', result)
                    if printcsv:
                        dfANOVA.to_csv('Rcode/' + varName + '_' + str(numRelevantDimensions) + 'D.csv')
                    
                    t, p = ttest_rel(dfANOVA[dfANOVA['informed']==True].groupby('participant').mean()['dv'], dfANOVA[dfANOVA['informed']==False].groupby('participant').mean()['dv'])
                    print('Paired t-test: ' + str(numRelevantDimensions) + 'D')
                    print('t = ' + str(round(t,2)) + '; p = ' + str(round(p,4)))

                sns.despine()
        
        if runANOVA:
            dfANOVA = pandas.concat(dfANOVAList_all)
#             aovrm = AnovaRM(data=dfANOVA, depvar='dv', subject='participant', within=['informed', 'numRelevantDimensions', 'trial'], aggregate_func=np.nanmean)
#             result = aovrm.fit()
#             print('Repeated measures ANOVA (three-way):', result)
            if printcsv:
                dfANOVA.to_csv('Rcode/' + varName + '.csv')
                
        if ifPublish:
            for ax in axes:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', length=3.5, labelsize=fontsize, pad=4.5, width=axes_linewidth)
                ax.set_xlim([0, 30])
                xticklabels = [0, 15, 30]
                ax.set_xticks(xticklabels)
                ax.set_xticklabels(xticklabels)
                if wTitleAxLabel:
                    ax.set_xlabel('Trial', labelpad=8, fontsize=fontsize)
                    if not wTitle:
                        axes[0].set_ylabel(title[varName], fontsize=fontsize)
                    else:
                        fig.suptitle(title[varName], fontsize=fontsize+0.5, y=0.9)
                ax.spines['bottom'].set_linewidth(axes_linewidth)
                ax.spines['left'].set_linewidth(axes_linewidth)
                    
            plt.subplots_adjust(wspace=0.4, top=0.75, bottom=0.25)
            # plt.tight_layout()
            # plt.show()
        else:
            for ax in axes:
                ax.set_xticks(xticklabels)
                ax.set_xticklabels(xticklabels)
            if wTitleAxLabel:
                fig.text(0.5, -0.05, 'Trial', ha='center', fontsize=fontsize)
                fig.text(0.5, 1, title[varName], ha='center', fontsize=fontsize)
            if wLegend:
                if plotType == 'collapsed':
                    fig.legend(handles=[varPlot, refPlot], labels=['Participants', 'Chance'], loc="center right", fontsize=fontsize,
                               frameon=False)
                else:
                    fig.legend(handles=[varPlots[0], varPlots[1], refPlots[0], refPlots[1]],
                               labels=['Known', 'Unknown', 'Chance for known', 'Chance for unknown'], fontsize=fontsize,
                               loc="center right", frameon=False)
                plt.subplots_adjust(right=0.825, wspace=0.3)
            else:
                plt.subplots_adjust(wspace=0.3)

    elif varName == 'typeFeaturesLearningCurve':
        
        if ifPublish:
            fig, axes = plt.subplots(2, 3, sharex=True, figsize=(8, 5))
            fontsize = 12.5
            plt.rcParams.update({'font.size': fontsize})
            linewidth = 2
            axes_linewidth = 1 #1.3
        else:
            fig, axes = plt.subplots(2, 3, sharex=True, figsize=(20, 12 / 3) if wLegend else (18, 12 / 3))
            linewidth = 5

        for numRelevantDimension in np.arange(3) + 1:
            # for informed and uninformed games separately
            for idx, informed in enumerate([True, False]):
                rewarding_curves = []
                wrong_curves = []
                irrelevant_curves = []
                for participant, workerId in enumerate(workerIds):
                    current_df = data[
                        (data['workerId'] == workerId) & (data['informed'] == informed) & (
                                data['numRelevantDimensions'] == numRelevantDimension)].copy()
                    learning_curves = selectedFeatureTypeCounts(current_df, 'learningCurve')
                    rewarding_curves.append(learning_curves[0])
                    wrong_curves.append(learning_curves[1])
                    irrelevant_curves.append(learning_curves[2])
                ax = axes[idx, numRelevantDimension - 1]
                p = [None] * 3
                p[0], _ = plot_learning_curves(ax, varName, rewarding_curves, wIndividualCurves, 'green', np.nan, '', None, gameLength,
                                     numRelevantDimension, fontsize=fontsize, linewidth=linewidth)
                p[1], _ = plot_learning_curves(ax, varName, wrong_curves, wIndividualCurves, 'orange', np.nan, '', None, gameLength,
                                     numRelevantDimension, fontsize=fontsize, linewidth=linewidth)
                p[2], _ = plot_learning_curves(ax, varName, irrelevant_curves, wIndividualCurves, 'gray', np.nan, '', None, gameLength,
                                     numRelevantDimension, fontsize=fontsize, linewidth=linewidth)
            
        if ifPublish:
            for numRelevantDimension in np.arange(3) + 1:
                for idx, informed in enumerate([True, False]):
                    ax = axes[idx, numRelevantDimension - 1]
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(axis='both', length=3.5, labelsize=fontsize, pad=4.5, width=axes_linewidth)
                    ax.set_xlim([0, 30])
                    xticklabels = [0, 15, 30]
                    ax.set_xticks(xticklabels)
                    ax.set_xticklabels(xticklabels)
                    if not informed:
                        ax.set_xlabel('Trial', labelpad=8, fontsize=fontsize)
                    else:
                        ax.set_title(f'{numRelevantDimension}D-relevant', fontsize=fontsize)
                    if numRelevantDimension == 1:
                        ax.set_ylabel('Known' if informed else 'Unknown', labelpad=8, fontsize=fontsize)
                        
                    ax.spines['bottom'].set_linewidth(axes_linewidth)
                    ax.spines['left'].set_linewidth(axes_linewidth)
            fig.suptitle('Feature selection type', fontsize=fontsize+0.5, y=0.9)
            plt.subplots_adjust(wspace=0.4, top=0.75, bottom=0.25)
            if wLegend:
                fig.legend(handles=p, labels=['Correct feature', 'Incorrect feature', 'False positive'], bbox_to_anchor=(1, 0.75), frameon=False)
        else:
            for numRelevantDimension in np.arange(3) + 1:
                for idx, informed in enumerate([True, False]):
                    ax = axes[idx, numRelevantDimension - 1]
                    ax.set_ylim(0, 3)
                    if numRelevantDimension == 1:
                        ax.set_ylabel('Known' if informed else 'Unknown', fontsize=fontsize)
                    if idx == 0:
                        ax.set_xlabel(str(numRelevantDimension) + 'D relevant', fontsize=fontsize)
                        ax.xaxis.set_label_position('top')
                    ax.tick_params(axis='both', labelsize=fontsize)
            if wTitleAxLabel:
                fig.text(0.5, 0.05, 'Trial', ha='center', fontsize=fontsize)
                fig.text(0.5, 0.95, '# features selected per type', ha='center', fontsize=fontsize)
            if wLegend:
                fig.legend(handles=p, labels=['Correct feature', 'Incorrect feature', 'False positive'], loc="center right")
            
    elif varName == 'typeChoiceChange':

        if ifPublish:
            fig, axes = plt.subplots(2, 3, sharex=True, figsize=(8, 5))
            fontsize = 12.5
            plt.rcParams.update({'font.size': fontsize})
            linewidth = 2
            axes_linewidth = 1 #1.3
        else:
            fig, axes = plt.subplots(2, 3, sharex=True, figsize=(20, 12 / 3) if wLegend else (18, 12 / 3))
            linewidth = 5

        for numRelevantDimension in np.arange(3) + 1:
            # for informed and uninformed games separately
            for idx, informed in enumerate([True, False]):
                all_curves = [None] * 5
                p = [None] * 5
                for iCurve in range(5):
                    all_curves[iCurve] = []
                for participant, workerId in enumerate(workerIds):
                    current_df = data[
                        (data['workerId'] == workerId) & (data['informed'] == informed) & (
                                data['numRelevantDimensions'] == numRelevantDimension)].copy()
                    learning_curves = choiceChangeTypeCounts(current_df)
                    for iCurve in range(5):
                        all_curves[iCurve].append(learning_curves[iCurve])
                ax = axes[idx, numRelevantDimension - 1]
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                for iCurve in range(5):
                    p[iCurve], _ = plot_learning_curves(ax, varName, all_curves[iCurve], wIndividualCurves, colors[iCurve],
                                         np.nan, '', None, gameLength, numRelevantDimension, fontsize=fontsize, linewidth=linewidth)
                
        if ifPublish:
            for numRelevantDimension in np.arange(3) + 1:
                for idx, informed in enumerate([True, False]):
                    ax = axes[idx, numRelevantDimension - 1]
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(axis='both', length=3.5, labelsize=fontsize, pad=4.5, width=axes_linewidth)
                    ax.set_xlim([0, 30])
                    xticklabels = [0, 15, 30]
                    ax.set_xticks(xticklabels)
                    ax.set_xticklabels(xticklabels)
                    if not informed:
                        ax.set_xlabel('Trial', labelpad=8, fontsize=fontsize)
                    else:
                        ax.set_title(f'{numRelevantDimension}D-relevant', fontsize=fontsize)
                    if numRelevantDimension == 1:
                        ax.set_ylabel('Known' if informed else 'Unknown', labelpad=8, fontsize=fontsize)
                    ax.spines['bottom'].set_linewidth(axes_linewidth)
                    ax.spines['left'].set_linewidth(axes_linewidth)
            fig.suptitle('Choice change type', fontsize=fontsize+0.5, y=0.9)
            plt.subplots_adjust(wspace=0.4, top=0.75, bottom=0.25)
            if wLegend:
                fig.legend(handles=p, labels=['Add', 'Drop', 'Switch within dimension', 'Switch across dimensions', 'Other'], bbox_to_anchor=(1, 0.75), frameon=False)
        else:
            for numRelevantDimension in np.arange(3) + 1:
                for idx, informed in enumerate([True, False]):
                    ax = axes[idx, numRelevantDimension - 1]
                    ax.set_ylim(0, 1)
                    if numRelevantDimension == 1:
                        ax.set_ylabel('Unknown' if informed else 'Unknown', fontsize=fontsize)
                    if idx == 0:
                        ax.set_xlabel(str(numRelevantDimension) + 'D relevant', fontsize=fontsize)
                        ax.xaxis.set_label_position('top')
                    ax.tick_params(axis='both', labelsize=fontsize)
            if wTitleAxLabel:
                fig.text(0.5, 0.05, 'Trial', ha='center', fontsize=fontsize)
                fig.text(0.5, 0.95, 'Frequency', ha='center', fontsize=fontsize)
            if wLegend:
                fig.legend(handles=p, labels=['add', 'drop', 'switch_within', 'switch_across', 'mixed'], loc="center right")
    
    if showFigure:
        plt.tight_layout()
        plt.show()
    else:
        # plt.tight_layout()
        plt.savefig('figures/' + expVersion + '_' + varName + '.png', bbox_inches='tight')
    
    return fig, axes
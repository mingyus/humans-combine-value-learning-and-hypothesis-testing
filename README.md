This repository contains data and code that accompany the paper titled "Humans combine value learning and hypothesis testing strategically in multi-dimensional probabilistic reward learning".


## Data

`data/data_all.csv` contains the data from all participants. Each row in the data file corresponds to a trial in a game. Each column contain the follow information (columns marked with [game] are game-level information, and have the same values for all trials in a game):

* `workerId`: index of participants
* `game`: [game] index of game
* `trial`: index of trial
* `informed`: [game] whether this game is known (True) or unknown (False)
* `numRelevantDimensions`: [game] the number of relevant dimensions in this game
* `rt`: reaction time of the trial (the time from trial start to the participant hitting "Done" button, including the time they spent selecting features) 
* `reward`: reward of the trial, 0 or 1
* `ifRelevantDimension_color/shape/pattern`: [game] whether color/shape/pattern is the relevant dimension
* `rewardingFeature_color/shape/pattern`: [game] the rewarding feature on the color/shape/pattern dimension
* `selectedFeature_color/shape/pattern`: what feature participant selected on the color/shape/pattern dimension; null if the participant did not select any feature on this dimension
* `randomlySelectedFeature_color/shape/pattern`: what feature was randomly selected for the color/shape/pattern dimension; null if the participant made a feature selection on this dimension
* `builtFeature_color/shape/pattern`: what feature appeared in the final stimulus on the color/shape/pattern dimension; same as either `selectedFeature` or `randomlySelectedFeature`
* `order_color/shape/pattern`: [game] the position that color/shape/pattern dimension appeared on the screen, from 1 to 3 (top to bottom); the positions were randomly shuffled across games
* `idxFirstClick_color/shape/pattern`: the order in which the participant clicked on the color/shape/pattern dimension (only considering the first click within each dimension if there were multiple), from 1 to 3; null if there the participant did not click/select on this dimension
* `numSelectedFeatures`: the number of features selected by the participant on this trial
* `postGameAnswer_color/shape/pattern`: [game] the participant answer on the color/shape/pattern dimension about what feature was the rewarding feature in that dimension; "not-important" if that dimension was not relevant
* `postGameConfidence_color`: [game] the participant's confidence rating for the above answer, from 0 to 100

## Models and model fitting

### Models

* Bayesian rule learning model: `funcsMyopicBayesian.py`
* Feature RL with decay model: `funcsFeatureRLwDecay.py`
* Serial hypothesis testing models: `funcsInferSerialHypoTesting.py`

### Model fitting and simulation scripts

* `ModelFittingCVleave1game_script.py`: the script for model fitting (leave-one-game-out cross validation)
* `ModelFitting_script.py`: the script for model fitting (with data from all games; to generate the fitted parameter values used in model simulation)
* `ModelSimulation_script.py`: the script for model simulation

### Model fits and simulation results

Note that most models (especially serial hypothesis testing models) take a long time to fit and simulate. To help the readers reproduce the analysis results in the paper, we provide model fitting and simulation results under the following directories:

* `fittingResultsCollectCV/`: total model likelihood
* `fittingResultsCVTrialLik/`: trial-by-trial model likelihood
* `fittingResultsCollect/`: fitted parameter values used in model simulation
* `modelSimulation/`: simulated data

## Analyses and figures

The analysis results and figures in the paper can be reproduced with the following notebooks:

* Fig 2A,B: `Analyses_learningCurves.ipynb`
* Fig 2C: `Analyses_postGameQuestions.ipynb`
* Fig 4A: `Analysis_modelEvaluate.ipynb`
* Fig 4B,C: `Analyses_learningCurves.ipynb`
* Fig 5A,B: `Analysis_modelContributionCorrelate.ipynb`
* Fig 5C,D: `Analysis_modelEvaluate.ipynb`
* Fig S1A,B: `Analyses_learningCurves.ipynb`
* Fig S1C: `Analyses_postGameQuestions.ipynb`
* Fig S1D,E,F: `Analyses_behavior.ipynb`
* Fig S2: `Analysis_modelEvaluate.ipynb`
* Fig S3: `Analyses_learningCurves.ipynb`
* Fig S4B: `Analysis_modelEvaluate.ipynb`


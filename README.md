# MLB Swing Analysis

## Introduction
The goal of this analysis is to probabilistically predict whether or not a player will swing at a ball that comes their way. This analysis ultimately utilizes a Random Forest Algorithm in order to achieve this goal. Using 1500 estimators, this algorithm concludes with a 79.96% Accuracy Score and a 76.72% F1 Score. The model is expected to scale well and can handle both additional data records and additional features. The analysis that follows details the process and decisions made in order to produce the aforementioned model. Note: this analysis is intended for an audience with some, but perhaps not a deep understanding of Statistics and Baseball. However, the analysis should generally be understandable to those without this background.

## Setting Expectations
Of course, upon embarking on such an analysis, the most idealistic goal is perfect
accuracy and prediction. However, given the inherent randomness of baseball, along
with various shortcomings in the available data, it was important to establish a
benchmark for performance. The most naïve model, which would choose is_swing as
zero every single time, would evaluate to around a 53% accuracy mark. A model with
almost no pre-processing, hyperparameter tuning, or other feature engineering
appeared to evaluate somewhere in the mid-60% range for accuracy, depending on
the model. Various other attempts available online, including one published by
Baseball Prospectus, evaluate to an accuracy somewhere in the mid-70% range.
Finally, an attempt published in The Hardball Times evaluated a model to predict
contact vs no-contact on a swing to an 82.3% accuracy rate, which beat the naïve
approach of always choosing no-contact by only a couple percentage points.
All of this to say that predicting baseball is certainly a challenge, and model-builders
should contextualize their work and also be weary of certain results. (For example,
given the state of prediction in baseball, a 99% accuracy rate likely has some issues
lurking in the dark).

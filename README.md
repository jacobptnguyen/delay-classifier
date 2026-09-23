# Flight Delay Classifier

A classical ML notebook that predicts whether a US flight will be delayed from its airline, origin/destination airports, day of week, departure time and scheduled length.

## [▶ Live Demo](TODO)

## Dataset

[Airlines Dataset to predict a delay](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay) on Kaggle: 539,383 flights, no missing values. Download `Airlines.csv` into the project folder to run the notebook (it's not committed to the repo).

| Feature | Type | Notes |
|---|---|---|
| Airline | categorical (18) | strongest signal: WN delayed 70% of the time, YV 24% |
| AirportFrom / AirportTo | categorical (293 each) | busy hubs range from 32% to 74% delayed |
| DayOfWeek | 1–7 | weak signal |
| Time → Hour | departure hour | strong signal: ~25% early morning → ~50% evening |
| Length | minutes | weak alone, useful in combination |
| Delay | target (0/1) | 55.5% on time / 44.5% delayed |

`id` and `Flight` are identifiers and are dropped.

## Models used

| Model | Why it's included |
|---|---|
| Logistic Regression | A simple, interpretable starting point |
| Random Forest | Bagged trees; captures non-linear interactions |
| HistGradientBoosting | Boosted trees; handles categories natively and trains fast |

## Evaluation metrics

80/20 stratified split, scored on the 107,877-flight test set. ROC-AUC is the main metric because it doesn't depend on a decision threshold. Always predicting "on time" scores 55.5% accuracy, so every model has to beat that.

| Model | Accuracy | ROC-AUC | F1 | Fit time |
|---|---|---|---|---|
| Logistic Regression | 0.646 | 0.691 | 0.550 | 1.9s |
| Random Forest | 0.670 | 0.720 | 0.575 | 7.0s |
| HistGradientBoosting | 0.668 | 0.720 | 0.582 | 3.2s |

- Best model: Random Forest (0.7204 ROC-AUC), essentially tied with HistGradientBoosting (0.7201). Its recall on delayed flights is 0.50 (precision 0.67), so it misses about half of the real delays. Lowering the 0.5 threshold would catch more, at the cost of more false alarms.
- Every model plateaus around 0.72 AUC, so the limit is the data: there's no date, weather or upstream-delay information.
- Caveat: the dataset has no date column, so the same scheduled flight appears on many days, and a random split puts copies in both train and test. Test scores are slightly optimistic.

## Tech stack

Python · pandas · scikit-learn · matplotlib · seaborn · Jupyter · Streamlit (app, hosted on Streamlit Community Cloud)

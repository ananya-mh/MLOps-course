# XGBoost Model Visualization with Weights & Biases (W&B)

This lab trains and visualizes an **XGBoost multi-class classifier** using the **Dermatology dataset** from the UCI Machine Learning Repository.  
The experiment is tracked and analyzed using **Weights & Biases (W&B)** to monitor hyperparameters, metrics, and model performance.

---

## Dataset
**Source:** [UCI Dermatology Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/)  
The dataset classifies patients into one of six dermatological disease categories based on 33 clinical attributes.

Data is downloaded using:
```bash
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data -qq
```

## Experiment Workflow

- Loaded and split the dataset into training and testing sets.

- Configured XGBoost parameters as:

```python
param = {
    'objective': 'multi:softprob',
    'eta': 0.2,
    'max_depth': 4,
    'verbosity': 0,
    'nthread': 4,
    'num_class': 6
}
```


## Train the model

```python
bst = xgb.train(param, xg_train, num_round=5,
                evals=[(xg_train, 'train'), (xg_test, 'test')],
                callbacks=[wandb.xgboost.WandbCallback()])
```

- Evaluated performance using both error rate and accuracy.

- Visualized metrics and confusion matrix in W&B.

- Logged results and artifacts to W&B for tracking and analysis.
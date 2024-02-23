import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from scipy.stats import uniform, randint
import numpy as np
import matplotlib.pyplot as plt
import shap
import optuna
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer
import utility
from utility import load_data, plot_confusion_matrix, plot_average_f1_scores, train_model, get_hard_disk_path, plot_features_importance, get_shap, run_randomsearch_random_forest, evaluate_gridsearch

def perform_stratified_kfold_analysis(data_path, data_file, model_type, best_params, n_splits=5, random_state=42):
    """
    Perform a stratified k-fold cross-validation analysis using one of the models (Catboost, Xgboost, RandomForest). It calculates SHAP values, 
    F1 scores, and visualizes confusion matrices, average F1 scores, and feature importances.

    :param data_path: Path to the directory containing the data file.
    :type data_path: str
    :param data_file: Name of the data file.
    :type data_file: str
    :param model_type: Type of the model to train, defaults to 'Xgboost'.
    :type model_type: str, optional
    :param n_splits: Number of folds for StratifiedKFold, defaults to 5.
    :type n_splits: int, optional
    :param random_state: Seed used by the random number generator, defaults to 42.
    :type random_state: int, optional
    :return: Average F1 scores for each class and the trained classifier.
    :rtype: dict, classifier object
    """

    X, y, unique_labels, label_mapping, data = load_data(get_hard_disk_path(data_path), data_file)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    ix_training, ix_test = [], []
    for fold in kf.split(X, y):
        ix_training.append(fold[0]), ix_test.append(fold[1])

    average_shap_values_per_fold = []
    cumulative_f1_scores = {label: 0 for label in unique_labels}

    for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)):
        X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
        y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]
        if model_type == "RandomForest":
            imputer = SimpleImputer(strategy='mean')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

        classifier = train_model(X_train, y_train, unique_labels, model_type, best_params)
        y_pred = classifier.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, unique_labels)

        shap_values = get_shap(classifier, X_test)
        if isinstance(shap_values, list):
            shap_values = np.sum([np.abs(sv) for sv in shap_values], axis=0)

        average_shap_values_this_fold = np.mean(shap_values, axis=0)
        average_shap_values_per_fold.append(average_shap_values_this_fold)

        f1_scores_per_class = f1_score(y_test, y_pred, average=None)
        for idx, label in enumerate(unique_labels):
            cumulative_f1_scores[label] += f1_scores_per_class[idx]

    average_shap_values = np.mean(average_shap_values_per_fold, axis=0)
    average_shap_values_2d = average_shap_values.reshape(1, -1)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(average_shap_values_2d, features=X.columns, plot_type="bar", show=False)
    plt.tight_layout()
    plt.show()

    average_f1_scores = {label: score/n_splits for label, score in cumulative_f1_scores.items()} 
    ranked_labels = sorted(average_f1_scores, key=average_f1_scores.get, reverse=True)
    plot_average_f1_scores(ranked_labels, average_f1_scores)
    plot_features_importance(classifier)

    return average_f1_scores, classifier

def tune_and_evaluate_xgb_classifier(data_path, data_file, test_size=0.2, random_state=42, n_iter=200, cv_folds=10, verbose=2):
    """
    Load data, split into training and testing sets, tune an XGBoost classifier using RandomizedSearchCV, 
    and evaluate the best model on the test set using the F1 score.

    :param data_path: The path to the directory containing the data file. Used with `get_hard_disk_path`.
    :type data_path: str
    :param data_file: The name of the data file, e.g., 'statistics_features_IP.csv'.
    :type data_file: str
    :param test_size: Proportion of the dataset to include in the test split, defaults to 0.2.
    :type test_size: float, optional
    :param random_state: Controls the shuffling applied to the data, defaults to 42.
    :type random_state: int, optional
    :param n_iter: Number of parameter settings sampled in RandomizedSearchCV, defaults to 200.
    :type n_iter: int, optional
    :param cv_folds: Number of folds in StratifiedKFold for RandomizedSearchCV, defaults to 10.
    :type cv_folds: int, optional
    :param verbose: Verbosity level of RandomizedSearchCV, defaults to 2.
    :type verbose: int, optional
    :return: The best XGBoost classifier model and its F1 score on the test dataset.
    :rtype: tuple(XGBClassifier, float)

    :Example:

    >>> best_model, test_f1_score = tune_and_evaluate_xgb_classifier(
            data_path="Segmentation",
            data_file="statistics_features_IP.csv",
            test_size=0.2,
            random_state=42,
            n_iter=200,
            cv_folds=10,
            verbose=2
        )
    """
    # Load the data
    X, y, unique_labels, label_mapping, data = load_data(get_hard_disk_path(data_path), data_file)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define the parameter distributions
    param_dist = {
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'n_estimators': randint(100, 1000),
        'colsample_bytree': uniform(0.3, 0.7),
    }

    # Create the base model to tune
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Define the scorer, choosing an averaging method suitable for your problem
    f1_scorer = make_scorer(f1_score, average='weighted')  # or 'macro', 'micro'

    # Configure RandomizedSearchCV with F1-score as the scoring metric
    random_search = RandomizedSearchCV(
        estimator=xgb_model, param_distributions=param_dist, 
        n_iter=n_iter, cv=cv_folds, verbose=verbose, random_state=random_state, n_jobs=-1,
        scoring=f1_scorer  # Use the F1-score for evaluation
    )

    # Fit the random search model
    random_search.fit(X_train, y_train)

    # Print the best parameters and the corresponding F1-score
    print("Best parameters found: ", random_search.best_params_)
    print("Best F1-score: ", random_search.best_score_)

    # Predict on the test set with the best found parameters
    best_model = random_search.best_estimator_
    predictions = best_model.predict(X_test)

    # Evaluate the best model's F1-score on the test set
    test_f1_score = f1_score(y_test, predictions, average='weighted')  # Use 'weighted' to account for label imbalance
    print("Test F1-score: ", test_f1_score)

    return best_model, test_f1_score

def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "loss_function": "MultiClass",  # For multiclass classification
        "bootstrap_type": "Bernoulli",
        "silent": True
    }

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_test, y_test)])
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions, average='macro')  # Use 'binary' for binary classification
    return f1

def optimize_catboost_with_optuna(data_path, data_file, test_size=0.2, random_state=42, n_trials=100):
    """
    Optimize CatBoost classifier hyperparameters using Optuna for a given dataset to maximize the F1 score.

    This function loads the data, splits it into training and testing sets, creates an Optuna study, and 
    optimizes the CatBoost classifier hyperparameters. After optimization, it prints the results of the best trial.

    :param data_path: The path to the directory containing the data file. Used with `get_hard_disk_path`.
    :type data_path: str
    :param data_file: The name of the data file (e.g., 'statistics_features_IP.csv').
    :type data_file: str
    :param test_size: The proportion of the dataset to include in the test split. Default is 0.2.
    :type test_size: float, optional
    :param random_state: Controls the shuffling applied to the data before applying the split. Default is 42.
    :type random_state: int, optional
    :param n_trials: The number of trials for hyperparameter optimization in Optuna. Default is 100.
    :type n_trials: int, optional
    :return: The best trial object from the Optuna study.
    :rtype: optuna.trial.FrozenTrial

    :Example:

    >>> best_trial = optimize_catboost_with_optuna(
            data_path="Segmentation",
            data_file="statistics_features_IP.csv",
            test_size=0.2,
            random_state=42,
            n_trials=100
        )
    """
    X, y, unique_labels, label_mapping, data = load_data(get_hard_disk_path(data_path), data_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    return study.best_trial

def optimize_and_evaluate_random_forest(data_path, data_file, random_state=30):
    """
    Optimizes RandomForest hyperparameters using random search and evaluates the model 
    on a test set to return the best parameters and ROC AUC score.

    :param data_path: The path to the directory containing the data file. Used with `get_hard_disk_path`.
    :type data_path: str
    :param data_file: The name of the data file (e.g., 'statistics_features_IP.csv').
    :type data_file: str
    :param random_state: Seed used by the random number generator.
    :type random_state: int
    :return: ROC AUC score of the best model and the best hyperparameters found.
    :rtype: tuple(float, dict)

    :Example:

    >>> auc_score, best_params = optimize_and_evaluate_random_forest(data_path="Segmentation", data_file="statistics_features_IP.csv", random_state=30)
    >>> print(f"ROC AUC Score: {auc_score}")
    >>> print("Best Hyperparameters:")
    >>> print(best_params)
    """

    X, y, unique_labels, label_mapping, _ = load_data(get_hard_disk_path(data_path), data_file)
    grid_search, X_test, y_test = run_randomsearch_random_forest(X, y, random_state=random_state)
    auc_score, best_params = evaluate_gridsearch(grid_search, X_test, y_test, unique_labels)

    return auc_score, best_params
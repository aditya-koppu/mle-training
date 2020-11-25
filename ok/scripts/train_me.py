"""Trains all models. No pickle generation used

    :return: lin_reg, tree_reg, rnd_search, grid_search, final_model
    :rtype: list
"""
import logging

from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from scripts.data_test_train import housing_labels, housing_prepared

LOG_FORMAT = "%(filename)s %(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="My_log.log", level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()
logger.info("Started training")


def train_all():

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    # Decision Tree --fit
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    # random-forest fit
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)

    # hyperparameters
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    # hyper -train
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    # feature_importances = grid_search.best_estimator_.feature_importances_
    # sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
    final_model = grid_search.best_estimator_
    print("Done")

    # print("To print the model results type python check_models.py")
    logger.info("Finished training")
    return [lin_reg, tree_reg, rnd_search, grid_search, final_model]

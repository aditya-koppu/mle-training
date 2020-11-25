""" Scores all the models with train data. Best model will be used
on test data
    """
import logging

import numpy as np
from sklearn.metrics import mean_squared_error

import scripts.train_me as t
from scripts.data_test_train import housing_labels, housing_prepared

LOG_FORMAT = "%(filename)s %(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="My_log.log", level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()
logger.info("Started Scoring")

lin_reg, tree_reg, rnd_search, grid_search, final_model = t.train_all()


def score():
    # Score
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    logger.debug("Linear_reg rmse {}".format(lin_rmse))

    # lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    # lin_mae

    # Score
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    logger.debug("tree_reg rmse {}".format(tree_rmse))

    # score
    cvres = rnd_search.cv_results_
    for mean_score in cvres["mean_test_score"]:
        logger.debug("Random_forest rmse:", np.sqrt(-mean_score))

    # scores
    grid_search.best_params_
    cvres = grid_search.cv_results_
    for mean_score in cvres["mean_test_score"]:
        logger.debug("Hyper parameterized rmse:", np.sqrt(-mean_score))

    logger.info("Finished Scoring")
    print("done")

    return

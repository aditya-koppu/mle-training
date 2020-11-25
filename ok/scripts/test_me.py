"""
Tests the final model with test data. Gives out score.
    
"""
import logging

import numpy as np
from sklearn.metrics import mean_squared_error

import scripts.train_me as t
from scripts.data_test_train import X_test_prepared, y_test

LOG_FORMAT = "%(filename)s %(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="My_log.log", level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()
logger.info("Started Prediction")

lin_reg, tree_reg, rnd_search, grid_search, final_model = t.train_all()


def final_pred_res():
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    logger.debug("final rmse:{}".format(final_rmse))
    logger.info("Finished Prediction")
    print("Done")
    return

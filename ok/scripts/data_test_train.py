""" The initial part of scource file where all the data is processed. We download 
    from git and make required processe.
"""
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)

# downloading data links
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# download data
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    df = pd.read_csv(csv_path)
    return df


# assign data
housing = load_housing_data()

# train test split this is unnecessay
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# created a new col with nominal vals
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

# split 1
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# a new col as function
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


# split2
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# What will this do? It compares id stratified or random is better
# looks like stratified is better
compare_props = pd.DataFrame(
    {
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }
).sort_index()
compare_props["Rand. %error"] = (
    100 * compare_props["Random"] / compare_props["Overall"] - 100
)
compare_props["Strat. %error"] = (
    100 * compare_props["Stratified"] / compare_props["Overall"] - 100
)
# We will drop the created income cats
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# reassigning housing strat train set copy
housing = strat_train_set.copy()
# scatter plot long vs latitutude. But why?
# housing.plot(kind="scatter", x="longitude", y="latitude")
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# median income shows high corr with median house val
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# making changes here dont matter
# housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
# housing["population_per_household"] = housing["population"] / housing["households"]

# removing the above steps
housing = strat_train_set.drop(
    "median_house_value", axis=1
)  # drop labels for training set
# copy from source
housing_labels = strat_train_set["median_house_value"].copy()

# median imputing missing vals
imputer = SimpleImputer(strategy="median")

# cannot impute ocean prox as median value
housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
# this our X?
X = imputer.transform(housing_num)

# Why do this?
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
housing_tr["bedrooms_per_room"] = (
    housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
)
housing_tr["population_per_household"] = (
    housing_tr["population"] / housing_tr["households"]
)
# check this
housing_cat = housing[["ocean_proximity"]]
housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))


X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_num = X_test.drop("ocean_proximity", axis=1)
X_test_prepared = imputer.transform(X_test_num)
X_test_prepared = pd.DataFrame(
    X_test_prepared, columns=X_test_num.columns, index=X_test.index
)
X_test_prepared["rooms_per_household"] = (
    X_test_prepared["total_rooms"] / X_test_prepared["households"]
)
X_test_prepared["bedrooms_per_room"] = (
    X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
)
X_test_prepared["population_per_household"] = (
    X_test_prepared["population"] / X_test_prepared["households"]
)

X_test_cat = X_test[["ocean_proximity"]]
X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

# housing_prepared , housing_lables , X_test_prepared, y_test

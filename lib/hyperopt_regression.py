from hyperopt import hp, Trials, fmin, STATUS_OK, tpe
from hyperopt.pyll import scope
from hyperopt.early_stop import no_progress_loss

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

from functools import partial

from warnings import warn
from typing import Literal, List


__version__ = "0.7"


regressors = {
    "SVR": SVR,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "SGDRegressor": SGDRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    }


PARAMS_STRING = {
    "SVR": {
        "kernel": ("rbf", "poly"),
     },

    "SGDRegressor": {
        "loss": ('squared_epsilon_insensitive', 'epsilon_insensitive', 'huber', 'squared_error'),
        "penalty": ("l1", "l2", "elasticnet", None),
    },

    "DecisionTreeRegressor": {
        "criterion": ('friedman_mse', 'poisson', 'squared_error', 'absolute_error'),
        "max_depth": [None] + [i for i in range(1, 10)],
    },

    "RandomForestRegressor": {
        "criterion": ('friedman_mse', 'poisson', 'squared_error', 'absolute_error'),
        "max_depth": [None] + [i for i in range(1, 10)],
        "max_features": ("sqrt", "log2", None),
    },
}


SEARCH_SPACE = {
    "SVR": {
        "C": hp.lognormal("C", 1e-10, 10),
        "kernel": hp.choice(
            "kernel",
            PARAMS_STRING["SVR"]["kernel"],
        ),
        "degree": scope.int(hp.quniform("degree", 2, 10, 1)),
        "gamma": hp.lognormal("gamma", 1e-10, 5)
    },

    "SGDRegressor": {
        "loss": hp.choice("loss", PARAMS_STRING["SGDRegressor"]["loss"]),
        "penalty": hp.choice("penalty", PARAMS_STRING["SGDRegressor"]["penalty"]),
        "l1_ratio": hp.quniform("l1_ratio", 0, 1, 0.05),
        "alpha": hp.quniform("alpha", 1e-10, 1, 1e-2),
        "max_iter": 2000,
    },

    "DecisionTreeRegressor": {
        "criterion": hp.choice("criterion", PARAMS_STRING["DecisionTreeRegressor"]["criterion"]),
        "max_depth": hp.choice("max_depth", PARAMS_STRING["DecisionTreeRegressor"]["max_depth"]),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
    },

    "RandomForestRegressor": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 500, 50)),
        "criterion": hp.choice("criterion", PARAMS_STRING["RandomForestRegressor"]["criterion"]),
        "max_depth": hp.choice("max_depth", PARAMS_STRING["RandomForestRegressor"]["max_depth"]),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
        "max_features": hp.choice("max_features", PARAMS_STRING["RandomForestRegressor"]["max_features"]),
    },

}

scores_weights = {
    "neg_mean_squared_error": -1,
}


REGRESSOR_NAMES = tuple(regressors.keys())
SCORE_NAMES = tuple(scores_weights)


def _objective(params, X_train, y_train, regressor, scoring, cv=5):
    reg_instance = regressor(**params)

    reg_instance.fit(X_train, y_train)
    score = cross_val_score(reg_instance, X_train, y_train, scoring=scoring, cv=cv).mean()

    return {"loss": scores_weights[scoring] * score, "status": STATUS_OK}


def get_best_params(X, y, scoring: str, estimators: (list | None) = None, cv: int = 5, max_evals: int = 1000, early_stop: None|int = 1000) -> List:
    '''
    :param estimators: estimator name string list. if None will test all availiable estimators
    :param X: X, array-like, attributes
    :param y: y, array-like, target
    :param scoring: str, for classification scores - ("f1", "roc_auc", "recall", "precision", "accuracy")
    :param cv: int, n_fold, default is 5
    :param max_evals: int, max evaluation count
    :param early_stop: int, stop fitting when loss not changed over early_stop eval(s).

    returns
    -------
    :return: dict, model information reached best score.
    '''

    # vailadate arguments
    if estimators is None:
        regressor_candidates = tuple(regressors.keys())

    else:
        for estimator in estimators:
            if estimator not in REGRESSOR_NAMES:
                raise ValueError(str(estimator) + " is currently unsupported estimator.")

        regressor_candidates = estimators

    if scoring not in SCORE_NAMES:
        raise ValueError(str(scoring) + " is currently unsupported scoring method.")

    if early_stop is not None and early_stop > max_evals:
        warn("warning: early_stop is bigger than max_evals, setting early_stop to max_evals.")
        early_stop = max_evals


    best_estim = {
        "regressor": [],
        "args": [],
    }

    best_loss = int(1e10)


    indexer = 1
    for regressor_name in regressor_candidates:
        print("\rgetting scores...", indexer, '/', len(regressor_candidates), '-', regressor_name)
        trials = Trials()
        fmin_objective = partial(
            _objective,
            X_train=X,
            y_train=y,
            regressor=regressors[regressor_name],
            scoring=scoring,
            cv=cv,
        )

        estim_best = fmin(
            fn=fmin_objective,
            space=SEARCH_SPACE[regressor_name],
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            early_stop_fn=no_progress_loss(early_stop) if early_stop is not None else None,
        )

        if trials.best_trial is None or estim_best is None:
            raise RuntimeError()
        
        loss = round(trials.best_trial["result"]["loss"], 2)

        if loss <= best_loss:
            for parameter_name in PARAMS_STRING[regressor_name]: # index to string
                estim_best[parameter_name] = PARAMS_STRING[regressor_name][parameter_name][estim_best[parameter_name]]

            if loss == best_loss:
                best_estim["args"].append(estim_best)
                best_estim["regressor"].append(regressor_name)

            else:
                best_estim["args"] = [estim_best]
                best_estim["regressor"] = [regressor_name]
                best_loss = loss

        indexer += 1
        print("done.")

    return [best_estim, best_loss]


def get_regressor_candidates(X, y, scoring: str, cv: int = 5):
    '''
    params
    ------
    :param estimators: estimator name string list. if None will test all availiable estimators
    :param X: X, {array-like}, attributes
    :param y: y, {array-like}, target
    :param scoring: str, for classification scores - ("neg_mean_squared_error")
    :param cv: int, n_fold, default is 5
    :param ovo: uses OvO with multiclass classification, False uses OvR.

    returns
    -------
    dict, model information reached best score.
    '''

    best_estimators = []
    best_score = -int(1e10)

    i = 1
    for regressor in regressors.keys():
        print(f"\r[{i}/{len(regressors.keys())}]\t{regressor}", end="")
        regressor_instance = regressors[regressor]()
        regressor_instance.fit(X, y)
        
        score = cross_val_score(regressor_instance, X, y, cv=cv, n_jobs=-1, scoring=scoring).mean()


        if score > best_score:
            best_estimators = [regressor]
            best_score = score
            print("!")
        
        elif score == best_score:
            best_estimators.append(regressor)
        
        i += 1

    print("\ndone.")
    return (best_estimators, best_score)

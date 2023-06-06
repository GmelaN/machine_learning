from hyperopt import hp, Trials, fmin, STATUS_OK, tpe
from hyperopt.pyll import scope
from hyperopt.early_stop import no_progress_loss

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from functools import partial

from warnings import warn


__version__ = "0.7"


classifiers = {
    "SVC": SVC,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "SGDClassifier": SGDClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    }


PARAMS_STRING = {
    "SVC": {
        "kernel": ("rbf", "poly"),
     },

    "SGDClassifier": {
        "loss": ("hinge", "log_loss", "perceptron"),
        "penalty": ("l1", "l2", "elasticnet", None),
    },

    "DecisionTreeClassifier": {
        "criterion": ("gini", "entropy"),
        "max_depth": [None] + [i for i in range(1, 10)],
    },

    "RandomForestClassifier": {
        "criterion": ("gini", "entropy"),
        "max_depth": [None] + [i for i in range(1, 10)],
        "max_features": ("sqrt", "log2", None),
    },
}


SEARCH_SPACE = {
    "SVC": {
        "C": hp.lognormal("C", 0, 10),
        "kernel": hp.choice(
            "kernel",
            PARAMS_STRING["SVC"]["kernel"]
        ),
        "degree": scope.int(hp.quniform("degree", 2, 10, 1)),
        "gamma": hp.lognormal("gamma", 1e-10, 5)
    },

    "SGDClassifier": {
        "loss": hp.choice("loss", PARAMS_STRING["SGDClassifier"]["loss"]),
        "penalty": hp.choice("penalty", PARAMS_STRING["SGDClassifier"]["penalty"]),
        "l1_ratio": hp.quniform("l1_ratio", 0, 1, 0.05),
        "alpha": hp.quniform("alpha", 1e-10, 1, 1e-5),
        "max_iter": 2000,
        "n_jobs": -1,
        "random_state": 36,
    },

    "LogisticRegression": {
        "n_jobs": -1,
    },


    "DecisionTreeClassifier": {
        "criterion": hp.choice("criterion", PARAMS_STRING["DecisionTreeClassifier"]["criterion"]),
        "max_depth": hp.choice("max_depth", PARAMS_STRING["DecisionTreeClassifier"]["max_depth"]),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
    },

    "RandomForestClassifier": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 200, 50)),
        "criterion": hp.choice("criterion", PARAMS_STRING["RandomForestClassifier"]["criterion"]),
        "max_depth": hp.choice("max_depth", PARAMS_STRING["RandomForestClassifier"]["max_depth"]),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
        "max_features": hp.choice("max_features", PARAMS_STRING["RandomForestClassifier"]["max_features"]),
        "random_state": 36,

    },

}

scores_weights = {
    "f1": -1,
    "roc_auc": -1,
    "recall": -1,
    "precision": -1,
    "accuracy": -1,
}


CLASSIFIER_NAMES = tuple(classifiers.keys())
SCORE_NAMES = tuple(scores_weights)


def _objective(params, X_train, y_train, classifier, scoring, cv=5, multiclass_classifier=None):
    if multiclass_classifier is not None:
        clf_instance = multiclass_classifier(classifier(**params), n_jobs=-1)
    else:
        clf_instance = classifier(**params)

    clf_instance.fit(X_train, y_train)
    score = cross_val_score(clf_instance, X_train, y_train, scoring=scoring, cv=cv).mean()

    return {"loss": scores_weights[scoring] * score, "status": STATUS_OK}


def get_best_params(X, y, scoring: str, estimators: (list | None) = None, cv: int = 5, max_evals: int = 1000, early_stop: int = 1000, ovo=True):
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
        classifier_candidates = tuple(classifiers.keys())

    else:
        for estimator in estimators:
            if estimator not in CLASSIFIER_NAMES:
                raise ValueError(str(estimator) + " is currently unsupported estimator.")

        classifier_candidates = estimators

    if scoring not in SCORE_NAMES:
        raise ValueError(str(scoring) + " is currently unsupported scoring method.")

    if early_stop > max_evals:
        warn("warning: early_stop is bigger than max_evals, setting early_stop to max_evals.")
        early_stop = max_evals


    best = {
        "classifier": [],
        "args": [],
        "loss": 1e1,
    }

    is_multiclass = len(set(y)) > 2

    if is_multiclass:
        multiclass_classifier = OneVsOneClassifier if ovo else OneVsRestClassifier
    else:
        multiclass_classifier = None


    indexer = 1
    for classifier_name in classifier_candidates:
        print("\rgetting scores...", indexer, '/', len(classifier_candidates), '-', classifier_name)
        trials = Trials()
        fmin_objective = partial(
            _objective,
            X_train=X,
            y_train=y,
            classifier=classifiers[classifier_name],
            scoring=scoring,
            cv=cv,
            multiclass_classifier=multiclass_classifier,
        )

        clf_best = fmin(
            fn=fmin_objective,
            space=SEARCH_SPACE[classifier_name],
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            early_stop_fn=no_progress_loss(early_stop)
        )

        if trials.best_trial is None or clf_best is None:
            raise RuntimeError()

        if trials.best_trial["result"]["loss"] <= best["loss"]:
            for parameter_name in PARAMS_STRING[classifier_name]: # index to string
                clf_best[parameter_name] = PARAMS_STRING[classifier_name][parameter_name][clf_best[parameter_name]]

            if trials.best_trial["result"]["loss"] == best["loss"]:
                best["args"].append(clf_best)
                best["classifier"].append(classifier_name)

            else:
                best["args"] = [clf_best]
                best["classifier"] = [classifier_name]
                best["loss"] = trials.best_trial["result"]["loss"]

        indexer += 1
        print("done.")

    return best


def get_classifier_candidates(X, y, scoring: str, cv: int = 5, ovo=True):
    '''
    params
    ------
    :param estimators: estimator name string list. if None will test all availiable estimators
    :param X: X, {array-like}, attributes
    :param y: y, {array-like}, target
    :param scoring: str, for classification scores - ("f1", "roc_auc", "recall", "precision", "accuracy")
    :param cv: int, n_fold, default is 5
    :param ovo: uses OvO with multiclass classification, False uses OvR.

    returns
    -------
    dict, model information reached best score.
    '''

    is_multiclass = len(set(y)) > 2

    if is_multiclass:
        multiclass_classifier = OneVsOneClassifier if ovo else OneVsRestClassifier

    best_estimators = []
    best_score = 0

    for classifier in classifiers.keys():
        classifier_instance = classifiers[classifier]()

        if is_multiclass:
            classifier_instance = multiclass_classifier(classifier_instance, n_jobs=-1)

        classifier_instance.fit(X, y)
        
        score = cross_val_score(classifier_instance, X, y, cv=cv, n_jobs=-1, scoring=scoring).mean()

        if score > best_score:
            best_estimators = [classifier]
            best_score = score
        
        elif score == best_score:
            best_estimators.append(classifier)
    

    return (best_estimators, best_score)

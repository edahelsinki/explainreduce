from slisemap.tuning import hyperparameter_tune
import functools
import gc
import sys
from glob import glob
from timeit import default_timer as timer
from itertools import chain

import numpy as np
import pandas as pd
import torch
from project_paths import RESULTS_DIR
from scipy.stats import trim_mean
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from slisemap.utils import softmax_column_kernel, softmax_row_kernel, squared_distance

OUTPUT_DIR = RESULTS_DIR / "hyperparam_optim"

import explainreduce.localmodels as lm
import data
from slisemap import Slisemap
from slisemap.local_models import LinearRegression, LogisticRegression
from slisemap.slipmap import Slipmap
from slisemap.tuning import hyperparameter_tune
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer


def get_bb(bb, cls, X, y, dname=None):
    if bb == "AdaBoost":
        if cls:
            m = AdaBoostClassifier(random_state=42).fit(X, y)
            pred_fn = m.predict_proba
        else:
            if dname == "Gas Turbine":
                m = AdaBoostRegressor(random_state=42, learning_rate=0.05).fit(X, y)
            else:
                m = AdaBoostRegressor(random_state=42).fit(X, y)
            pred_fn = m.predict
    elif bb == "Random Forest":
        if cls:
            if dname == "Jets":
                m = RandomForestClassifier(
                    n_jobs=-1, max_leaf_nodes=100, max_features=3, random_state=42
                ).fit(X, y[:, 1])
            else:
                m = RandomForestClassifier(
                    n_jobs=-1, max_leaf_nodes=100, random_state=42
                ).fit(X, y[:, 1])
            pred_fn = m.predict_proba
        else:
            m = RandomForestRegressor(n_jobs=-1, max_leaf_nodes=100, random_state=42)
            if y.ndim > 1:
                y = y.ravel()
            m.fit(X, y)
            pred_fn = m.predict
    elif bb == "Gradient Boosting":
        if cls:
            if dname == "Higgs":
                m = GradientBoostingClassifier(
                    random_state=42, learning_rate=0.1, max_depth=5
                ).fit(X, y[:, 1])
            else:
                m = GradientBoostingClassifier(random_state=42).fit(X, y[:, 1])
            pred_fn = m.predict_proba
        else:
            m = GradientBoostingRegressor(random_state=42).fit(X, y)
            pred_fn = m.predict
    elif bb == "SVM":
        if cls:
            m = SVC(random_state=42, probability=True).fit(X, y[:, 1])
            pred_fn = m.predict_proba
        else:
            m = SVR().fit(X, y)
            pred_fn = m.predict
    elif bb == "Neural Network":
        if cls:
            m = MLPClassifier((64, 32, 16), random_state=42, early_stopping=True)
            m.fit(X, y)
            pred_fn = m.predict_proba
        else:
            if y.ndim == 2:
                y = y.ravel()
            m = MLPRegressor((128, 64, 32, 16), random_state=42, early_stopping=True)
            m.fit(X, y)
            pred_fn = m.predict
    else:
        raise NotImplementedError(f"[BB: {bb}] not implemented for [classifier: {cls}]")
    return m, pred_fn


def get_data(differentiable=False):
    """Get data dictionary in the form of dset name:  (X, y, bb_name, classification)"""
    if differentiable:
        # use differentiable BB models
        return {
            # use lambdas for lazy loading
            "Synthetic": lambda: (
                *data.get_rsynth(N=5000, k=5, s=2.0)[1:3],
                "Neural Network",
                False,
            ),
            "Air Quality": lambda: (*data.get_airquality(), "SVM", False),
            "Gas Turbine": lambda: (*data.get_gas_turbine(), "Neural Network", False),
            "QM9": lambda: (*data.get_qm9(), "Neural Network", False),
            "Higgs": lambda: (*data.get_higgs(), "SVM", True),
            "Jets": lambda: (*data.get_jets(), "Neural Network", True),
            "Life Expectancy": lambda: (
                *data.get_life(blackbox=None),
                "Neural Network",
                False,
            ),
            "Vehicles": lambda: (*data.get_vehicle(blackbox=None), "SVM", False),
        }

    else:
        return {
            # use lambdas for lazy loading
            "Synthetic": lambda: (
                *data.get_rsynth(N=5000, k=5, s=2.0)[1:3],
                "Random Forest",
                False,
            ),
            "Air Quality": lambda: (*data.get_airquality(), "Random Forest", False),
            "Gas Turbine": lambda: (*data.get_gas_turbine(), "AdaBoost", False),
            "QM9": lambda: (*data.get_qm9(), "Neural Network", False),
            "Higgs": lambda: (*data.get_higgs(), "Gradient Boosting", True),
            "Jets": lambda: (*data.get_jets(), "Random Forest", True),
            "Life Expectancy": lambda: (
                *data.get_life(blackbox=None),
                "Neural Network",
                False,
            ),
            "Vehicles": lambda: (*data.get_vehicle(blackbox=None), "SVM", False),
        }


def get_hyperopts():
    return {
        "SLIPMAP": functools.partial(
            hyperopt_slipmap, weighted=True, squared=True, density=True
        ),
        "SLISEMAP": hyperopt_slisemap,
        "LIME": hyperopt_lime,
        # "SHAP": hyperopt_shap,
        "SmoothGrad": hyperopt_smoothgrad,
    }


def hyperopt_shap(X_train, y_train, X_test, y_test, classifier, pred_fn):
    return


def hyperopt_smoothgrad(
    X_train, y_train, X_test, y_test, classifier, model, n_calls=15, verbose=False
):
    default_kernel_width = 1e-4
    space = [
        Real(
            0.1 * default_kernel_width,
            10 * default_kernel_width,
            "uniform",
            name="perturbation",
        ),
        Real(
            0.01,
            1.0,
            "uniform",
            name="noise_level",
        ),
    ]

    @use_named_args(space)
    def objective(**params):
        exp = lm.SmoothGradExplainer(
            X_train, y_train, classifier, black_box_model=model, **params
        )
        exp.fit()
        yhat = exp.predict(X_test)
        return exp.loss_fn(yhat, y_test).mean().item()

    res = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=min(10, max(3, (n_calls - 1) // 3 + 1)),
        random_state=42,
    )
    params = {}
    for s, v in zip(space, res.x):
        params[s.name] = v
    if verbose:
        print("Final parameter values:", params)

    return params


def hyperopt_slipmap(
    X_train,
    y_train,
    X_test,
    y_test,
    classifier,
    model=None,
    weighted=False,
    row_kernel=False,
    squared=True,
    density=False,
):
    params = hyperparameter_tune(
        Slipmap,
        X_train,
        y_train,
        X_test,
        y_test,
        model=False,
        local_model=LogisticRegression if classifier else LinearRegression,
        kernel=softmax_row_kernel if row_kernel else softmax_column_kernel,
        distance=squared_distance if squared else torch.cdist,
        prototypes=1.0 if density else 52,
        predict_kws=dict(weighted=weighted),
    )
    return params


def hyperopt_slisemap(
    X_train, y_train, X_test, y_test, classifier, model=None, pred_fn=None
):
    params = hyperparameter_tune(
        Slisemap,
        X_train,
        y_train,
        X_test,
        y_test,
        model=False,
        radius=3.5,
        n_calls=10,
        local_model=LogisticRegression if classifier else LinearRegression,
    )
    return params


def hyperopt_lime(
    X_train,
    y_train,
    X_test,
    y_test,
    classifier,
    model,
    n_calls=15,
    random_state=42,
    verbose=False,
):
    # from LIME documentation
    default_kernel = np.sqrt(0.75 * X_train.shape[1])
    space = [
        Real(0.1 * default_kernel, 2 * default_kernel, "uniform", name="kernel_width")
    ]

    @use_named_args(space)
    def objective(**params):
        exp = lm.LIMEExplainer(
            X_train, y_train, classifier, black_box_model=model, **params
        )
        exp.fit()
        yhat = exp.predict(X_test)
        return exp.loss_fn(yhat, y_test).mean().item()

    res = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=min(10, max(3, (n_calls - 1) // 3 + 1)),
        random_state=42,
    )
    params = {}
    for s, v in zip(space, res.x):
        params[s.name] = v
    if verbose:
        print("Final parameter values:", params)

    return params


def evaluate(job):
    print(f"Job: {job}", flush=True)
    np.random.seed(42 + job)
    torch.manual_seed(42 + job)
    file = OUTPUT_DIR / f"hyper_{job:02d}.parquet"
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.exists():
        results = pd.read_parquet(file)
    else:
        results = pd.DataFrame()

    for dname, dfn in get_data(False).items():
        print(f"Loading:    {dname}", flush=True)
        X, y, bb, cls = dfn()
        m, pred_fn = get_bb(bb, cls, X, y, dname=dname)
        ntrain = min(10_000, X.shape[0] * 3 // 4)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=ntrain, random_state=424242 + job, shuffle=True
        )
        print(f"Training:   {dname} - {bb}", flush=True)
        p_train = torch.as_tensor(pred_fn(X_train))
        if p_train.ndim < 2:
            p_train = p_train[:, None]
        p_test = torch.as_tensor(pred_fn(X_test))
        if p_test.ndim < 2:
            p_test = p_test[:, None]
        for mname, hyperopt in get_hyperopts().items():
            results = _eval_hyperparams(
                dname,
                mname,
                bb,
                m,
                hyperopt,
                X_train,
                p_train,
                X_test,
                p_test,
                cls,
                job,
                results,
                file,
            )
    print("Start training differentiable models.", flush=True)
    for dname, dfn in get_data(True).items():
        print(f"Loading:    {dname}", flush=True)
        X, y, bb, cls = dfn()
        m, pred_fn = get_bb(bb, cls, X, y, dname=dname)
        ntrain = min(10_000, X.shape[0] * 3 // 4)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=ntrain, random_state=424242 + job, shuffle=True
        )
        print(f"Training:   {dname} - {bb}", flush=True)
        p_train = torch.as_tensor(pred_fn(X_train))
        if p_train.ndim < 2:
            p_train = p_train[:, None]
        p_test = torch.as_tensor(pred_fn(X_test))
        if p_test.ndim < 2:
            p_test = p_test[:, None]
        for mname, hyperopt in get_hyperopts().items():
            results = _eval_hyperparams(
                dname,
                mname,
                bb,
                m,
                hyperopt,
                X_train,
                p_train,
                X_test,
                p_test,
                cls,
                job,
                results,
                file,
            )
    print("Done", flush=True)


def _eval_hyperparams(
    dname,
    mname,
    bb,
    m,
    hyperopt_fn,
    X_train,
    y_train,
    X_test,
    y_test,
    cls,
    job,
    results,
    file,
):
    if not results.empty:
        mask = results["data"] == dname
        mask &= results["method"] == mname
        mask &= results["BB"] == bb
        if mask.any():
            return results
    print(f"Tuning:     {dname}-{mname}-{bb}", flush=True)
    gc.collect()
    # try:
    time = timer()
    params = hyperopt_fn(X_train, y_train, X_test, y_test, cls, m)
    time = timer() - time
    res = {
        "job": job,
        "data": dname,
        "BB": bb,
        "method": mname,
        "time": time,
        **params,
    }
    results = pd.concat((results, pd.DataFrame([res])), ignore_index=True)
    results.to_parquet(file)
    # except Exception as e:
    #     print(f"Failed:     {dname}-{mname}-{bb}\n", e, flush=True)
    return results


def get_params(method, data, black_box):
    if black_box is None:
        black_box = ""
    try:
        return PARAM_CACHE[(method, data, black_box)]
    except:
        raise NotImplementedError(
            f"No cached parameters for {method} & {data} & {black_box}"
        )


PARAM_CACHE = {
    ("LIME", "Air Quality", "Random Forest"): {"kernel_width": np.float64(0.35037)},
    ("LIME", "Gas Turbine", "AdaBoost"): {"kernel_width": np.float64(0.41508)},
    ("LIME", "Higgs", "Gradient Boosting"): {"kernel_width": np.float64(1.86389)},
    ("LIME", "Jets", "Random Forest"): {"kernel_width": np.float64(0.22913)},
    ("LIME", "Life Expectancy", "Neural Network"): {
        "kernel_width": np.float64(0.56309)
    },
    ("LIME", "QM9", "Neural Network"): {"kernel_width": np.float64(0.77626)},
    ("LIME", "Synthetic", "Random Forest"): {"kernel_width": np.float64(2.94358)},
    ("LIME", "Vehicles", "SVM"): {"kernel_width": np.float64(0.3894)},
    ("SLIPMAP", "Air Quality", "Random Forest"): {
        "lasso": np.float64(0.0512),
        "ridge": np.float64(0.00469),
        "radius": np.float64(1.64729),
    },
    ("SLIPMAP", "Gas Turbine", "AdaBoost"): {
        "lasso": np.float64(0.20543),
        "ridge": np.float64(0.00205),
        "radius": np.float64(2.63776),
    },
    ("SLIPMAP", "Higgs", "Gradient Boosting"): {
        "lasso": np.float64(7.77108),
        "ridge": np.float64(0.07604),
        "radius": np.float64(2.6361),
    },
    ("SLIPMAP", "Jets", "Random Forest"): {
        "lasso": np.float64(0.00295),
        "ridge": np.float64(0.00011),
        "radius": np.float64(1.87088),
    },
    ("SLIPMAP", "Life Expectancy", "Neural Network"): {
        "lasso": np.float64(0.02155),
        "ridge": np.float64(0.0068),
        "radius": np.float64(2.34745),
    },
    ("SLIPMAP", "QM9", "Neural Network"): {
        "lasso": np.float64(0.03416),
        "ridge": np.float64(0.00044),
        "radius": np.float64(1.5),
    },
    ("SLIPMAP", "Synthetic", "Random Forest"): {
        "lasso": np.float64(0.26804),
        "ridge": np.float64(0.00163),
        "radius": np.float64(2.37795),
    },
    ("SLIPMAP", "Vehicles", "SVM"): {
        "lasso": np.float64(0.01155),
        "ridge": np.float64(0.01002),
        "radius": np.float64(1.75377),
    },
    ("SLISEMAP", "Air Quality", "Random Forest"): {
        "lasso": np.float64(0.00147),
        "ridge": np.float64(0.0001),
        "radius": np.float64(3.5),
    },
    ("SLISEMAP", "Gas Turbine", "AdaBoost"): {
        "lasso": np.float64(0.00221),
        "ridge": np.float64(0.0001),
        "radius": np.float64(3.5),
    },
    ("SLISEMAP", "Higgs", "Gradient Boosting"): {
        "lasso": np.float64(0.00524),
        "ridge": np.float64(0.02638),
        "radius": np.float64(3.5),
    },
    ("SLISEMAP", "Jets", "Random Forest"): {
        "lasso": np.float64(0.00155),
        "ridge": np.float64(0.00053),
        "radius": np.float64(3.5),
    },
    ("SLISEMAP", "Life Expectancy", "Neural Network"): {
        "lasso": np.float64(0.00776),
        "ridge": np.float64(0.0001),
        "radius": np.float64(3.5),
    },
    ("SLISEMAP", "QM9", "Neural Network"): {
        "lasso": np.float64(0.0011),
        "ridge": np.float64(0.0001),
        "radius": np.float64(3.5),
    },
    ("SLISEMAP", "Synthetic", "Random Forest"): {
        "lasso": np.float64(0.272),
        "ridge": np.float64(0.00424),
        "radius": np.float64(3.5),
    },
    ("SLISEMAP", "Vehicles", "SVM"): {
        "lasso": np.float64(0.00176),
        "ridge": np.float64(0.0001),
        "radius": np.float64(3.5),
    },
    ("SmoothGrad", "Air Quality", "Random Forest"): {
        "perturbation": np.float64(0.00078),
        "noise_level": np.float64(0.1916),
    },
    ("SmoothGrad", "Gas Turbine", "AdaBoost"): {
        "perturbation": np.float64(0.0008),
        "noise_level": np.float64(0.19353),
    },
    ("SmoothGrad", "Higgs", "Gradient Boosting"): {
        "perturbation": np.float64(0.001),
        "noise_level": np.float64(0.99953),
    },
    ("SmoothGrad", "Jets", "Random Forest"): {
        "perturbation": np.float64(1e-05),
        "noise_level": np.float64(1.0),
    },
    ("SmoothGrad", "Life Expectancy", "Neural Network"): {
        "perturbation": np.float64(5e-05),
        "noise_level": np.float64(0.28341),
    },
    ("SmoothGrad", "QM9", "Neural Network"): {
        "perturbation": np.float64(0.00068),
        "noise_level": np.float64(0.24786),
    },
    ("SmoothGrad", "Synthetic", "Random Forest"): {
        "perturbation": np.float64(0.00073),
        "noise_level": np.float64(0.46207),
    },
    ("SmoothGrad", "Vehicles", "SVM"): {
        "perturbation": np.float64(0.00053),
        "noise_level": np.float64(0.34992),
    },
}


if __name__ == "__main__":
    if len(sys.argv) == 1:
        files = glob(str(OUTPUT_DIR / "*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df_cache = {
            k1: {k2: np.round(v2, 5) for k2, v2 in v1.items() if np.isfinite(v2)}
            for k1, v1 in df.drop(["job", "time"], axis=1)
            .groupby(["method", "data", "BB"])
            .aggregate(trim_mean, 0.2)
            .to_dict("index")
            .items()
        }
        print("PARAM_CACHE = ", str(df_cache).replace(", 'radius': 3.5", ""))
    else:
        evaluate(int(sys.argv[1]))

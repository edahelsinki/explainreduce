"""
    This script loads datasets (including downloading them if necessary).
    Note that all datasets are also normalised in some way.
"""

from urllib.request import urlretrieve
from typing import Literal, Optional, Tuple, Union, List, Sequence
from pathlib import Path
from urllib.request import urlretrieve
from itertools import combinations
from zipfile import ZipFile

from scipy.io import arff
import numpy as np
import pandas as pd
import openml
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
import re


def find_path(dir_name: str = "data") -> Path:
    path = Path(dir_name)
    if not path.is_absolute():
        # Use the path of this file to find the root directory of the project
        path = Path(__file__).parent.parent.parent / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_dataset(id: int, cache_dir: Union[str, Path]) -> openml.OpenMLDataset:
    openml.config.apikey = "c1994bdb7ecb3c6f3c8f3b35f4b47f1f"
    # OpenML can handle the caching of datasets
    openml.config.cache_directory = cache_dir
    return openml.datasets.get_dataset(
        id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )


def _get_predictions(id: int, columns: Sequence[str], cache_dir: Union[str, Path]):
    openml.config.apikey = "c1994bdb7ecb3c6f3c8f3b35f4b47f1f"
    openml.config.cache_directory = cache_dir
    run = openml.runs.get_run(id, False)
    dir = cache_dir / "org" / "openml" / "www" / "runs" / str(id)
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / "predictions.arff"
    if not path.exists():
        urlretrieve(run.predictions_url, path)
    data, meta = arff.loadarff(path)
    pred = np.stack(tuple(data[c] for c in columns), -1)
    return pred[np.argsort(data["row_id"])]


def get_boston(
    blackbox: Optional[str] = None,
    names: bool = False,
    normalise: bool = True,
    remove_B: bool = False,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    dir = find_path(data_dir)
    dataset = _get_dataset(531, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    X, y = X.to_numpy(), y.to_numpy()
    if remove_B:
        X = X[:, [n != "B" for n in attribute_names]]
        attribute_names.remove("B")
    if blackbox is None:
        pass
    elif blackbox.lower() in ("svm", "svr"):
        y = _get_predictions(9918403, ("prediction",), dir)[:, 0]
    else:
        raise Exception(f"Unimplemented black box for boston: '{blackbox}'")
    if normalise:
        X = StandardScaler().fit_transform(X)
        y = StandardScaler().fit_transform(y[:, None])[:, 0]
    if names:
        return X, y, attribute_names
    else:
        return X, y


def get_fashion_mnist(
    blackbox: Optional[str] = None, names: bool = False, data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    dir = find_path(data_dir)
    dataset = _get_dataset(40996, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    X, y = X.to_numpy(), y.to_numpy()
    X = X / 255.0
    if blackbox is None:
        Y = np.eye(10, dtype=X.dtype)[y]
    elif blackbox == "cnn":
        # This is a convolutional neural network with 94% accuracy
        # The predictions are from a 10-fold crossvalidation
        Y = _get_predictions(9204216, (f"confidence.{c:d}" for c in range(10)), dir)
    else:
        raise Exception(f"Unimplemented black box for fashion mnist: '{blackbox}'")
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_mnist(
    blackbox: Optional[str] = None, names: bool = False, data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    dir = find_path(data_dir)
    dataset = _get_dataset(554, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    X, y = X.to_numpy(), y.to_numpy()
    X = X / 255.0
    if blackbox is None:
        Y = np.eye(10, dtype=X.dtype)[y]
    elif blackbox == "cnn":
        Y = _get_predictions(9204129, (f"confidence.{c:d}" for c in range(10)), dir)
    else:
        raise Exception(f"Unimplemented black box for mnist: '{blackbox}'")
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_emnist(
    blackbox: Optional[str] = None, names: bool = False, data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    dir = find_path(data_dir)
    dataset = _get_dataset(41039, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    X, y = X.to_numpy(), y.to_numpy().astype(int)
    mask = y < 10
    X = X[mask]
    y = y[mask]
    X = X / 255.0
    X = np.reshape(np.transpose(np.reshape(X, (-1, 28, 28)), (0, 2, 1)), (-1, 28 * 28))
    if blackbox is None:
        Y = np.eye(10, dtype=X.dtype)[y]
    elif blackbox == "cnn":
        Y = _get_predictions(9204295, (f"confidence.{c:d}" for c in range(10)), dir)
        Y = Y[mask]
    else:
        raise Exception(f"Unimplemented black box for mnist: '{blackbox}'")
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_spam(
    blackbox: Optional[Literal["rf"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    # https://archive.ics.uci.edu/ml/datasets/spambase
    dir = find_path(data_dir)
    dataset = _get_dataset(44, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    X, y = X.to_numpy(), y.to_numpy()
    X = X / np.max(X, 0, keepdims=True)
    if blackbox is None:
        Y = np.eye(2, dtype=X.dtype)[y]
    elif blackbox.lower() in ("rf", "random forest", "randomforest"):
        Y = _get_predictions(9132654, (f"confidence.{c:d}" for c in range(2)), dir)
    else:
        raise Exception(f"Unimplemented black box for spam: '{blackbox}'")
    if normalise:
        X = StandardScaler().fit_transform(X)
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_higgs(
    blackbox: Optional[Literal["gb"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    # https://archive.ics.uci.edu/ml/datasets/HIGGS
    dir = find_path(data_dir)
    dataset = _get_dataset(23512, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    X, y = X.to_numpy(), y.to_numpy().astype(int)
    X = X[:-1]
    y = y[:-1]
    if blackbox is None:
        Y = np.eye(2, dtype=X.dtype)[y]
    elif blackbox.lower() in ("gb", "gradient boosting", "gradientboosting"):
        Y = _get_predictions(9907793, (f"confidence.{c:d}" for c in range(2)), dir)
        Y = Y[:-1]
    else:
        raise Exception(f"Unimplemented black box for higgs: '{blackbox}'")
    if normalise:
        X = StandardScaler().fit_transform(X)
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_covertype(
    blackbox: Optional[Literal["lb"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    # https://archive.ics.uci.edu/ml/datasets/Covertype
    dir = find_path(data_dir)
    dataset = _get_dataset(150, dir)  # This is the already normalised version
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    X, y = X.to_numpy(), y.to_numpy()
    X = X[:-1]
    X = X.astype(float)
    y = y[:-1]
    y = y.astype(int) - 1
    if blackbox is None:
        Y = np.eye(7, dtype=X.dtype)[y]
    elif blackbox.lower() in ("lb", "logit boost", "logitboost"):
        Y = _get_predictions(157511, (f"confidence.{c:d}" for c in range(1, 8)), dir)
        Y = Y[:-1]
    else:
        raise Exception(f"Unimplemented black box for covertype: '{blackbox}'")
    if normalise:
        X = StandardScaler().fit_transform(X)
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_covertype3(
    blackbox: Optional[Literal["lb"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Like `get_covertype` but with the rarest classes combined into one"""
    X, Y, attribute_names = get_covertype(blackbox, True, normalise, data_dir)
    Y2 = Y[:, :3]
    Y2[:, 2] = Y[:, 2:].max(1)
    if blackbox is not None:
        Y2 = Y2 / Y2.sum(1, keepdims=True)
    if names:
        return X, Y2, attribute_names
    else:
        return X, Y2


def get_autompg(
    blackbox: Optional[str] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    # https://archive-beta.ics.uci.edu/ml/datasets/auto+mpg
    dir = find_path(data_dir)
    dataset = _get_dataset(196, dir)
    X, y, _, anames = dataset.get_data(target=dataset.default_target_attribute)
    X = np.concatenate(
        (X.values[:, :-1].astype(float), np.eye(3)[X["origin"].values.astype(int) - 1]),
        1,
    )
    mask = ~np.isnan(X[:, 2])
    X = X[mask]
    y = y[mask]
    anames = anames[:-2] + ["year", "origin USA", "origin Europe", "origin Japan"]
    if blackbox is None:
        Y = y.values
    elif blackbox.lower() in ("svm", "svr"):
        Y = _get_predictions(9918402, ("prediction",), dir)[mask, 0]
    elif blackbox.lower() in ("rf", "randomforest", "random forest"):
        random_forest = RandomForestRegressor(random_state=42).fit(X, y.ravel())
        Y = random_forest.predict(X)
    else:
        raise Exception(f"Unimplemented black box for Auto MPG: '{blackbox}'")
    if normalise:
        X[:, :-3] = StandardScaler().fit_transform(X[:, :-3])
        Y = StandardScaler().fit_transform(Y[:, None])[:, 0]
    if names:
        return X, Y, anames
    else:
        return X, Y


def get_iris(
    blackbox: Optional[str] = None, names: bool = False, data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    dataset = _get_dataset(61, find_path(data_dir))
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    y = y.cat.codes
    X, y = X.to_numpy(), y.to_numpy()
    Y = np.eye(3, dtype=X.dtype)[y]
    # TODO: get blackbox predictions from openml
    X = StandardScaler().fit_transform(X)
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_airquality(
    blackbox: Optional[Literal["rf"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Get the Air Quality dataset.

    Cleaned and preprocessed as in:

        Oikarinen E, Tiittanen H, Henelius A, Puolamäki K (2021)
        Detecting virtual concept drift of regressors without ground truth values.
        Data Mining and Knowledge Discovery 35(3):726-747, DOI 10.1007/s10618-021-00739-7

    Args:
        blackbox (Optional[str]): Return predictions from a black box instead of y (currently not implemented). Defaults to None.
        names (bool, optional): Return the names of the columns. Defaults to False.
        data_dir (str, optional): Directory where the data is saved. Defaults to "data".

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[List[str]]]: X and y (and column names).
    """
    path = find_path(data_dir) / "AQ_cleaned_version.csv"
    if not path.exists():
        url = "https://raw.githubusercontent.com/edahelsinki/drifter/master/TiittanenOHP2019-code/data/AQ_cleaned_version.csv"
        urlretrieve(url, path)
    AQ = pd.read_csv(path)
    columns = [
        "PT08.S1(CO)",
        "C6H6(GT)",
        "PT08.S2(NMHC)",
        "NOx(GT)",
        "PT08.S3(NOx)",
        "NO2(GT)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH",
    ]
    nnames = [
        "CO(sensor)",
        "C6H6(GT)",
        "NMHC(sensor)",
        "NOx(GT)",
        "NOx(sensor)",
        "NO2(GT)",
        "NO2(sensor)",
        "O3(sensor)",
        "Temperature",
        "Relative hum.",
        "Absolute hum.",
    ]

    X = AQ[columns].to_numpy()
    y = AQ["CO(GT)"].to_numpy()
    if normalise:
        X = StandardScaler().fit_transform(X)
        y = StandardScaler().fit_transform(y[:, None])[:, 0]
    if blackbox is None:
        pass
    elif blackbox.lower() in ("rf", "random forest", "randomforest"):
        y = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X, y).predict(X)
    else:
        raise Exception(f"Unimplemented black box for airquality: '{blackbox}'")
    if names:
        return X, y, nnames
    else:
        return X, y


def get_qm9(
    blackbox: Optional[Literal["nn"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Get the QM9 dataset as used in the "Slisemap application" paper: http://arxiv.org/abs/2310.15610"""
    path = find_path(data_dir) / "slisemap_phys.zip"
    if not path.exists():
        url = "https://www.edahelsinki.fi/papers/SI_slisemap_phys.zip"
        urlretrieve(url, path)
    df = pd.read_feather(ZipFile(path).open("SI/data/qm9_interpretable.feather"))
    df.drop(columns="index", inplace=True)
    X = df.to_numpy(np.float32)
    if blackbox == "nn":
        y = pd.read_feather(ZipFile(path).open("SI/data/qm9_nn.feather"))
        y = y["homo"].to_numpy()
    elif blackbox is None:
        y = pd.read_feather(ZipFile(path).open("SI/data/qm9_label.feather"))
        y = y["homo"].to_numpy()
    else:
        raise Exception(f"Unimplemented black box for qm9: '{blackbox}'")
    if normalise:
        X = StandardScaler().fit_transform(X)
        y = StandardScaler().fit_transform(y[:, None])[:, 0]
    if names:
        return X, y, df.columns
    else:
        return X, y


def get_jets(
    blackbox: Optional[Literal["rf"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Get the Jets dataset as used in the "Slisemap application" paper: http://arxiv.org/abs/2310.15610"""
    path = find_path(data_dir) / "slisemap_phys.zip"
    if not path.exists():
        url = "https://www.edahelsinki.fi/papers/SI_slisemap_phys.zip"
        urlretrieve(url, path)
    df = pd.read_feather(ZipFile(path).open("SI/data/jets.feather"))
    y = df["particle"]
    df.drop(columns=["index", "particle"], inplace=True)
    X = df.to_numpy(np.float32)
    if blackbox == "rf":
        y = pd.read_feather(ZipFile(path).open("SI/data/jets_rf.feather"))
        y = y.drop(columns="index").to_numpy()
    elif blackbox is None:
        y = (y == "Quark").astype(np.int32)
        y = np.eye(2, dtype=X.dtype)[y]
    else:
        raise Exception(f"Unimplemented black box for jets: '{blackbox}'")
    if normalise:
        X = StandardScaler().fit_transform(X)
    if names:
        return X, y, df.columns
    else:
        return X, y


def get_gas_turbine(
    blackbox: Optional[Literal["ada"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
    target: Literal["CO", "NOX"] = "NOX",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Get the Gas turbine dataset from https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set"""
    path = find_path(data_dir) / "gas_turbine.zip"
    if not path.exists():
        url = "https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip"
        urlretrieve(url, path)
    df = pd.concat(
        [
            pd.read_csv(ZipFile(path).open(f"gt_{year}.csv"))
            for year in (2011, 2012, 2013, 2014, 2015)
        ],
        ignore_index=True,
    )
    y = df[target].to_numpy(np.float32)
    df.drop(columns=["CO", "NOX"], inplace=True)
    X = df.to_numpy(np.float32)
    if normalise:
        X = StandardScaler().fit_transform(X)
        y = StandardScaler().fit_transform(y[:, None])[:, 0]
    if blackbox is None:
        pass
    elif blackbox.lower() in ("ada", "adaboost"):
        y = AdaBoostRegressor(random_state=42, learning_rate=0.03).fit(X, y).predict(X)
    else:
        raise Exception(f"Unimplemented black box for gas turbine: '{blackbox}'")
    if names:
        return X, y, df.columns
    else:
        return X, y


def get_rsynth(
    N: int = 100,
    M: int = 11,
    k: int = 3,
    s: float = 0.25,
    se: float = 0.1,
    seed: Union[None, int, np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression data

    Args:
        N (int, optional): Number of rows in X. Defaults to 100.
        M (int, optional): Number of columns in X. Defaults to 11.
        k (int, optional): Number of clusters (with their own true model). Defaults to 3.
        s (float, optional): Scale for the randomisation of the cluster centers. Defaults to 0.25.
        se (float, optional): Scale for the noise of y. Defaults to 0.1.
        seed (Union[None, int, np.random.RandomState], optional): Local random seed. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: cluster_ids[N], X[N,M+1], y[N], B[k,M+1].
    """
    if seed is None:
        npr = np.random
    elif isinstance(seed, np.random.RandomState):
        npr = seed
    else:
        npr = np.random.RandomState(seed)

    B = npr.normal(size=[k, M + 1])  # k x (M+1)
    while not _are_models_different(B):
        B = npr.normal(size=[k, M + 1])
    c = npr.normal(scale=s, size=[k, M])  # k X M
    while not _are_centroids_different(c, s * 0.5):
        c = npr.normal(scale=s, size=[k, M])
    j = npr.randint(k, size=N)  # N
    e = npr.normal(scale=se, size=N)  # N
    X = npr.normal(loc=c[j, :])  # N x M
    X = StandardScaler().fit_transform(X)
    y = (B[j, :-1] * X).sum(axis=1) + e + B[j, -1]
    return j, X, y, B


def get_rsynth2(
    N: int = 100,
    M: int = 11,
    k1: int = 3,
    k2: int = 3,
    s1: float = 2.0,
    s2: float = 5.0,
    se: float = 0.1,
    seed: Union[None, int, np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression data 2 (where half of the variables are adversarial)

    Args:
        N (int, optional): Number of rows in X. Defaults to 100.
        M (int, optional): Number of columns in X. Defaults to 11.
        k1 (int, optional): Number of true clusters (with their own true model). Defaults to 3.
        k2 (int, optional): Number of false clusters (not affecting y). Defaults to 3.
        s1 (float, optional): Scale for the randomisation of the true cluster centers. Defaults to 2.0.
        s2 (float, optional): Scale for the randomisation of the false cluster centers. Defaults to 5.0.
        se (float, optional): Scale for the noise of y. Defaults to 0.1.
        seed (Union[None, int, np.random.RandomState], optional): Local random seed. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: cluster_ids[N], X[N,M], y[N], B[k,M+1].
    """
    if seed is None:
        npr = np.random
    elif isinstance(seed, np.random.RandomState):
        npr = seed
    else:
        npr = np.random.RandomState(seed)

    B = npr.normal(size=[k1, M // 2 + 1])  # k1 x (M/2+1)
    while not _are_models_different(B):
        B = npr.normal(size=[k1, M + 1])
    c1 = npr.normal(scale=s1, size=[k1, M // 2])  # k1 X M/2
    while not _are_centroids_different(c1, s1 * 0.5):
        c1 = npr.normal(scale=s1, size=[k1, M // 2])
    c2 = npr.normal(scale=s2, size=[k2, M - M // 2])  # k2 X M/2
    while not _are_centroids_different(c2, s2 * 0.5):
        c1 = npr.normal(scale=s2, size=[k2, M - M // 2])
    j1 = npr.randint(k1, size=N)  # N
    j2 = npr.randint(k2, size=N)  # N
    e = npr.normal(scale=se, size=N)  # N
    X1 = npr.normal(loc=c1[j1])  # N x M/2
    X2 = npr.normal(loc=c2[j2])  # N x M/2
    X = np.concatenate((X1, X2), 1)  # N x M
    y = (B[j1, :-1] * X1).sum(axis=1) + e + B[j1, -1]
    B = np.concatenate((B[:, :-1], np.zeros((k1, M - M // 2)), B[:, -1:]), 1)
    return j1, X, y, B


def get_life(
    blackbox: Optional[Literal["nn"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Get the cleaned life dataset"""
    path = find_path(data_dir) / "life.csv"
    if not path.exists():
        import kagglehub

        dl_path = kagglehub.dataset_download("kumarajarshi/life-expectancy-who")
        path = list(Path(dl_path).glob("*.csv"))[0]
    df = pd.read_csv(path)

    df.dropna(inplace=True)

    df.drop(columns=["Country"], inplace=True)
    df.columns = df.columns.str.strip()

    # Map 'Status' to 0 and 1, rename it to 'Developed'
    df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})
    df.rename(columns={"Status": "Developed"}, inplace=True)

    # Separate features and target
    X = df.drop(columns="Life expectancy")
    y = df["Life expectancy"]

    if normalise:
        # Identify categorical and numerical features
        categorical = ["Developed"]
        numerical = X.columns.difference(categorical)

        # Define the preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical),
            ],
            remainder="passthrough",
        )

        # Preprocess features
        X = preprocessor.fit_transform(X)
        # Reconstruct X as a DataFrame with appropriate column names
        X = pd.DataFrame(
            X,
            columns=list(numerical) + categorical,
        )

        # Standardize the target variable
        y = StandardScaler().fit_transform(y.values.reshape(-1, 1)).squeeze()
    else:
        X = X.to_numpy()
        y = y.to_numpy()

    if blackbox == "nn":
        # Train a neural network regressor
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), solver="lbfgs", learning_rate_init=0.01
        ).fit(X, y)
        # Use the model's predictions as the target variable
        y = model.predict(X)
    elif blackbox is not None:
        raise Exception(f"Unimplemented black box for life expectancy: '{blackbox}'")

    if names:
        return X.to_numpy(dtype=np.float32), y, X.columns.tolist()
    else:
        return X.to_numpy(dtype=np.float32), y


def get_vehicle(
    blackbox: Optional[Literal["rf"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "../data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    path = find_path(data_dir) / "vehicle.csv"
    if not path.exists():
        import kagglehub

        dl_path = kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho")
        path = list(Path(dl_path).glob("*v4.csv"))[0]
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    # Drop unnecessary columns
    df.drop(
        columns=[
            "Make",
            "Model",
            "Color",
            "Location",
            "Length",
            "Width",
            "Height",
            "Drivetrain",
        ],
        inplace=True,
    )

    # Filter out rows with incomplete 'Max Power' and 'Max Torque' formats
    def incomplete_format(value):
        return bool(re.match(r"^\d+\.?\d*\s*bhp\s*@\s*$", value))

    df = df[
        ~df["Max Power"].apply(incomplete_format)
        & ~df["Max Torque"].apply(incomplete_format)
    ]

    # Preprocess categorical features
    # Filter and map 'Fuel Type'
    df = df[df["Fuel Type"].isin(["Diesel", "Petrol", "CNG"])]
    # Filter and map 'Previous Owner'
    if "Previous Owner" not in df.columns:
        df = df.rename(columns={"Owner": "Previous Owner"})
    df = df[df["Previous Owner"].isin(["First", "Second", "Third"])]
    df["Previous Owner"] = df["Previous Owner"].map(
        {"First": 1.0, "Second": 2.0, "Third": 3.0}
    )
    # Map 'Transmission' and 'Seller Type' to binary values
    df["Transmission"] = df["Transmission"].map({"Manual": 0, "Automatic": 1})
    df["Seller Type"] = df["Seller Type"].map(
        {"Individual": 1, "Corporate": 0, "Commercial Registration": 0}
    )
    df.rename(
        columns={"Transmission": "Automatic", "Seller Type": "Individual"}, inplace=True
    )
    # Map 'Engine' from "n cc" to n
    df["Engine"] = df["Engine"].str.extract(r"(\d+)").astype(int)

    # Transform and split 'Max Power' and 'Max Torque' into two columns each
    def transform_power_torque(value):
        # Handle format 'x@y' by converting it to 'x bhp @ y rpm'
        if re.match(r"^\d+\.?\d*@\d+\.?\d*$", value):
            parts = value.split("@")
            return f"{parts[0]} bhp @ {parts[1]} rpm"
        return value

    df["Max Power"] = df["Max Power"].apply(transform_power_torque)
    df["Max Torque"] = df["Max Torque"].apply(transform_power_torque)

    # Extract numerical values from 'Max Power' and 'Max Torque'
    def extract_values(value):
        parts = value.split(" ")
        v1 = float(parts[0])
        v2 = float(parts[3])
        return v1, v2

    df[["Power (bhp)", "Power (rpm)"]] = df["Max Power"].apply(
        lambda x: pd.Series(extract_values(x))
    )
    df[["Torque (Nm)", "Torque (rpm)"]] = df["Max Torque"].apply(
        lambda x: pd.Series(extract_values(x))
    )
    df.drop(columns=["Max Power", "Max Torque"], inplace=True)

    # Separate features and target variable
    X = df.drop(columns="Price")
    y = df["Price"]

    if normalise:
        # Identify categorical and numerical features
        categorical = ["Fuel Type", "Automatic", "Individual"]
        fuel_types = ["Diesel", "Petrol", "CNG"]
        numerical = X.columns.difference(categorical)

        # Define the preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "oh",
                    OneHotEncoder(sparse_output=False, categories=[fuel_types]),
                    ["Fuel Type"],
                ),
                ("num", StandardScaler(), numerical),
            ],
            remainder="passthrough",
        )

        # Preprocess features
        X_processed = preprocessor.fit_transform(X)
        # Construct column names after preprocessing
        feature_names = fuel_types + list(numerical) + ["Automatic", "Individual"]
        X_processed = pd.DataFrame(X_processed, columns=feature_names)

        # Standardize the target variable
        y_processed = StandardScaler().fit_transform(y.values.reshape(-1, 1)).squeeze()
    else:
        X_processed = X.copy()
        y_processed = y.copy()
        feature_names = X.columns.tolist()

    if names:
        return X_processed.to_numpy(dtype=np.float32), y_processed, feature_names
    else:
        return X_processed.to_numpy(dtype=np.float32), y_processed


def _are_models_different(B: np.ndarray, treshold: float = 0.5) -> bool:
    """Check if a set of linear models are different enough (using cosine similarity).

    Args:
        B (np.ndarray): Matrix where the rows are linear models.
        treshold (float, optional): Upper treshold for cosine similarity. Defaults to 0.5.

    Returns:
        bool: True if no pair of models are more similar than the treshold.
    """
    for i, j in combinations(range(B.shape[0]), 2):
        cosine_similarity = B[i] @ B[j] / (B[i] @ B[i] * B[j] @ B[j])
        if cosine_similarity > treshold:
            return False
    return True


def _are_centroids_different(c: np.ndarray, treshold: float = 0.5) -> bool:
    """Check if a set of linear models are different enough (using euclidean distance).

    Args:
        c (np.ndarray): Matrix where the rows are centroids.
        treshold (float, optional): Lower treshold for euclidean distance. Defaults to 0.5.

    Returns:
        bool: True if no pair of models are more similar than the treshold.
    """
    for i, j in combinations(range(c.shape[0]), 2):
        if np.sum((c[i] - c[j]) ** 2) < treshold**2:
            return False
    return True


if __name__ == "__main__":
    print("Downloading all datasets...")
    get_boston("svm")
    get_fashion_mnist("cnn")
    get_iris()
    get_airquality()
    get_mnist("cnn")
    get_emnist("cnn")
    get_spam("rf")
    get_higgs("gb")
    get_covertype("lb")
    get_autompg("svr")
    get_qm9("nn")
    get_jets("rf")
    get_life("nn")
    get_vehicle(None)

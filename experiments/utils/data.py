import os
import openml
import numpy as np
import pandas as pd
from scipy.io import arff
from pathlib import Path
from urllib.request import urlretrieve
from typing import Union, Sequence, Optional, Tuple, List, Literal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from zipfile import ZipFile
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
import kagglehub
import re
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations
from configparser import ConfigParser

curr_path = Path(__file__)
config = ConfigParser()
config.read(str(curr_path.parent.parent.parent / "config.ini"))
apikey = config["Keys"]["OPENML_APIKEY"]


def find_path(dir_name: str = "data") -> Path:
    """Find the path to the directory that stores data files."""
    path = Path(dir_name)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_dataset(
    id: int = None, cache_dir: Union[str, Path] = None
) -> openml.OpenMLDataset:
    """Fetch dataset from OpenML API and store it in cache directory."""

    # Set the API key, and cache directory for OpenML connector
    openml.config.apikey = apikey
    openml.config.cache_directory = cache_dir

    # Fetch dataset with OpenML API and store it in cache directory
    return openml.datasets.get_dataset(
        id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )


def _get_predictions(
    id: int = None,
    columns: Sequence[str] = None,
    cache_dir: Union[str, Path] = None,
) -> np.ndarray:
    """Fetch predictions from OpenML API and store them in cache directory."""

    # Set the API key for OpenML connector, and create the cache directory if it doesn't exist
    openml.config.apikey = apikey
    dir = Path(cache_dir) / "org" / "openml" / "www" / "runs" / str(id)
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / "predictions.arff"

    # Fetch the OpenML run object and download the predictions file
    run = openml.runs.get_run(id, False)
    if not path.exists():
        urlretrieve(run.predictions_url, path)

    # Load and extract the predictions by specified columns, stack them into a NumPy array
    data, _ = arff.loadarff(path)
    pred = np.stack(tuple(data[c] for c in columns), axis=-1)

    # Return sorted predictions based on row_id for consistency
    return pred[np.argsort(data["row_id"])]


def get_higgs(
    blackbox: Optional[Literal["gb"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Load the HIGGS dataset from OpenML, and return the features and target labels."""
    # Download and cache the dataset https://archive.ics.uci.edu/ml/datasets/HIGGS in 'explainreduce/data' directory
    dir = find_path(data_dir)
    dataset = _get_dataset(23512, dir)

    # Extract the features and target labels from the dataset, the last row is removed due to missing target label
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    X, y = X.to_numpy()[:-1], y.to_numpy().astype(int)[:-1]

    # Load the predictions from the specified black box model
    if blackbox is None:
        Y = np.eye(2, dtype=X.dtype)[y]
    elif blackbox.lower() in ("gb", "gradient boosting", "gradientboosting"):
        Y = _get_predictions(9907793, (f"confidence.{c:d}" for c in range(2)), dir)
        Y = Y[:-1]
    else:
        raise Exception(f"Unimplemented black box for higgs: '{blackbox}'")

    # Normalise the input data if required, and return the features, target labels, and attribute names (if requested)
    if normalise:
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
    """Load the AirQuality dataset from OpenML, and return the features and target labels."""
    # Download and cache the dataset
    path = find_path(data_dir) / "AQ_cleaned_version.csv"
    if not path.exists():
        url = "https://raw.githubusercontent.com/edahelsinki/drifter/master/TiittanenOHP2019-code/data/AQ_cleaned_version.csv"
        urlretrieve(url, path)
    AQ = pd.read_csv(path)

    # Extract the features and target labels from the dataset
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
    X = AQ[columns].to_numpy()
    y = AQ["CO(GT)"].to_numpy()

    # Apply black box model and normalise the input data if required
    if normalise:
        X = StandardScaler().fit_transform(X)
        y = StandardScaler().fit_transform(y[:, None])[:, 0]
    if blackbox is None:
        pass
    elif blackbox.lower() in ("rf", "random forest", "randomforest"):
        y = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X, y).predict(X)
    else:
        raise Exception(f"Unimplemented black box for airquality: '{blackbox}'")

    # Return the features, target labels, and attribute names (if requested)
    if names:
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
    # Download and cache the dataset
    path = find_path(data_dir) / "slisemap_phys.zip"
    if not path.exists():
        url = "https://www.edahelsinki.fi/papers/SI_slisemap_phys.zip"
        urlretrieve(url, path)

    # Load the dataset and remove the index column
    df = pd.read_feather(ZipFile(path).open("SI/data/qm9_interpretable.feather"))
    df.drop(columns="index", inplace=True)

    # Extract the features and target labels from the dataset
    X = df.to_numpy(np.float32)
    if blackbox == "nn":
        y = pd.read_feather(ZipFile(path).open("SI/data/qm9_nn.feather"))
        y = y["homo"].to_numpy()
    elif blackbox is None:
        y = pd.read_feather(ZipFile(path).open("SI/data/qm9_label.feather"))
        y = y["homo"].to_numpy()
    else:
        raise Exception(f"Unimplemented black box for qm9: '{blackbox}'")

    # Normalise if required
    if normalise:
        X = StandardScaler().fit_transform(X)
        y = StandardScaler().fit_transform(y[:, None])[:, 0]

    # Return the features, target labels, and attribute names (if requested)
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
    # Download and cache the dataset
    path = find_path(data_dir) / "slisemap_phys.zip"
    if not path.exists():
        url = "https://www.edahelsinki.fi/papers/SI_slisemap_phys.zip"
        urlretrieve(url, path)

    # Load the dataset, extract features and target label
    df = pd.read_feather(ZipFile(path).open("SI/data/jets.feather"))
    y = df["particle"]
    df.drop(columns=["index", "particle"], inplace=True)
    X = df.to_numpy(np.float32)

    # Adjust target label based on black box model
    if blackbox == "rf":
        y = pd.read_feather(ZipFile(path).open("SI/data/jets_rf.feather"))
        y = y.drop(columns="index").to_numpy()
    elif blackbox is None:
        y = (y == "Quark").astype(np.int32)
        y = np.eye(2, dtype=X.dtype)[y]
    else:
        raise Exception(f"Unimplemented black box for jets: '{blackbox}'")

    # Normalise if required
    if normalise:
        X = StandardScaler().fit_transform(X)

    # Return the features, target labels, and attribute names (if requested)
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
    # Download and cache the dataset
    path = find_path(data_dir) / "gas_turbine.zip"
    if not path.exists():
        url = "https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip"
        urlretrieve(url, path)

    # Load the dataset, extract features and target label
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

    # Normalise if required, and apply black box model
    if normalise:
        X = StandardScaler().fit_transform(X)
        y = StandardScaler().fit_transform(y[:, None])[:, 0]
    if blackbox is None:
        pass
    elif blackbox.lower() in ("ada", "adaboost"):
        y = AdaBoostRegressor(random_state=42, learning_rate=0.03).fit(X, y).predict(X)
    else:
        raise Exception(f"Unimplemented black box for gas turbine: '{blackbox}'")

    # Return the features, target labels, and attribute names (if requested)
    if names:
        return X, y, df.columns
    else:
        return X, y


def get_life(
    blackbox: Optional[Literal["nn"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Get the cleaned life dataset"""
    # Download and cache the dataset
    path = find_path(data_dir) / "life.csv"
    if not path.exists():
        dl_path = kagglehub.dataset_download("kumarajarshi/life-expectancy-who")
        path = list(Path(dl_path).glob("*.csv"))[0]

    # Load the dataset, drop missing values, and clean the columns
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df.drop(columns=["Country"], inplace=True)
    df.columns = df.columns.str.strip()
    df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})
    df.rename(columns={"Status": "Developed"}, inplace=True)

    # Extract features and target
    X = df.drop(columns="Life expectancy")
    y = df["Life expectancy"]

    # Normalise if required
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

        # Preprocess features and target
        X = preprocessor.fit_transform(X)
        X = pd.DataFrame(
            X,
            columns=list(numerical) + categorical,
        )
        y = StandardScaler().fit_transform(y.values.reshape(-1, 1)).squeeze()
    else:
        X = X.to_numpy()
        y = y.to_numpy()

    # Train a neural network regressor if required
    if blackbox == "nn":
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), solver="lbfgs", learning_rate_init=0.01
        ).fit(X, y)
        y = model.predict(X)
    elif blackbox is not None:
        raise Exception(f"Unimplemented black box for life expectancy: '{blackbox}'")

    # Return the features, target labels, and attribute names (if requested)
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
    """Get the vehicle dataset from https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho"""
    # Download and cache the dataset
    path = find_path(data_dir) / "vehicle.csv"
    if not path.exists():
        dl_path = kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho")
        path = list(Path(dl_path).glob("*v4.csv"))[0]

    # Load the dataset, drop missing values, and clean the columns
    df = pd.read_csv(path)
    df.dropna(inplace=True)
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

    # Normalise if required
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

        # Preprocess features and targets
        X_processed = preprocessor.fit_transform(X)
        feature_names = fuel_types + list(numerical) + ["Automatic", "Individual"]
        X_processed = pd.DataFrame(X_processed, columns=feature_names)
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


def get_churn(
    blackbox: Optional[Literal["rf"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    # https://archive.ics.uci.edu/ml/datasets/spambase
    dir = find_path(data_dir)
    dataset = _get_dataset(40701, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    cat_columns = X.select_dtypes(["category"]).columns
    X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
    X, y = X.to_numpy(), y.to_numpy().astype(int)
    if blackbox is None:
        Y = np.eye(2, dtype=X.dtype)[y]
    else:
        raise Exception(f"Unimplemented black box for spam: '{blackbox}'")
    if normalise:
        X = StandardScaler().fit_transform(X)
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_telescope(
    blackbox: Optional[Literal["svc"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    # https://archive.ics.uci.edu/ml/datasets/spambase
    dir = find_path(data_dir)
    dataset = _get_dataset(1120, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    X, y = X.to_numpy(), (y == "g").astype(int).to_numpy()
    if blackbox is None:
        Y = np.eye(2, dtype=X.dtype)[y]
    else:
        raise Exception(f"Unimplemented black box for telescope: '{blackbox}'")
    if normalise:
        X = StandardScaler().fit_transform(X)
    if names:
        return X, Y, attribute_names
    else:
        return X, Y


def get_adult(
    blackbox: Optional[Literal["nn"]] = None,
    names: bool = False,
    normalise: bool = True,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    # https://archive.ics.uci.edu/ml/datasets/spambase
    dir = find_path(data_dir)
    dataset = _get_dataset(1590, dir)
    X, y, _, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    missing_vals = X.isna().any(axis=1)
    X, y = X.loc[~missing_vals], y.loc[~missing_vals]
    cat_columns = X.select_dtypes(["category"]).columns
    X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
    X, y = X.to_numpy(), (y == ">50K").astype(int).to_numpy()
    if blackbox is None:
        Y = np.eye(2, dtype=X.dtype)[y]
    else:
        raise Exception(f"Unimplemented black box for telescope: '{blackbox}'")
    if normalise:
        X = StandardScaler().fit_transform(X)
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
    X, y = X.to_numpy(), y.to_numpy().astype(int)
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


if __name__ == "__main__":
    print("Downloading all datasets...")
    get_airquality()
    get_spam("rf")
    get_higgs("gb")
    get_qm9("nn")
    get_jets("rf")
    get_life("nn")
    get_vehicle(None)
    get_adult(None)
    get_churn(None)
    get_telescope(None)

import os
import openml
import numpy as np
from dotenv import load_dotenv
from scipy.io import arff
from pathlib import Path
from urllib.request import urlretrieve
from typing import Union, Sequence, Optional, Tuple, List, Literal
from sklearn.preprocessing import StandardScaler


def find_path(dir_name: str = "data") -> Path:
    """Find the path to the directory that stores data files."""
    path = Path(dir_name)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_dataset(
    apikey: str = None, id: int = None, cache_dir: Union[str, Path] = None
) -> openml.OpenMLDataset:
    """Fetch dataset from OpenML API and store it in cache directory."""
    # Load API key from environment variable if not provided, and check if it's available
    if apikey is None:
        load_dotenv()
        apikey = os.getenv("OPENML_APIKEY")
    if not apikey:
        raise ValueError("API key is required.")

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
    apikey: str = None,
    id: int = None,
    columns: Sequence[str] = None,
    cache_dir: Union[str, Path] = None,
) -> np.ndarray:
    """Fetch predictions from OpenML API and store them in cache directory."""
    # Load API key from environment variable if not provided, and check if it's available
    if apikey is None:
        load_dotenv()
        apikey = os.getenv("OPENML_APIKEY")
    if not apikey:
        raise ValueError("API key is required.")

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
    """Load the HIGGS dataset from OpenML, and return the input and output data."""
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

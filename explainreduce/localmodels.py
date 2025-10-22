import torch
from torch.optim import LBFGS
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from slisemap.local_models import LinearRegression, LogisticRegression
import slise
import warnings
import pandas as pd
from slisemap import Slisemap, Slipmap
from typing import Any, List, Callable, Tuple, Dict, Union
import explainreduce.utils as utils
from abc import ABC, abstractmethod
from sklearn.exceptions import NotFittedError
from tqdm import tqdm
import statistics
from lore_sa.dataset import TabularDataset
from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.lore import TabularRandomGeneratorLore
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.neighgen import RandomGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
import sys
from pathlib import Path
from configparser import ConfigParser

curr_path = Path(__file__)
config = ConfigParser()
config.read(str(curr_path.parent.parent / "config.ini"))
GLOCALX_PATH = config["Paths"]["GLOCALX_PATH"]
sys.path.append(GLOCALX_PATH)


class Explainer(ABC):
    """Abstract base class for an Explainer object, representing a local explanation method.

    Attributes
    ----------
    X: torch.Tensor
        Training items as rows, features as columns.
    y: torch.Tensor
        Training labels.
    classifier: bool, default False.
        Flag whether the explainer is for classification or regression.
    loss_fn: function, default torch.nn.MSELoss
        A loss function of the form L(y, yhat).
    black_box_model: function, default None
        The "explainee" function of the form F(X) = y. Optional.
    explainer_kwargs:
        Specific settings for a given local explanation method.
    mapping:
        A dictionary mapping the data items (rows in the X tensor) to local models.
    local_models:
        A list of functions f of the form ŷ = f(X).
    vector_representation:
        A vector representation of the local models (such as coefficients of a linear
        model).
    is_fit:
        A flag to mark whether or not the Explainer has been fit.
    L:
        A cached loss matrix of the shape (data items x local models).
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        classifier=False,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        black_box_model=None,
        dtype=torch.float32,
        verbose=False,
        **explainer_kwargs,
    ) -> None:
        self.X = torch.as_tensor(X, dtype=dtype)
        if self.X.ndim < 2:
            self.X = self.X[:, None]
        self.y = torch.as_tensor(y, dtype=dtype)
        if self.y.ndim < 2:
            self.y = self.y[:, None]
            if classifier:
                self.y = torch.hstack((self.y, 1.0 - self.y))
        self.mapping = None
        self.local_models = None
        self.classifier = classifier
        self.vector_representation = None
        self.is_fit = False
        if loss_fn is None:
            if self.classifier:
                self.loss_fn = logistic_regression_loss
            else:
                self.loss_fn = torch.nn.MSELoss(reduction="none")
        else:
            self.loss_fn = loss_fn
        self.L = None
        self.black_box_model = black_box_model
        self.dtype = dtype
        self.verbose = verbose
        self.explainer_kwargs = explainer_kwargs

    def fit(self) -> None:
        """Fit the Explainer object. Train local models, generate a mapping and
        calculate vector representations.
        """
        self.local_models, self.mapping = self._generate_local_models()
        self.vector_representation = self._generate_vector_representation()
        self.is_fit = True

    def clone_reduced(
        self,
        local_models: List[Callable[[torch.Tensor], torch.Tensor]],
        mapping: Dict[int, int],
        vector_representation: torch.Tensor,
        proxy_ids=None,
    ):
        """Clone this explainer with a new mapping and local model set."""
        if not self.is_fit:
            raise NotFittedError(
                "Explainer not fitted! Please call .fit() before .clone_reduced()!"
            )
        reduced = self.__class__(
            self.X, self.y, self.classifier, dtype=self.dtype, **self.explainer_kwargs
        )
        reduced.loss_fn = self.loss_fn
        reduced.black_box_model = self.black_box_model
        reduced.local_models = local_models
        reduced.mapping = mapping
        reduced.vector_representation = vector_representation
        reduced.is_fit = True
        return reduced

    @abstractmethod
    def _generate_local_models(
        self,
    ) -> Tuple[List[Callable[[torch.Tensor], torch.Tensor]], Dict[int, int]]:
        """Generate a list of local models and a mapping from training items to local models."""
        pass

    @abstractmethod
    def _generate_vector_representation(self) -> torch.Tensor:
        """Generate a vector representation of the local models."""
        pass

    def _map_to_training_data(self, X_new: torch.Tensor, mode="data_euclidean"):
        """Map new instances to training items."""
        X_new = torch.as_tensor(X_new)
        match mode:
            case "data_euclidean":
                D = torch.cdist(self.X, X_new)
                neighbours = torch.argmin(D, axis=0).tolist()
                return neighbours
            case _:
                raise NotImplementedError(f"Mapping mode {mode} not implemented!")

    def predict(self, X_new: torch.Tensor, mapping_mode="data_euclidean"):
        """Predict using the local models."""
        if not self.is_fit:
            raise NotFittedError(
                "Explainer not fitted! Please call .fit() before .predict()!"
            )
        mapped = self._map_to_training_data(X_new, mode=mapping_mode)
        yhat = torch.atleast_2d(
            torch.empty((X_new.shape[0], self.y.shape[1]), dtype=self.dtype)
        )
        for i in range(X_new.shape[0]):
            yhat[i, :] = self.local_models[self.mapping[mapped[i]]](X_new[i, None])
        return yhat

    # Changed to rows representing local models and columns representing training items, please keep it in the merge!
    def get_L(self, X: torch.Tensor = None, y: torch.Tensor = None):
        """Fetch or generate a loss matrix L
        Rows correspond to local models, columns correspond to instances.
        """
        if not self.is_fit:
            raise NotFittedError(
                "Explainer not fitted! Please call .fit() before .get_L()!"
            )
        if (X is None and y is None) and self.L is not None:
            return self.L
        if X is None:
            X = self.X
        if y is None:
            y = self.y.clone()
        L = torch.zeros((len(self.local_models), X.shape[0]), dtype=self.dtype)
        if y.ndim < 2:
            y = y[:, np.newaxis]
        for j in range(len(self.local_models)):
            yhat = self.local_models[j](X)
            if yhat.ndim < 2:
                yhat = yhat[:, np.newaxis]
            if yhat.shape[1] > y.shape[1] and self.classifier:
                y = torch.hstack((y, 1.0 - y))
            loss_vector = self.loss_fn(yhat.float(), y.float())
            # for multiple classification, take the mean loss across items
            # TODO: maybe there is better way? make L to be a higher dimensional vector?
            if loss_vector.ndim > 1:
                loss_vector = torch.mean(loss_vector, dim=-1)
            L[j, :] = loss_vector
        # cache the result for future use
        if X is None and y is None:
            self.L = L
        return L

    def mapped_vector_representation(self):
        """Generate a vector representation in the shape of X_train."""
        return self.vector_representation[list(self.mapping.values()), :]

    def black_box_predict(self):
        if self.black_box_model is None:
            raise ValueError("Black-box model not set!")
        if (self.classifier) and (hasattr(self.black_box_model, "predict_proba")):
            return self.black_box_model.predict_proba
        else:
            return self.black_box_model.predict


class SLISEMAPExplainer(Explainer):

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        classifier=False,
        loss_fn=None,
        dtype=torch.float32,
        verbose=False,
        **explainer_kwargs,
    ) -> None:
        super().__init__(
            X, y, classifier, loss_fn, dtype=dtype, verbose=verbose, **explainer_kwargs
        )

    def _generate_local_models(
        self,
    ) -> Tuple[List[Callable[[torch.Tensor], torch.Tensor]] | Dict[int, int]]:
        y = self.y
        if self.classifier and y.shape[1] == 1:
            y = torch.hstack((self.y, torch.ones_like(self.y) - self.y))
        lm = LogisticRegression if self.classifier else LinearRegression
        self.sm = Slisemap(
            self.X,
            y,
            local_model=lm,
            dtype=self.dtype,
            **self.explainer_kwargs,
        )
        self.sm.optimise(verbose=self.verbose)
        B = self.sm.get_B(False)

        def create_exp(i):
            return lambda X: self.sm.local_model(self.sm._as_new_X(X), B[None, i])[
                0, ...
            ]

        if self.verbose:
            local_models = [create_exp(i) for i in tqdm(range(self.X.shape[0]))]
        else:
            local_models = [create_exp(i) for i in range(self.X.shape[0])]
        mapping = {i: i for i in range(self.X.shape[0])}
        return local_models, mapping

    def _generate_vector_representation(self) -> torch.Tensor:
        return self.sm.get_B(numpy=False)


class SLIPMAPExplainer(SLISEMAPExplainer):

    def _generate_local_models(
        self,
    ) -> Tuple[List[Callable[[torch.Tensor], torch.Tensor]] | Dict[int, int]]:

        y = self.y
        if self.classifier and y.shape[1] == 1:
            y = torch.hstack((self.y, torch.ones_like(self.y) - self.y))
        lm = LogisticRegression if self.classifier else LinearRegression
        self.sm = Slipmap(
            self.X,
            y,
            local_model=lm,
            dtype=self.dtype,
            **self.explainer_kwargs,
        )
        self.sm.optimise(verbose=self.verbose)
        B = self.sm.get_B(False)

        def create_exp(i):
            return lambda X: self.sm.local_model(self.sm._as_new_X(X), B[None, i])[
                0, ...
            ]

        if self.verbose:
            local_models = [create_exp(i) for i in tqdm(range(self.X.shape[0]))]
        else:
            local_models = [create_exp(i) for i in range(self.X.shape[0])]
        mapping = {i: i for i in range(self.X.shape[0])}
        return local_models, mapping


class SLISEExplainer(Explainer):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
        classifier=False,
        loss_fn=None,
        dtype=torch.float32,
        verbose=False,
        **explainer_kwargs,
    ) -> None:
        super().__init__(
            X, y, classifier, loss_fn, dtype=dtype, verbose=verbose, **explainer_kwargs
        )
        self.epsilon = epsilon

    def _generate_local_models(
        self,
    ) -> Tuple[List[Callable[[torch.Tensor], torch.Tensor]] | Dict[int, int]]:

        if self.classifier:
            y = self.y[:, 0]
        self.s = slise.SliseExplainer(
            self.X, y, epsilon=self.epsilon**0.5, **self.explainer_kwargs
        )

        def create_exp(i: int):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.s.explain(i)
            if self.classifier:
                return lambda X: np.stack(
                    (self.s.predict(X), 1.0 - self.s.predict(X)), -1
                )
            return self.s.predict

        if self.verbose:
            local_models = [create_exp(i) for i in tqdm(range(self.X.shape[0]))]
        else:
            local_models = [create_exp(i) for i in range(self.X.shape[0])]
        mapping = {i: i for i in range(self.X.shape[0])}
        return local_models, mapping

    def _generate_vector_representation(self) -> torch.Tensor:

        def get_coeffs(i: int):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.s.explain(i)
            return self.s.coefficients()

        coeffs = torch.empty_like(self.X)
        for i in range(self.X.shape[0]):
            coeffs[i, :] = get_coeffs(i)
        return coeffs


class SmoothGradExplainer(Explainer):

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        classifier=False,
        loss_fn=None,
        black_box_model=None,
        dtype=torch.float32,
        verbose=False,
        **explainer_kwargs,
    ) -> None:
        super().__init__(
            X,
            y,
            classifier,
            loss_fn,
            black_box_model,
            dtype=dtype,
            verbose=verbose,
            **explainer_kwargs,
        )

    def _generate_local_models(
        self,
    ) -> Tuple[List[Callable[[torch.Tensor], torch.Tensor]], Dict[int, int]]:

        B = get_local_models_smoothgrad(
            self.X.numpy(),
            self.black_box_model,
            "regression" if not self.classifier else "classification",
            dtype=self.dtype,
            **self.explainer_kwargs,
        )
        self.B = B

        def create_exp(i):
            if self.classifier:
                return lambda X: logistic_regression(X, B, i)
            else:
                return lambda X: (B[i, :-1] @ X.T + B[i, -1]).clone().detach()

        if self.verbose:
            local_models = [create_exp(i) for i in tqdm(range(self.X.shape[0]))]
        else:
            local_models = [create_exp(i) for i in range(self.X.shape[0])]
        mapping = {i: i for i in range(self.X.shape[0])}
        return local_models, mapping

    def _generate_vector_representation(self) -> torch.Tensor:
        return self.B


class SmoothGradTorchExplainer(Explainer):

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        classifier=False,
        loss_fn=None,
        black_box_model=None,
        dtype=torch.float32,
        verbose=False,
        **explainer_kwargs,
    ) -> None:
        super().__init__(
            X,
            y,
            classifier,
            loss_fn,
            black_box_model,
            dtype=dtype,
            verbose=verbose,
            **explainer_kwargs,
        )

    def _generate_local_models(
        self,
    ) -> Tuple[List[Callable[[torch.Tensor], torch.Tensor]], Dict[int, int]]:

        B = get_local_models_smoothgrad(
            self.X,
            self.black_box_model,
            "regression" if not self.classifier else "classification",
            **self.explainer_kwargs,
        )
        self.B = B

        def create_exp(i):
            if self.classifier:
                return lambda X: logistic_regression(X, B, i)
            else:
                return lambda X: (B[i, :-1] @ X.T + B[i, -1]).clone().detach()

        if self.verbose:
            local_models = [create_exp(i) for i in tqdm(range(self.X.shape[0]))]
        else:
            local_models = [create_exp(i) for i in range(self.X.shape[0])]
        mapping = {i: i for i in range(self.X.shape[0])}
        return local_models, mapping

    def _generate_vector_representation(self) -> torch.Tensor:
        return self.B


class LIMEExplainer(Explainer):

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        classifier=False,
        loss_fn=None,
        black_box_model=None,
        dtype=torch.float32,
        verbose=False,
        num_samples=5_000,
        **explainer_kwargs,
    ) -> None:
        super().__init__(
            X,
            y,
            classifier,
            loss_fn,
            black_box_model,
            dtype=dtype,
            verbose=verbose,
            **explainer_kwargs,
        )
        self.num_samples = num_samples

    def _generate_local_models(
        self,
    ) -> Tuple[List[Callable[[torch.Tensor], torch.Tensor]], Dict[int, int]]:

        B, local_models, LIME_object = get_lime(
            self.X,
            self.y,
            self.black_box_model,
            classifier=self.classifier,
            dtype=self.dtype,
            verbose=self.verbose,
            num_samples=self.num_samples,
            **self.explainer_kwargs,
        )
        self.B = B

        mapping = {i: i for i in range(self.X.shape[0])}
        self._LIME_explainer = LIME_object
        return local_models, mapping

    def _generate_vector_representation(self) -> torch.Tensor:
        return self.B


class KernelSHAPExplainer(Explainer):

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        classifier=False,
        loss_fn=None,
        black_box_model=None,
        dtype=torch.float32,
        verbose=False,
        **explainer_kwargs,
    ) -> None:
        super().__init__(
            X,
            y,
            classifier,
            loss_fn,
            black_box_model,
            dtype=dtype,
            verbose=verbose,
            **explainer_kwargs,
        )

    def _generate_local_models(
        self,
    ) -> Tuple[List[Callable[[torch.Tensor], torch.Tensor]], Dict[int, int]]:

        B = get_local_models_kernelshap(
            self.X.numpy(),
            self.black_box_model,
            "regression" if not self.classifier else "classification",
            dtype=self.dtype,
            **self.explainer_kwargs,
        )
        self.B = B

        def create_exp(i):
            if self.classifier:
                return lambda X: logistic_regression(X, B, i)
            else:
                return lambda X: (B[i, :-1] @ X.T + B[i, -1]).clone().detach()

        if self.verbose:
            local_models = [create_exp(i) for i in tqdm(range(self.X.shape[0]))]
        else:
            local_models = [create_exp(i) for i in range(self.X.shape[0])]
        mapping = {i: i for i in range(self.X.shape[0])}
        return local_models, mapping

    def _generate_vector_representation(self) -> torch.Tensor:
        return self.B


class KernelSHAPExplainerLegacy(Explainer):

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        classifier=False,
        loss_fn=None,
        black_box_model=None,
        **explainer_kwargs,
    ) -> None:
        super().__init__(X, y, classifier, loss_fn, black_box_model, **explainer_kwargs)

    def _generate_local_models(
        self,
    ) -> Tuple[List[Callable[[torch.Tensor], torch.Tensor]], Dict[int, int]]:

        B, local_models = get_shap(
            self.X,
            self.y,
            self.black_box_model,
            classifier=self.classifier,
            **self.explainer_kwargs,
        )
        self.B = B
        mapping = {i: i for i in range(self.X.shape[0])}
        return local_models, mapping

    def _generate_vector_representation(self) -> torch.Tensor:
        return self.B


class GlobalLinearExplainer(Explainer):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        classifier=False,
        loss_fn=None,
        black_box_model=None,
        dtype=torch.float32,
        verbose=False,
        **explainer_kwargs,
    ) -> None:
        super().__init__(
            X,
            y,
            classifier,
            loss_fn,
            black_box_model,
            dtype=dtype,
            verbose=False,
            **explainer_kwargs,
        )
        self.lasso = explainer_kwargs.get("lasso", 0.0)
        self.ridge = explainer_kwargs.get("ridge", 0.0)
        self.intercept = explainer_kwargs.get("intercept", True)

    def _generate_local_models(
        self,
    ) -> Tuple[List[Callable[[torch.Tensor], torch.Tensor]], Dict[int, int]]:

        self.B = get_single_global_model(
            self.X,
            self.y,
            self.classifier,
            self.loss_fn,
            self.lasso,
            self.ridge,
            intercept=self.intercept,
        )

        def local_model(X: torch.Tensor):
            if self.intercept:
                X = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
            if self.classifier:
                yhat = torch.sigmoid(X @ self.B)[:, None]
                return torch.hstack((yhat, 1.0 - yhat))
            else:
                return (X @ self.B)[:, None]

        local_models = [local_model] * self.X.shape[0]
        mapping = {i: 0 for i in range(self.X.shape[0])}
        return local_models, mapping

    def _generate_vector_representation(self) -> torch.Tensor:
        return self.B.repeat(self.X.shape[0], 1)


class LORERuleExplainer(Explainer):
    try:
        from models import Rule
    except ModuleNotFoundError as e:
        e.add_note("Importing GLocalX failed.")
        e.add_note(
            f"Ensure GLocalX path is set correctly in config.ini (currently {GLOCALX_PATH})."
        )
        raise

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        classifier=True,
        loss_fn=None,
        black_box_model=None,
        dtype=torch.float32,
        verbose=False,
        strategy="majority",
        num_instances=1000,
        **explainer_kwargs,
    ) -> None:
        if not classifier:
            raise ValueError("LOREExplainer only available for classifier models!")
        # convert y to labels if necessary; LORE expects clean labels
        y = torch.round(torch.as_tensor(y))
        super().__init__(
            X,
            y,
            classifier,
            loss_fn,
            black_box_model,
            dtype=dtype,
            verbose=verbose,
            **explainer_kwargs,
        )
        # set up dataset
        df = pd.DataFrame(utils.tonp(X))
        df.loc[:, "target"] = utils.tonp(y[:, 1])
        df.loc[:, "target"] = df["target"].astype("category")
        df.columns = [str(c) for c in df.columns]
        self.dataset = TabularDataset(df, class_name="target")
        # set up LORE-specific objects
        self.bbox = sklearn_classifier_bbox.sklearnBBox(black_box_model)
        self.lore = TabularRandomGeneratorLore(self.bbox, self.dataset)
        self.tabular_enc = ColumnTransformerEnc(self.dataset.descriptor)
        self.gen = RandomGenerator(
            bbox=self.bbox, dataset=self.dataset, encoder=self.tabular_enc, ocr=0.1
        )
        self.verbose = verbose
        self.strategy = strategy
        self.num_instances = num_instances
        if self.strategy == "majority":
            self.mode_y = int(statistics.mode(self.y[:, 1]))

    def _generate_local_models(
        self,
    ) -> Tuple[List[Callable[[torch.Tensor], torch.Tensor]], Dict[int, int]]:

        def _create_exp(i: int):
            x = self.dataset.df.iloc[i][:-1]
            z = self.tabular_enc.encode([x.values])[0].astype(float)

            neighbour = self.gen.generate(
                z,
                self.num_instances,
                self.dataset.descriptor,
                self.tabular_enc,
                **self.explainer_kwargs,
            )

            neighb_train_X = self.tabular_enc.decode(neighbour)
            neighb_train_y = self.bbox.predict(neighb_train_X)
            if neighb_train_y.ndim == 2:
                neighb_train_y = neighb_train_y[:, 1]
            # encode the target class to the surrogate model
            neighb_train_yz = self.tabular_enc.encode_target_class(
                neighb_train_y.reshape(-1, 1)
            ).squeeze()

            dt = DecisionTreeSurrogate()
            dt.train(neighbour, neighb_train_yz)
            rule = dt.get_rule(z, self.tabular_enc)

            return rule.to_dict()

        if self.verbose:
            rules = [_create_exp(i) for i in tqdm(range(self.X.shape[0]))]
        else:
            rules = [_create_exp(i) for i in range(self.X.shape[0])]
        self.rules = self.lore_to_glocalx(rules)
        mapping = {i: i for i in range(self.X.shape[0])}
        local_models = [
            lambda x: self._predict_from_rule(self.rules[i], x)
            for i in range(len(self.rules))
        ]
        return local_models, mapping

    def _predict_from_rule(self, rule: Rule, X: torch.Tensor):
        valid_train_items = torch.tensor(
            [rule.covers(X[i, :]) for i in range(X.shape[0])]
        )
        yhat = torch.zeros((X.shape[0], 2), dtype=self.dtype)
        yhat[valid_train_items, rule.consequence] = 1.0
        if self.strategy == "negation":
            yhat[torch.logical_not(valid_train_items), 1 - rule.consequence] = 1.0
        elif self.strategy == "majority":
            yhat[torch.logical_not(valid_train_items), self.mode_y] = 1.0
        return yhat

    def predict(self, X_new: torch.Tensor, mode=None):
        ruleset_losses = torch.zeros(len(self.local_models))

        for i, rule in enumerate(self.local_models):
            yhat = rule(self.X)
            loss = ((torch.sqrt(yhat) - torch.sqrt(self.y)) ** 2).sum(dim=-1) * 0.5
            ruleset_losses[i] = torch.mean(loss)

        yhat_new = np.zeros((X_new.shape[0], 2))
        for i in range(X_new.shape[0]):
            item = X_new[i, :]
            applicable_rules = [j for j, r in enumerate(self.rules) if r.covers(item)]
            relevant_losses = ruleset_losses[applicable_rules]
            min_loss = torch.min(relevant_losses)
            best_rules = torch.where(torch.isclose(relevant_losses, min_loss))[0]
            rule_idx = np.random.choice(utils.tonp(best_rules))
            chosen_rule = self.rules[rule_idx]
            yhat_new[i, chosen_rule.consequence] = 1.0
        return torch.as_tensor(yhat_new, dtype=self.dtype)

    def clone_reduced(
        self,
        local_models: List[Callable[[torch.Tensor], torch.Tensor]],
        mapping: Dict[int, int],
        vector_representation: torch.Tensor,
        proxy_ids=None,
    ):
        """Clone this explainer with a new mapping and local model set."""
        if not self.is_fit:
            raise NotFittedError(
                "Explainer not fitted! Please call .fit() before .clone_reduced()!"
            )
        reduced = self.__class__(
            self.X, self.y, self.classifier, dtype=self.dtype, **self.explainer_kwargs
        )
        reduced.loss_fn = self.loss_fn
        reduced.black_box_model = self.black_box_model
        reduced.local_models = local_models
        reduced.mapping = mapping
        reduced.vector_representation = vector_representation
        reduced.is_fit = True
        if proxy_ids is not None:
            reduced.rules = [self.rules[i] for i in proxy_ids]
        return reduced

    def _generate_vector_representation(self) -> torch.Tensor:
        return torch.zeros_like(self.X, dtype=self.dtype) - 1.0

    def lore_to_glocalx(self, rules: List[dict[str, Any]]):
        class_values = list(range(self.y.shape[1] + 1))
        feature_names = list(self.dataset.df.columns)
        output_rules = []
        for rule in rules:
            consequence = class_values.index(int(rule["consequence"]["val"]))
            premises = rule["premises"]
            features = [feature_names.index(premise["attr"]) for premise in premises]
            ops = [premise["op"] for premise in premises]
            values = [premise["val"] for premise in premises]
            values_per_feature = {
                feature: [val for f, val in zip(features, values) if f == feature]
                for feature in features
            }
            ops_per_feature = {
                feature: [op for f, op in zip(features, ops) if f == feature]
                for feature in features
            }

            output_premises = {}
            for f in features:
                values, operators = values_per_feature[f], ops_per_feature[f]
                # 1 value, either <= or >
                if len(values) == 1:
                    if operators[0] == "<=":
                        output_premises[f] = (-np.inf, values[0])
                    else:
                        output_premises[f] = (values[0], np.inf)
                # 2 values, < x <=
                else:
                    output_premises[f] = (min(values), max(values))

            transformed_rule = Rule(
                premises=output_premises, consequence=consequence, names=feature_names
            )
            output_rules.append(transformed_rule)

        return output_rules


# Get the explainer
def get_slise(X, y, pred_fn, epsilon: float, classifier, **slise_args):
    """Get SLISE prediction function.

    Args:
        X: data matrix.
        y: labels.
        pred_fn: black box function of the form yhat = pred_fn(X). NOT UTILIZED.
        classifier: Marks the task as classification. Defaults to False. If set to True,
        assumes binary classification where the first column of y corresponds to
        probability of class 0.

    Returns:
        A function to predict the label of new instances based on the SLISE explanations.
    """
    if classifier:
        y = y[:, 0]
    s = slise.SliseExplainer(X, y, epsilon**0.5, **slise_args)

    def explain(i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.explain(i)
        if classifier:
            return lambda X: np.stack((s.predict(X), 1.0 - s.predict(X)), -1)
        return s.predict

    return explain


def get_shap(X, y, model, partition=False, classifier=False):
    """Get SHAP prediction function.

    Args:
        X: data matrix.
        y: labels.
        pred_fn: black box function of the form yhat = pred_fn(X).
        partition: Use shap.PartitionExplainer. Defaults to False.
        classifier: Marks the task as classification. Defaults to False. If set to True,
        assumes binary classification where the first column of y corresponds to
        probability of class 0.

    Returns:
        A function to predict the label of new instances based on the SHAP explanations.
    """

    dtype = X.dtype
    pred_fn = model.predict_proba if classifier else model.predict

    def _shap_get_scale(X, y, shapX, b, intercept, link):
        # to make predictions, the SHAP kernel parameters matter. Hence, we optimise the
        # scale of the kernel for optimal loss (MAE).
        dist = torch.square(torch.as_tensor((X - shapX)))
        scale = torch.ones_like(dist[:1, :], requires_grad=True)
        Y = torch.as_tensor(link(utils.tonp(y)), dtype=dist.dtype)

        if len(Y.shape) == 1:
            Y = Y[:, None]

        b = torch.as_tensor(b, dtype=dist.dtype)

        def loss():
            kernel = torch.exp(-(torch.abs(scale) + 1e-6) * dist)
            P = torch.sum(kernel * b, -1, keepdim=True) + intercept
            return torch.mean(torch.abs(Y - P))

        optimiser = LBFGS([scale], line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            loss_val = loss()
            loss_val.backward()

            return loss_val

        loss_val = optimiser.step(closure)
        return utils.tonp(scale)

    def _shap_predict(X, shapX, scale, b, intercept, classifier):
        dist = utils.tonp(X - shapX) ** 2
        kernel = np.exp(-(np.abs(scale) + 1e-6) * dist)
        P = np.sum(kernel * b, -1, keepdims=True) + utils.tonp(intercept)
        if classifier:
            P = shap.links.logit.inverse(P)
            return torch.as_tensor(np.stack((P, 1.0 - P), -1)).squeeze()
        return torch.as_tensor(P, dtype=dtype)

    if classifier:
        link = shap.links.logit
        y = y[:, 0]
        old_pred = pred_fn
        pred_fn = lambda X: old_pred(X)[:, 0]
    else:
        link = shap.links.identity

    # the PartitionExplainer performs hierarchical clustering on input features and uses those
    # as "meta-features". It can work well with text/images but is recommended to be avoided
    # with tabular data
    if partition:
        # generate the clustering
        mask = shap.maskers.Partition(utils.tonp(X), max_samples=1_000)
        # explainer is now PartitionExplainer based on the above clustering
        explainer = shap.explainers.Partition(
            pred_fn, masker=mask, link=link, linearize_link=False
        )

        def explain(i):
            shapX = utils.tonp(X[None, i, :])
            exp = explainer(shapX, silent=True)
            b = exp.values.reshape((exp.values.shape[0], -1))
            inter = float(exp.base_values)
            scale = _shap_get_scale(X, y, shapX, b, inter, link)
            return b, lambda X: _shap_predict(X, shapX, scale, b, inter, classifier)

    else:
        explainer = shap.explainers.Sampling(pred_fn, utils.tonp(X))

        def explain(i):
            shapX = utils.tonp(X[None, i, :])
            b = explainer.shap_values(shapX, silent=True)
            inter = torch.as_tensor(explainer.expected_value)
            scale = _shap_get_scale(X, y, shapX, b, inter, link)
            return b, lambda X: _shap_predict(X, shapX, scale, b, inter, classifier)

    shap_models = [explain(i) for i in range(X.shape[0])]
    B = torch.vstack([torch.as_tensor(sm[0], dtype=dtype) for sm in shap_models])
    explanations = [sm[1] for sm in shap_models]
    return B, explanations


def get_lime(
    X,
    y,
    model,
    classifier=False,
    discretize=True,
    dtype=None,
    verbose=False,
    num_samples=5_000,
    **lime_kwargs,
):
    """Get LIME prediction function.

    Args:
        X: data matrix.
        y: labels.
        pred_fn: black box function of the form yhat = pred_fn(X).
        discretize: Use binary discretization on features. Defaults to True.
        classifier: Marks the task as classification. Defaults to False. If set to True,
        assumes binary classification where the first column of y corresponds to
        probability of class 0.

    Returns:
        A function to predict the label of new instances based on the LIME explanations.
    """
    if dtype is None:
        dtype = X.dtype
    X = utils.tonp(X)
    y = utils.tonp(y)
    explainer = LimeTabularExplainer(
        X,
        "classification" if classifier else "regression",
        y,
        discretize_continuous=discretize,
        **lime_kwargs,
    )
    pred_fn = model.predict_proba if classifier else model.predict

    def explain(i):

        def _lime_predict(X):
            X = utils.tonp(X)
            if discretize:
                X = explainer.discretizer.discretize(X) == x1
            Y = np.sum(X * b, -1, keepdims=True) + inter

            if classifier:
                Y = np.clip(Y, 0.0, 1.0)
                Y = np.concatenate((1.0 - Y, Y), -1)

            return torch.as_tensor(Y, dtype=dtype)

        exp = explainer.explain_instance(X[i, :], pred_fn, num_samples=num_samples)
        b = np.zeros((1, X.shape[1]))
        for j, v in exp.as_map()[1]:
            b[0, j] = v
        inter = exp.intercept[1]
        if discretize:
            x1 = explainer.discretizer.discretize(X[i : i + 1, :])

        return b, _lime_predict

    if verbose:
        lime_models = [explain(i) for i in tqdm(range(X.shape[0]))]
    else:
        lime_models = [explain(i) for i in range(X.shape[0])]
    B = torch.vstack([torch.as_tensor(lm[0], dtype=dtype) for lm in lime_models])
    explanations = [lm[1] for lm in lime_models]
    return B, explanations, explainer


def get_slisemap(X, y, classifier=False, **slisemap_params):
    """Get a SLISEMAP explainer."""
    lm = LogisticRegression if classifier else LinearRegression
    sm = Slisemap(X, y, local_model=lm, random_state=42, **slisemap_params)
    sm.optimise()
    B = sm.get_B(False)
    return lambda i: lambda X: utils.tonp(
        sm.local_model(sm._as_new_X(X), B[None, i])[0, ...]
    )


# Get the local models
def get_local_models_slisemap(X, model, mode):
    from slisemap.local_models import LogisticRegression, logistic_regression_loss

    if mode == "regression":
        sm = Slisemap(X, model.predict(X), lasso=0.01)
    else:
        ypred = model.predict_proba(X)
        sm = Slisemap(
            X,
            ypred,
            lasso=0.1,
            intercept=False,
            local_model=LogisticRegression,
            local_loss=logistic_regression_loss,
        )

    sm.optimise()
    B = torch.tensor(sm.get_B(), dtype=torch.float32)

    return B


def get_local_models_lime(X, model, mode, **lime_kwargs):
    B = torch.zeros((X.shape[0], X.shape[1] + 1))

    # Use LIME exclusively for regression tasks
    explainer = LimeTabularExplainer(
        training_data=X,
        mode=mode,
        discretize_continuous=False,
        sample_around_instance=True,
        **lime_kwargs,
    )
    for instance_id in range(X.shape[0]):
        e = explainer.explain_instance(
            data_row=X[instance_id], predict_fn=model.predict
        )
        for (
            feature_id,
            feature_grad,
        ) in e.as_map()[1]:
            B[instance_id, feature_id] = feature_grad
        B[instance_id, -1] = e.intercept[1]

    return B


def _compute_smoothgrad(
    X: np.ndarray,
    model,
    mode: str,
    perturbation: float,
    noise_level: float,
    sample_count: int,
    feature_id: int,
) -> np.ndarray:
    """Compute the averaged gradients for a single feature across all instances in regression mode."""
    """Compute the averaged gradients for a single feature across all instances."""
    n_samples, n_features = X.shape
    grads_list = []

    n_perturbation = 25

    for _ in range(n_perturbation):
        noise = np.random.normal(0, noise_level, (n_samples, sample_count, n_features))
        X_noisy = (
            X[:, np.newaxis, :] + noise
        )  # Shape: [n_samples x sample_count x n_features]

        # Reshape the noisy data to (n_samples * sample_count) x n_features
        X_noisy_reshaped = X_noisy.reshape(-1, n_features)

        # Perturb the specified feature
        X_noisy_perturbed_pos = X_noisy_reshaped.copy()
        X_noisy_perturbed_neg = X_noisy_reshaped.copy()
        X_noisy_perturbed_pos[:, feature_id] += perturbation
        X_noisy_perturbed_neg[:, feature_id] -= perturbation

        if mode == "regression":
            # Compute the gradients
            preds_perturbed_pos = model.predict(X_noisy_perturbed_pos)
            preds_perturbed_neg = model.predict(X_noisy_perturbed_neg)
            grads = (preds_perturbed_pos - preds_perturbed_neg) / (2 * perturbation)
        elif mode == "classification":
            # Compute the gradients
            prob_perturbed_pos = model.predict_proba(X_noisy_perturbed_pos)[:, 1]
            prob_perturbed_neg = model.predict_proba(X_noisy_perturbed_neg)[:, 1]
            log_odds_perturbed_pos = np.log(
                prob_perturbed_pos / (1 - prob_perturbed_pos)
            )
            log_odds_perturbed_neg = np.log(
                prob_perturbed_neg / (1 - prob_perturbed_neg)
            )
            grads = (log_odds_perturbed_pos - log_odds_perturbed_neg) / (
                2 * perturbation
            )
        else:
            raise ValueError("Invalid mode. Choose 'regression' or 'classification'.")

        grads = grads.reshape(n_samples, sample_count)
        grads_list.append(grads)

    # Average over the repeats and sample_count
    grads_mean = np.mean(np.stack(grads_list), axis=(0, 2))  # Shape: [n_samples]

    return grads_mean


def get_local_models_smoothgrad(
    X: np.ndarray,
    model,
    mode: str,
    dtype,
    noise_level: float = 0.1,
    num_samples: int = 25,
    perturbation: float = 1e-4,
) -> torch.Tensor:
    """Get the local models using SmoothGrad: add Gaussian noise and average gradients.

    Args:
        X: input data.
        model: the model to explain.
        mode: 'regression' or 'classification'.

    Returns:
        A [n x (d + 1)] matrix of local models, where each row corresponds to the local model
    """

    n_samples, n_features = X.shape

    if mode == "regression":
        B = torch.zeros((n_samples, n_features + 1), dtype=dtype)
        preds_original = model.predict(X).flatten()
    elif mode == "classification":
        B = torch.zeros((n_samples, n_features), dtype=dtype)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compute the gradients for each feature
    for k in range(n_features):
        avg_grads = _compute_smoothgrad(
            X=X,
            model=model,
            mode=mode,
            perturbation=perturbation,
            noise_level=noise_level,
            sample_count=num_samples,
            feature_id=k,
        )
        B[:, k] = torch.from_numpy(avg_grads)

    if mode == "regression":
        # Compute the intercepts
        grads_np = B[:, :-1].numpy()
        intercepts = preds_original - np.sum(grads_np * X, axis=1)
        B[:, -1] = torch.from_numpy(intercepts)

    return B


def get_local_models_kernelshap(X, model, mode, dtype, **shap_kwargs):
    # Use 25 samples for the background dataset
    background = shap.sample(X, 50)
    if mode == "regression":
        # Initialize the explainer with background data
        explainer = shap.KernelExplainer(
            model.predict, background, n_jobs=-1, **shap_kwargs
        )
        B = torch.zeros((X.shape[0], X.shape[1] + 1), dtype=dtype)
        shap_values = explainer.shap_values(X, nsamples=25, n_jobs=-1)
        shap_values = np.clip(shap_values, -1e6, 1e6)
        gradients = shap_values.reshape(X.shape[0], X.shape[1])
        preds = model.predict(X).flatten()
        dot_products = np.sum(gradients * X, axis=1)
        intercepts = preds - dot_products
        B[:, :-1] = torch.from_numpy(gradients)
        B[:, -1] = torch.from_numpy(intercepts)

    else:

        def f(x):
            return model.predict_proba(x)[:, 0]

        explainer = shap.Explainer(f, background, link=shap.links.logit)
        shap_values = explainer(X)
        B = torch.from_numpy(shap_values.values)

    return B


def get_local_models(X: pd.DataFrame, model: Any, method: str, mode: str):
    if method == "lime":
        B = get_local_models_lime(X.to_numpy(), model, mode)
    elif method == "slisemap":
        B = get_local_models_slisemap(X, model, mode)
    elif method == "smoothgrad":
        B = get_local_models_smoothgrad(X.to_numpy(), model, mode)
    elif method == "kernelshap":
        B = get_local_models_kernelshap(X.to_numpy(), model, mode)
    else:
        raise ValueError(f"Unknown method: {method}")
    return B


# Loss function
def logistic_regression(X: torch.Tensor, B: torch.Tensor, i: int):
    """Prediction function for logistic regression using the ith row in a coefficient
    matrix B."""
    n_x, m = X.shape
    _, o = B.shape
    p = 1 + torch.div(o, m, rounding_mode="trunc")
    a = torch.zeros([n_x, p], dtype=B.dtype)
    for j in range(p - 1):
        a[:, j] = B[i, (j * m) : ((j + 1) * m)] @ X.T
    return torch.nn.functional.softmax(a, dim=1)


def logistic_regression_loss(Y, Yhat):
    """Calculate the Hellinger distance between two (multidimensional) probability distributions."""
    return ((Yhat.sqrt() - Y.sqrt().expand(Yhat.shape)) ** 2).sum(dim=-1) * 0.5


def get_single_global_model(
    X, Y, classifier, loss_fn, lasso=0.0, ridge=0.0, intercept=False
):
    n_samples, n_features = X.shape
    if intercept:
        X = torch.cat([X, torch.ones(n_samples, 1)], dim=1)
        n_features += 1
    B = torch.zeros(n_features, dtype=X.dtype, device=X.device, requires_grad=True)

    def loss() -> torch.Tensor:
        yhat = (X @ B)[:, None]
        if classifier:
            yhat = torch.sigmoid(yhat)
            yhat = torch.hstack((yhat, 1.0 - yhat))
        loss_val = loss_fn(Y, yhat).mean()
        if lasso > 0:
            loss_val += lasso * torch.sum(B.abs())
        if ridge > 0:
            loss_val += ridge * torch.sum(torch.square(B))
        return loss_val

    optimiser = LBFGS([B], line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss_val = loss()
        loss_val.backward()

        return loss_val

    loss_val = optimiser.step(closure)
    return B.detach()

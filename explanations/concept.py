import abc
import logging
import numpy as np
import torch
import torch.nn.functional as F
import optuna
from abc import ABC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import permutation_test_score, train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class ConceptExplainer(ABC):
    """
    An abstract class that contains the interface for any post-hoc concept explainer
    """

    @abc.abstractmethod
    def __init__(self, device: torch.device, batch_size: int = 50):
        self.concept_reps = None
        self.concept_labels = None
        self.classifier = None
        self.device = device
        self.batch_size = batch_size

    @abc.abstractmethod
    def fit(self, concept_reps: np.ndarray, concept_labels: np.ndarray) -> None:
        """
        Fit the concept classifier to the dataset (latent_reps, concept_labels)
        Args:
            concept_reps: latent representations of the examples illustrating the concept
            concept_labels: labels indicating the presence (1) or absence (0) of the concept
        """
        assert concept_reps.shape[0] == concept_labels.shape[0]
        self.concept_reps = concept_reps
        self.concept_labels = concept_labels

    @abc.abstractmethod
    def predict(self, latent_reps: np.ndarray) -> np.ndarray:
        """
        Predicts the presence or absence of the concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts labels indicating the presence (1) or absence (0) of the concept
        """

    @abc.abstractmethod
    def concept_importance(self, latent_reps):
        """
        Predicts the relevance of a concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts scores for each example
        """

    @abc.abstractmethod
    def permutation_test(
        self,
        concept_reps: np.ndarray,
        concept_labels: np.ndarray,
        n_perm: int = 100,
        n_jobs: int = -1,
    ) -> float:
        """
        Computes the p-value of the concept-label permutation test
        Args:
            concept_labels: concept labels indicating the presence (1) or absence (0) of the concept
            concept_reps: representation of the examples
            n_perm: number of permutations
            n_jobs: number of jobs running in parallel

        Returns:
            p-value of the statistical significance test
        """

    def get_concept_reps(self, positive_set: bool) -> np.ndarray:
        """
        Get the latent representation of the positive/negative examples
        Args:
            positive_set: True returns positive examples, False returns negative examples
        Returns:
            Latent representations of the requested set
        """
        return self.concept_reps[self.concept_labels == int(positive_set)]


class CAR(ConceptExplainer, ABC):
    def __init__(
        self,
        device: torch.device,
        batch_size: int = 100,
        kernel: str = "rbf",
        kernel_width: float = None,
    ):
        super(CAR, self).__init__(device, batch_size)
        self.kernel = kernel
        self.kernel_width = kernel_width

    def fit(self, concept_reps: np.ndarray, concept_labels: np.ndarray) -> None:
        """
        Fit the concept classifier to the dataset (latent_reps, concept_labels)
        Args:
            concept_reps: latent representations of the examples illustrating the concept
            concept_labels: labels indicating the presence (1) or absence (0) of the concept
        """
        super(CAR, self).fit(concept_reps, concept_labels)
        classifier = SVC(kernel=self.kernel)
        classifier.fit(concept_reps, concept_labels)
        self.classifier = classifier

    def predict(self, latent_reps: np.ndarray) -> np.ndarray:
        """
        Predicts the presence or absence of the concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts labels indicating the presence (1) or absence (0) of the concept
        """
        return self.classifier.predict(latent_reps)

    def concept_importance(self, latent_reps: torch.Tensor) -> torch.Tensor:
        """
        Predicts the relevance of a concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts scores for each example
        """
        pos_density = self.concept_density(latent_reps, True)
        neg_density = self.concept_density(latent_reps, False)
        return pos_density - neg_density

    def permutation_test(
        self,
        concept_reps: np.ndarray,
        concept_labels: np.ndarray,
        n_perm: int = 100,
        n_jobs: int = -1,
    ) -> float:
        """
        Computes the p-value of the concept-label permutation test
        Args:
            concept_labels: concept labels indicating the presence (1) or absence (0) of the concept
            concept_reps: representation of the examples
            n_perm: number of permutations
            n_jobs: number of jobs running in parallel

        Returns:
            p-value of the statistical significance test
        """
        classifier = SVC(kernel=self.kernel)
        score, permutation_scores, p_value = permutation_test_score(
            classifier,
            concept_reps,
            concept_labels,
            n_permutations=n_perm,
            n_jobs=n_jobs,
        )
        return p_value

    def get_kernel_function(self) -> callable:
        """
        Get the kernel funtion underlying the CAR
        Returns: kernel function as a callable with arguments (h1, h2)
        """
        if self.kernel == "rbf":
            if self.kernel_width is not None:
                kernel_width = self.kernel_width
            else:
                kernel_width = 1.0
            latent_dim = self.concept_reps.shape[-1]
            # We unstack the tensors to return a kernel matrix of shape len(h1) x len(h2)!
            return lambda h1, h2: torch.exp(
                -torch.sum(
                    ((h1.unsqueeze(1) - h2.unsqueeze(0)) / (latent_dim * kernel_width))
                    ** 2,
                    dim=-1,
                )
            )
        elif self.kernel == "linear":
            return lambda h1, h2: torch.einsum(
                "abi, abi -> ab", h1.unsqueeze(1), h2.unsqueeze(0)
            )

    def concept_density(
        self, latent_reps: torch.Tensor, positive_set: bool
    ) -> torch.Tensor:
        """
        Computes the concept density for the given latent representations
        Args:
            latent_reps: latent representations for which the concept density should be evaluated
            positive_set: if True, only compute the density for the positive set. If False, only for the negative.


        Returns:
            The density of the latent representations under the relevant concept set
        """
        kernel = self.get_kernel_function()
        if type(latent_reps) is not np.ndarray:
            latent_reps = latent_reps.to(self.device)
            concept_reps = torch.from_numpy(self.get_concept_reps(positive_set)).to(
                self.device
            )
        else:
            latent_reps = torch.from_numpy(latent_reps).to(self.device)
            concept_reps = torch.from_numpy(self.get_concept_reps(positive_set)).to(
                self.device
            )
        density = kernel(concept_reps, latent_reps).mean(dim=0)
        return density

    def tune_kernel_width(self, concept_reps: np.ndarray, concept_labels: np.ndarray):
        """
        Args:
            concept_reps: training representations
            concept_labels: training labels
        Tune the kernel width to achieve good training classification accuracy with a Parzen classifier
        Returns:

        """
        super(CAR, self).fit(concept_reps, concept_labels)

        def train_acc(trial):
            kernel_width = trial.suggest_float("kernel_width", 0.1, 50)
            self.kernel_width = kernel_width
            density = []
            for reps_batch in np.split(concept_reps, self.batch_size):
                density.append(
                    self.concept_importance(torch.from_numpy(reps_batch)).cpu().numpy()
                )
            density = np.concatenate(density)
            return accuracy_score((density > 0).astype(int), concept_labels)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        # 添加进度条回调
        pbar = tqdm(total=1000, desc="Tuning kernel width", unit="trial")
        def callback(study, trial):
            pbar.update(1)
        study.optimize(train_acc, n_trials=1000, callbacks=[callback])
        pbar.close()

        self.kernel_width = study.best_params["kernel_width"]
        logging.info(
            f"Optimal kernel width {self.kernel_width:.3g} with training accuracy {study.best_value:.2g}"
        )

    def fit_cv(self, concept_reps: np.ndarray, concept_labels: np.ndarray) -> None:
        """
        Fit the concept classifier to the dataset (latent_reps, concept_labels) by tuning the kernel width
        Args:
            concept_reps: latent representations of the examples illustrating the concept
            concept_labels: labels indicating the presence (1) or absence (0) of the concept
        """
        super(CAR, self).fit(concept_reps, concept_labels)

        X_train, X_val, y_train, y_val = train_test_split(
            concept_reps,
            concept_labels,
            test_size=int(0.3 * len(concept_reps)),
            stratify=concept_labels,
        )

        def objective(trial: optuna.Trial) -> float:
            kernel = trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            )
            gamma = trial.suggest_loguniform("gamma", 1e-3, 1e3)
            C = trial.suggest_loguniform("C", 1e-3, 1e3)
            classifier = SVC(kernel=kernel, gamma=gamma, C=C)
            classifier.fit(X_train, y_train)
            return accuracy_score(y_val, classifier.predict(X_val))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=200, show_progress_bar=True)
        best_params = study.best_params
        self.classifier = SVC(**best_params)
        self.classifier.fit(concept_reps, concept_labels)
        self.kernel_width = best_params["gamma"]
        logging.info(
            f"Optimal hyperparameters {best_params} with validation accuracy {study.best_value:.2g}"
        )

    def concept_sensitivity_importance(
        self,
        latent_reps: np.ndarray,
        labels: torch.Tensor = None,
        num_classes: int = None,
        rep_to_output: callable = None,
    ) -> np.ndarray:
        """
        Compute the concept sensitivity of a set of predictions
        Args:
            latent_reps: representations of the test examples
            labels: the labels associated to the representations one-hot encoded
            num_classes: the number of classes
            rep_to_output: black-box mapping the representation space to the output space
        Returns:
            concepts scores for each example
        """
        one_hot_labels = F.one_hot(labels, num_classes).to(self.device)
        latent_reps = torch.from_numpy(latent_reps).to(self.device).requires_grad_()
        outputs = rep_to_output(latent_reps)
        grads = torch.autograd.grad(outputs, latent_reps, grad_outputs=one_hot_labels)[
            0
        ]

        densities = self.concept_importance(latent_reps).view((-1, 1))
        cavs = torch.autograd.grad(
            densities,
            latent_reps,
            grad_outputs=torch.ones((len(densities), 1)).to(self.device),
        )[0]

        if len(grads.shape) > 2:
            grads = grads.flatten(start_dim=1)
        if len(cavs.shape) > 2:
            cavs = cavs.flatten(start_dim=1)
        return torch.einsum("bi,bi->b", cavs, grads).detach().cpu().numpy()


class CAV(ConceptExplainer, ABC):
    def __init__(self, device: torch.device, batch_size: int = 50):
        super(CAV, self).__init__(device, batch_size)

    def fit(self, concept_reps: np.ndarray, concept_labels: np.ndarray) -> None:
        """
        Fit the concept classifier to the dataset (latent_reps, concept_labels)
        Args:
            kernel: kernel function
            latent_reps: latent representations of the examples illustrating the concept
            concept_labels: labels indicating the presence (1) or absence (0) of the concept
        """
        super(CAV, self).fit(concept_reps, concept_labels)
        classifier = SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3)
        classifier.fit(concept_reps, concept_labels)
        self.classifier = classifier

    def predict(self, latent_reps: np.ndarray) -> np.ndarray:
        """
        Predicts the presence or absence of the concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts labels indicating the presence (1) or absence (0) of the concept
        """
        return self.classifier.predict(latent_reps)

    def concept_importance(
        self,
        X_test,
        latent_reps: np.ndarray,
        labels: torch.Tensor = None,
        num_classes: int = None,
        rep_to_output: callable = None,
    ) -> np.ndarray:
        """
        Predicts the relevance of a concept for the latent representations
        Args:
            latent_reps: representations of the test examples
            labels: the labels associated to the representations one-hot encoded
            num_classes: the number of classes
            rep_to_output: black-box mapping the representation space to the output space
        Returns:
            concepts scores for each example
        """
        one_hot_labels = F.one_hot(labels, num_classes).to(self.device)
        if isinstance(latent_reps, np.ndarray):
            latent_reps = torch.from_numpy(latent_reps).to(self.device).requires_grad_()
        outputs = rep_to_output(latent_reps, X_test)
        grads = torch.autograd.grad(outputs, latent_reps, grad_outputs=one_hot_labels)[
            0
        ]
        cav = self.get_activation_vector()
        
        if len(grads.shape) > 2:
            grads = grads.flatten(start_dim=1)
        if len(cav.shape) > 2:
            cav = cav.flatten(start_dim=1)
        return torch.einsum("bi,bi->b", cav, grads).detach().cpu().numpy()

    def concept_importance_observable(
        self,
        observable_reps: np.ndarray | torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        obs_to_output: callable,
    ) -> np.ndarray:
        """
        Strict TCAV score in observable space:
            d f_k(z) / d z dot v_c

        Args:
            observable_reps: observable features z of shape [batch, n_features]
            labels: class labels associated with each sample
            num_classes: number of classes
            obs_to_output: mapping z -> logits
        Returns:
            per-example TCAV scores
        """
        if isinstance(observable_reps, np.ndarray):
            observable_reps = torch.from_numpy(observable_reps).to(self.device)
        observable_reps = observable_reps.to(self.device).float().requires_grad_()

        outputs = obs_to_output(observable_reps)
        one_hot_labels = F.one_hot(labels.to(self.device), num_classes).to(outputs.dtype)
        grads = torch.autograd.grad(
            outputs, observable_reps, grad_outputs=one_hot_labels
        )[0]

        cav = self.get_activation_vector().to(grads.dtype)
        if len(grads.shape) > 2:
            grads = grads.flatten(start_dim=1)
        if len(cav.shape) > 2:
            cav = cav.flatten(start_dim=1)
        if cav.shape[0] == 1 and grads.shape[0] > 1:
            cav = cav.expand(grads.shape[0], -1)

        return torch.einsum("bi,bi->b", cav, grads).detach().cpu().numpy()

    def concept_importance_chainrule(
        self,
        X_test: torch.Tensor,
        latent_reps: np.ndarray | torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        rep_to_output: callable,
        obs_fn: callable,
        reg: float = 1e-4,
    ) -> np.ndarray:
        """
        Strict TCAV via chain rule without a surrogate head.

        Computes gradients with respect to observable features z = obs_fn(h)
        while keeping the original model output path f(h, x):
            grad_h f = J(h)^T grad_z f,
        where J(h) = d z / d h. We solve for grad_z f with a
        Tikhonov-regularized least squares inverse.
        """
        if isinstance(latent_reps, np.ndarray):
            latent_reps = torch.from_numpy(latent_reps).to(self.device)
        latent_reps = latent_reps.to(self.device).requires_grad_()
        X_test = X_test.to(self.device)
        labels = labels.to(self.device)

        outputs = rep_to_output(latent_reps, X_test)
        if outputs.shape[-1] != num_classes:
            raise ValueError(
                f"Unexpected output dim {outputs.shape[-1]} (expected {num_classes})"
            )

        target_scores = outputs.gather(1, labels.view(-1, 1)).squeeze(1)
        grads_h = torch.autograd.grad(target_scores.sum(), latent_reps, retain_graph=True)[0]

        cav = self.get_activation_vector().to(self.device).float().reshape(-1)
        scores = []
        eye_cache = {}

        for idx in range(latent_reps.shape[0]):
            h_i = latent_reps[idx : idx + 1]
            z_i = obs_fn(h_i)
            m = z_i.shape[-1]

            jac_rows = []
            for j in range(m):
                grad_zj = torch.autograd.grad(
                    z_i[0, j], h_i, retain_graph=True
                )[0].reshape(-1)
                jac_rows.append(grad_zj)
            jac = torch.stack(jac_rows, dim=0)  # [m, n_h] complex

            grad_hi = grads_h[idx].reshape(-1)
            b_mat = torch.cat([jac.transpose(0, 1).real, jac.transpose(0, 1).imag], dim=0).float()
            rhs = torch.cat([grad_hi.real, grad_hi.imag], dim=0).float()

            if m not in eye_cache:
                eye_cache[m] = torch.eye(m, device=self.device, dtype=b_mat.dtype)
            lhs = b_mat.transpose(0, 1) @ b_mat + reg * eye_cache[m]
            grad_z = torch.linalg.solve(lhs, b_mat.transpose(0, 1) @ rhs)

            scores.append(torch.dot(cav.to(grad_z.dtype), grad_z))

        return torch.stack(scores).detach().cpu().numpy()

    def permutation_test(
        self,
        concept_reps: np.ndarray,
        concept_labels: np.ndarray,
        n_perm: int = 100,
        n_jobs: int = -1,
    ) -> float:
        """
        Computes the p-value of the concept-label permutation test
        Args:
            concept_labels: concept labels indicating the presence (1) or absence (0) of the concept
            concept_reps: representation of the examples
            n_perm: number of permutations
            n_jobs: number of jobs running in parallel

        Returns:
            p-value of the statistical significance test
        """
        classifier = SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3)
        score, permutation_scores, p_value = permutation_test_score(
            classifier,
            concept_reps,
            concept_labels,
            n_permutations=n_perm,
            n_jobs=n_jobs,
        )
        return p_value

    def get_activation_vector(self):
        return torch.tensor(self.classifier.coef_).to(self.device).float()

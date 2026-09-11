import numpy as np
import torch
from tqdm import tqdm
from captum import attr
from explanations.concept import ConceptExplainer
from torch.utils.data import DataLoader
def dm2vec(rho):
    """
    把 单个密度矩阵（复数）→ 实数特征向量
    输入：rho (DensityMatrix 或 np.array)
    输出：实数一维向量
    """
    # 转成 numpy 复数矩阵
    if hasattr(rho, 'data'):
        rho = rho.data

    # 实部 + 虚部 拼接（SVM只认实数）
    real_part = np.real(rho).flatten()
    imag_part = np.imag(rho).flatten()

    # 合并成一个实数向量
    feature = np.concatenate([real_part, imag_part])

    # 去掉极小值（避免数值噪声）
    feature = np.nan_to_num(feature, nan=0, posinf=0, neginf=0)
    return feature

class CARFeatureImportance:
    def __init__(
        self,
        attribution_name: str,
        concept_explainer: ConceptExplainer,
        black_box: torch.nn.Module,
        device: torch.device,
    ):
        assert attribution_name in {"Gradient Shap", "Integrated Gradient"}
        if attribution_name == "Gradient Shap":
            self.attribution_method = attr.GradientShap(self.concept_importance)
        elif attribution_name == "Integrated Gradient":
            self.attribution_method = attr.IntegratedGradients(self.concept_importance)
        self.concept_explainer = concept_explainer
        self.black_box = black_box.to(device)
        self.device = device

    def attribute(self, data_loader: DataLoader, **kwargs) -> np.ndarray:
        input_shape = list(data_loader.dataset[0][0].shape)
        attr = np.empty(shape=[0] + input_shape)
        baselines = kwargs["baselines"]
        for input_features, _ in tqdm(data_loader, unit="batch", leave=False):
            input_features = input_features.to(self.device)
            if isinstance(baselines, torch.Tensor):
                attr = np.append(
                    attr,
                    self.attribution_method.attribute(input_features, **kwargs)
                    .detach()
                    .cpu()
                    .numpy(),
                    axis=0,
                )
            elif isinstance(baselines, torch.nn.Module):
                internal_batch_size = kwargs["internal_batch_size"]
                attr = np.append(
                    attr,
                    self.attribution_method.attribute(
                        input_features,
                        baselines=baselines(input_features),
                        internal_batch_size=internal_batch_size,
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                    axis=0,
                )
            else:
                raise ValueError("Invalid baseline type")
        return attr

    def concept_importance(self, input_features: torch.tensor) -> torch.Tensor:
        input_features = input_features.to(self.device)
        latent_reps = self.black_box.input_to_representation(input_features)
        # The black box returns (complex) density matrices of shape
        # (bsz, dim, dim). The CAR classifier was fitted on the real feature
        # vectors produced by `dm2vec` (real part flattened, then imaginary
        # part flattened and concatenated). We reproduce the exact same
        # transformation here, but using differentiable torch operations so
        # that Integrated Gradients can still back-propagate through it.
        bsz = latent_reps.shape[0]
        if torch.is_complex(latent_reps):
            real_part = latent_reps.real.reshape(bsz, -1)
            imag_part = latent_reps.imag.reshape(bsz, -1)
            latent_reps = torch.cat([real_part, imag_part], dim=1)
        else:
            latent_reps = latent_reps.reshape(bsz, -1)
        return self.concept_explainer.concept_importance(latent_reps)


class VanillaFeatureImportance:
    def __init__(
        self, attribution_name: str, black_box: torch.nn.Module, device: torch.device
    ):
        assert attribution_name in {"Gradient Shap", "Integrated Gradient"}
        if attribution_name == "Gradient Shap":
            self.attribution_method = attr.GradientShap(black_box)
        elif attribution_name == "Integrated Gradient":
            self.attribution_method = attr.IntegratedGradients(black_box)
        self.black_box = black_box.to(device)
        self.device = device

    def attribute(self, data_loader: DataLoader, **kwargs) -> np.ndarray:
        input_shape = list(data_loader.dataset[0][0].shape)
        attr = np.empty(shape=[0] + input_shape)
        baselines = kwargs["baselines"]
        for input_features, targets in tqdm(data_loader, unit="batch", leave=False):
            targets = targets.to(self.device)
            input_features = input_features.to(self.device)
            if isinstance(baselines, torch.Tensor):
                attr = np.append(
                    attr,
                    self.attribution_method.attribute(
                        input_features, target=targets, **kwargs
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                    axis=0,
                )
            elif isinstance(baselines, torch.nn.Module):
                internal_batch_size = kwargs["internal_batch_size"]
                attr = np.append(
                    attr,
                    self.attribution_method.attribute(
                        input_features,
                        target=targets,
                        baselines=baselines(input_features),
                        internal_batch_size=internal_batch_size,
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                    axis=0,
                )
        return attr

import yaml
from dataclasses import dataclass, fields, field
from typing import Any, List, Union


@dataclass
class ExperimentConfig:
    epoch: int
    batch_size: int
    learing_rate: float
    updater: str
    l1_weight: float
    l2_weight: float


@dataclass
class ModelConfig:
    network: str
    latent_dim: int = None
    initialization: str = None


@dataclass
class GANConfig:
    discriminator: str


@dataclass
class DatasetConfig:
    size: str
    unique: bool
    blacklist: bool
    adni: List[str] = field(default_factory=list)
    ppmi: List[str] = field(default_factory=list)
    fourrtni: List[str] = field(default_factory=list)
    hd: List[str] = field(default_factory=list)
    pids: List[str] = field(default_factory=list)
    uids: List[int] = field(default_factory=list)


@dataclass
class SparseConifg:
    ro: float
    beta: float
    fcwd: float


@dataclass
class VAEConfig:
    mse_weight: float
    ssim_weight: float
    kld_weight: float
    localize_weight: float
    localize_amplify: float
    localize_th: float
    quantize: int


@dataclass
class MobileNetConfig:
    in_ch: int
    block_setting: List[List[int]] = field(default_factory=list)


@dataclass
class ResumeConfig:
    config_name: str


@dataclass
class ExternalConfig:
    sparse: SparseConifg
    vae: VAEConfig
    mobilenet: MobileNetConfig
    resume: ResumeConfig


@dataclass
class Config:
    name: str
    experiment: ExperimentConfig
    model: ModelConfig
    dataset: DatasetConfig = None
    gan: GANConfig = None
    external_config: ExternalConfig = None

    def __name__(self):
        return self.name


def load_from_dict(class_, dict_) -> Union[Config, Any]:
    try:
        fieldtypes = {f.name: f.type for f in fields(class_)}
        return class_(**{f: load_from_dict(fieldtypes[f], dict_[f]) for f in dict_})
    except TypeError:
        if dict_ == 'None':
            dict_ = None
        return dict_


def create_config(config_path: str) -> Config:
    with open(config_path, "r") as rf:
        config_data = yaml.safe_load(rf)
    config: Config = load_from_dict(Config, config_data)
    return config

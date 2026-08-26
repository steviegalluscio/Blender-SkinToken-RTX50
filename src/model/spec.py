from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from numpy import ndarray
from typing import Dict, List, Optional, final
from torch import Tensor, nn

import numpy as np
import torch

from ..data.transform import Transform 
from ..rig_package.info.asset import Asset
from ..tokenizer.spec import DetokenizeOutput

@dataclass
class ModelInput():
    asset: Asset
    tokens: Optional[ndarray]=None

def _to_plain_dict(cfg):
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return {k: _to_plain_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_to_plain_dict(x) for x in cfg]
    if hasattr(cfg, "to_container"):
        return cfg.to_container(resolve=True)
    return cfg

_CURRENT_CHECKPOINT_DIR: Optional[Path] = None

def resolve_model_path(path_str: str) -> str:
    if not path_str:
        return path_str
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return str(p)
    
    candidates = []
    global _CURRENT_CHECKPOINT_DIR
    if _CURRENT_CHECKPOINT_DIR is not None:
        candidates.append(_CURRENT_CHECKPOINT_DIR / p)
        candidates.append(_CURRENT_CHECKPOINT_DIR.parent / p)
        candidates.append(_CURRENT_CHECKPOINT_DIR.parent.parent / p)
    
    candidates.append(Path.cwd() / p)
    candidates.append(Path.home() / ".cache" / "skintokens" / "checkpoints" / p)
    candidates.append(Path.home() / ".cache" / "skintokens" / p)

    for c in candidates:
        if c.exists():
            return str(c)
            
    return str(candidates[0]) if candidates else str(p)

class ModelSpec(nn.Module, ABC):
    
    model_config: Dict
    transform_config: Dict
    tokenizer_config: Dict|None
    
    @abstractmethod
    def __init__(self, model_config, transform_config, tokenizer_config=None):
        super().__init__()
        model_cfg = _to_plain_dict(model_config)
        transform_cfg = _to_plain_dict(transform_config)
        tokenizer_cfg = _to_plain_dict(tokenizer_config)
        self.model_config = model_cfg # type: ignore
        self.transform_config = transform_cfg # type: ignore
        self.tokenizer_config = tokenizer_cfg # type: ignore
        self.hparams = {
            "model_config": model_cfg,
            "transform_config": transform_cfg,
            "tokenizer_config": tokenizer_cfg,
        }
    
    def save_hyperparameters(self, *args, **kwargs):
        pass

    @property
    def _trainer(self):
        return None
    
    @final
    def _process_fn(self, batch: List[ModelInput]) -> List[Dict]:
        n_batch = self.process_fn(batch)
        if self._trainer is None or not self.trainer.training:
            for k in n_batch[0].keys():
                if not isinstance(n_batch[0][k], ndarray) and not isinstance(n_batch[0][k], Tensor):
                    continue
                s = n_batch[0][k].shape
                for i in range(1, len(n_batch)):
                    assert n_batch[i][k].shape == s, f"{k} has different shape in batch"
            for (i, b) in enumerate(batch):
                non = n_batch[i].get('non', {})
                non['model_input'] = deepcopy(b)
                n_batch[i]['non'] = non
        else:
            for b in batch:
                del b.asset
        return n_batch
    
    @abstractmethod
    def process_fn(self, batch: List[ModelInput]) -> List[Dict]:
        """
        Fetch data from dataloader and turn it into Tensor objects.
        """
        raise NotImplementedError()
    
    def compile_model(self):
        """
        Compile the model. Do this before training and after loading state dicts.
        """
        pass
    
    def on_load_checkpoint(self, checkpoint):
        pass

    @classmethod
    def load_from_system_checkpoint(cls, checkpoint_path: str, strict: bool=True, **kwargs):
        resolved_path = resolve_model_path(checkpoint_path)
        global _CURRENT_CHECKPOINT_DIR
        _CURRENT_CHECKPOINT_DIR = Path(resolved_path).resolve().parent
        
        ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
        state_dict = ckpt['state_dict']
        model_config = kwargs.get('model_config', None)
        transform_config = kwargs.get('transform_config', None)
        tokenizer_config = kwargs.get('tokenizer_config', None)
        if model_config is None:
            model_config = ckpt['hyper_parameters']['model_config']
        if transform_config is None:
            transform_config = ckpt['hyper_parameters']['transform_config']
        if tokenizer_config is None:
            tokenizer_config = ckpt['hyper_parameters']['tokenizer_config']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("_orig_mod.", "")
            if k.startswith("model."):
                k = k[len("model.") :]
            new_state_dict[k] = v
        model = cls(
            model_config=model_config,
            transform_config=transform_config,
            tokenizer_config=tokenizer_config,
        )
        missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
        if missing or unexpected:
            print(f"[Warning] Missing keys: {missing}")
            print(f"[Warning] Unexpected keys: {unexpected}")
        if hasattr(model, "on_load_checkpoint"):
            model.on_load_checkpoint(ckpt)
        return model
    
    def get_train_transform(self) -> Transform|None:
        cfg = self.transform_config.get('train_transform', None)
        if cfg is None:
            return None
        return Transform.parse(**cfg)
    
    def get_validate_transform(self) -> Transform|None:
        cfg = self.transform_config.get('validate_transform', None)
        if cfg is None:
            return None
        return Transform.parse(**cfg)
    
    def get_predict_transform(self) -> Transform|None:
        cfg = self.transform_config.get('predict_transform', None)
        if cfg is None:
            return None
        return Transform.parse(**cfg)
    
    def predict_step(self, batch: Dict, no_cls: bool=False, skeleton_tokens=None) -> Dict:
        raise NotImplementedError()


@dataclass
class VaeInput():
    dense_cond: List[Tensor] # [(J, skin_samples, 6)]
    dense_skin: List[Tensor] # [(J, skin_samples)]
    dense_indices: List[List[int]] # [List[J]], corresponding indices of gt
    uniform_cond: Tensor # (B, N, 6)
    uniform_skin: List[Tensor] # [(N, J)]
    
    @property
    def B(self):
        return self.uniform_cond.shape[0]
    
    @property
    def max_J(self):
        return max([len(s) for s in self.dense_indices])
    
    def get_len(self, i) -> int:
        return len(self.dense_indices[i])
    
    def _clamp_j(self, i: int, j: int) -> int:
        return min(j, len(self.dense_indices[i])-1)
    
    def get_dense_cond(self, j: int) -> Tensor:
        """return (B, skin_samples, 6)"""
        return torch.stack([self.dense_cond[i][self._clamp_j(i=i, j=j)] for i in range(self.B)])
    
    def get_dense_skin(self, j: int) -> Tensor:
        """return (B, skin_samples)"""
        return torch.stack([self.dense_skin[i][self._clamp_j(i=i, j=j)] for i in range(self.B)])
    
    def get_full_cond(self, j: int) -> Tensor:
        """return (B, N+skin_samples, 6)"""
        return torch.cat([self.uniform_cond, self.get_dense_cond(j=j)], dim=1)
    
    def get_uniform_skin(self, j: int) -> Tensor:
        """return (B, N)"""
        return torch.stack([self.uniform_skin[i][:, self._clamp_j(i=i, j=j)] for i in range(self.B)])
    
    def get_full_skin(self, j: int) -> Tensor:
        """return (B, N+skin_samples)"""
        return torch.cat([self.get_uniform_skin(j=j), self.get_dense_skin(j=j)], dim=1)
    
    def get_flatten_uniform_cond(self) -> Tensor:
        """return (sum_J, N, 6)"""
        return self.uniform_cond[self.get_flatten_indices()]
    
    def get_flatten_dense_cond(self) -> Tensor:
        """return (sum_J, skin_samples, 6)"""
        return torch.cat(self.dense_cond, dim=0)
    
    def get_flatten_dense_skin(self) -> Tensor:
        """return (sum_J, skin_samples)"""
        return torch.cat(self.dense_skin, dim=0)
    
    def get_flatten_full_skin(self) -> Tensor:
        """return (sum_J, N+skin_samples)"""
        # (sum_J, N)
        s = torch.cat(self.uniform_skin, dim=-1).permute(1, 0)
        return torch.cat([s, self.get_flatten_dense_skin()], dim=1)
    
    def get_flatten_full_cond(self) -> Tensor:
        """return (sum_J, N+skin_samples, 6)"""
        return torch.cat([self.get_flatten_uniform_cond(), self.get_flatten_dense_cond()], dim=1)
    
    def get_flatten_indices(self) -> List[int]:
        """return (sum_J)"""
        return [i for i in range(self.B) for _ in range(self.get_len(i=i))]
    
    def true_j(self, i: int, j: int) -> int:
        """return (clamped) corresponding indice in the skeleton"""
        return self.dense_indices[i][self._clamp_j(i=i, j=j)]

@dataclass
class TokenRigResult():
    cond: Optional[Tensor]=None # [vertices, normals]
    cond_latents: Optional[Tensor]=None # (len, dim)
    input_ids: Optional[Tensor]=None # (l,)
    output_ids: Optional[Tensor]=None # (l,)
    skin_pred: Optional[Tensor]=None # (N, J)
    detokenize_output: Optional[DetokenizeOutput]=None
    asset: Optional[Asset]=None

@dataclass
class BoneVaeInput():
    dense_cond: List[Tensor] # [(J, skin_samples, 6)]
    dense_skin: List[Tensor] # [(J, skin_samples)]
    dense_indices: List[List[int]] # [List[J]], corresponding indices of gt
    bones: List[Tensor] # [(J, 6)]
    uniform_cond: Tensor # (B, N, 6)
    uniform_skin: List[Tensor] # [(N, J)]
    
    @property
    def total_samples(self) -> int:
        return self.dense_cond[0].shape[1] + self.uniform_cond.shape[1]
    
    @property
    def B(self) -> int:
        return self.uniform_cond.shape[0]
    
    @property
    def max_J(self) -> int:
        return max([len(s) for s in self.dense_indices])
    
    def get_len(self, i) -> int:
        return len(self.dense_indices[i])
    
    def _clamp_j(self, i: int, j: int) -> int:
        return min(j, len(self.dense_indices[i])-1)
    
    def get_dense_cond(self, j: int) -> Tensor:
        """return (B, skin_samples, 6)"""
        return torch.stack([self.dense_cond[i][self._clamp_j(i=i, j=j)] for i in range(self.B)])
    
    def get_dense_skin(self, j: int) -> Tensor:
        """return (B, skin_samples)"""
        return torch.stack([self.dense_skin[i][self._clamp_j(i=i, j=j)] for i in range(self.B)])
    
    def get_full_cond(self, j: int) -> Tensor:
        """return (B, N+skin_samples, 6)"""
        return torch.cat([self.uniform_cond, self.get_dense_cond(j=j)], dim=1)
    
    def get_uniform_skin(self, j: int) -> Tensor:
        """return (B, N)"""
        return torch.stack([self.uniform_skin[i][:, self._clamp_j(i=i, j=j)] for i in range(self.B)])
    
    def get_full_skin(self, j: int) -> Tensor:
        """return (B, N+skin_samples)"""
        return torch.cat([self.get_uniform_skin(j=j), self.get_dense_skin(j=j)], dim=1)
    
    def get_bones(self, j: int) -> Tensor:
        """return (B, 3)"""
        return torch.stack([self.bones[i][self._clamp_j(i=i, j=j)] for i in range(self.B)])
    
    def get_flatten_bones(self) -> Tensor:
        """return (sum_J, 3)"""
        return torch.cat([self.bones[i] for i in range(self.B)])
    
    def get_flatten_uniform_cond(self) -> Tensor:
        """return (sum_J, N, 6)"""
        return self.uniform_cond[self.get_flatten_indices()]
    
    def get_flatten_dense_cond(self) -> Tensor:
        """return (sum_J, skin_samples, 6)"""
        return torch.cat(self.dense_cond, dim=0)
    
    def get_flatten_dense_skin(self) -> Tensor:
        """return (sum_J, skin_samples)"""
        return torch.cat(self.dense_skin, dim=0)
    
    def get_flatten_full_skin(self) -> Tensor:
        """return (sum_J, N+skin_samples)"""
        # (sum_J, N)
        s = torch.cat(self.uniform_skin, dim=-1).permute(1, 0)
        return torch.cat([s, self.get_flatten_dense_skin()], dim=1)
    
    def get_flatten_full_cond(self) -> Tensor:
        """return (sum_J, N+skin_samples, 6)"""
        return torch.cat([self.get_flatten_uniform_cond(), self.get_flatten_dense_cond()], dim=1)
    
    def get_flatten_indices(self) -> List[int]:
        """return (sum_J)"""
        return [i for i in range(self.B) for _ in range(self.get_len(i=i))]
    
    def true_j(self, i: int, j: int) -> int:
        """return (clamped) corresponding indice in the skeleton"""
        return self.dense_indices[i][self._clamp_j(i=i, j=j)]
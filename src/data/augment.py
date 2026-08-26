"""Augments restored to match the SkinTokens checkpoint's predict_transform:
trim -> affine(normalize_into=[-1, 1]) -> normalize.
Math mirrors VAST-AI/SkinTokens src/data/augment.py exactly (deterministic at predict time).
"""
from abc import ABC
from typing import List, Tuple
import numpy as np
from ..rig_package.info.asset import Asset


def _trans_to_m(v: np.ndarray):
    m = np.eye(4, dtype=np.float32)
    m[0:3, 3] = v
    return m


def _scale_to_m(r):
    m = np.zeros((4, 4), dtype=np.float32)
    if isinstance(r, np.ndarray) and r.size >= 3:
        rr = r.astype(np.float32)
    else:
        v = float(np.asarray(r).reshape(-1)[0])
        rr = np.full(3, v, dtype=np.float32)
    m[0, 0] = rr[0]; m[1, 1] = rr[1]; m[2, 2] = rr[2]
    m[3, 3] = 1.0
    return m


class Augment(ABC):
    @classmethod
    def parse(cls, **kwargs) -> 'Augment':
        return cls()

    def transform(self, asset: Asset, **kwargs):
        pass


class AugmentTrim(Augment):
    def transform(self, asset: Asset, **kwargs):
        asset.trim_skeleton()


class AugmentAffine(Augment):
    def __init__(self, normalize_into=None, random_scale_p=0.0, random_scale=(1.0, 1.0),
                 random_shift_p=0.0, random_shift=(0.0, 0.0)):
        self.normalize_into = normalize_into
        self.random_scale_p = random_scale_p
        self.random_scale = tuple(random_scale)
        self.random_shift_p = random_shift_p
        self.random_shift = tuple(random_shift)

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAffine':
        return cls(
            normalize_into=kwargs.get('normalize_into', [-1.0, 1.0]),
            random_scale_p=kwargs.get('random_scale_p', 0.),
            random_scale=kwargs.get('random_scale', [1., 1.]),
            random_shift_p=kwargs.get('random_shift_p', 0.),
            random_shift=kwargs.get('random_shift', [0., 0.]),
        )

    def transform(self, asset: Asset, **kwargs):
        if asset.vertices is None:
            raise ValueError("do not have vertices")
        bound_min = asset.vertices.min(axis=0)
        bound_max = asset.vertices.max(axis=0)
        if asset.joints is not None:
            bound_min = np.minimum(bound_min, asset.joints.min(axis=0))
            bound_max = np.maximum(bound_max, asset.joints.max(axis=0))

        trans_vertex = np.eye(4, dtype=np.float32)
        trans_vertex = _trans_to_m(-(bound_max + bound_min) / 2) @ trans_vertex

        if self.normalize_into is not None:
            lo, hi = self.normalize_into
            scale = np.max((bound_max - bound_min) / (hi - lo))
            trans_vertex = _scale_to_m(np.array([1. / scale], dtype=np.float32)) @ trans_vertex
            bias = (lo + hi) / 2
            trans_vertex = _trans_to_m(np.array([bias, bias, bias], dtype=np.float32)) @ trans_vertex

        if np.random.rand() < self.random_scale_p:
            s = np.random.uniform(self.random_scale[0], self.random_scale[1])
            trans_vertex = _scale_to_m(np.full(3, s, dtype=np.float32)) @ trans_vertex

        if np.random.rand() < self.random_shift_p:
            l, r = self.random_shift
            shift_vals = np.array([np.random.uniform(l, r), np.random.uniform(l, r), np.random.uniform(l, r)], dtype=np.float32)
            trans_vertex = _trans_to_m(shift_vals) @ trans_vertex

        asset.transform(trans=trans_vertex)


class AugmentNormalize(Augment):
    def transform(self, asset: Asset, **kwargs):
        epsilon = 1e-10
        if asset.vertex_normals is not None:
            n = np.linalg.norm(asset.vertex_normals, axis=1, keepdims=True)
            asset.vertex_normals = np.nan_to_num(asset.vertex_normals / np.maximum(n, epsilon), nan=0., posinf=0., neginf=0.)
        if asset.face_normals is not None:
            n = np.linalg.norm(asset.face_normals, axis=1, keepdims=True)
            asset.face_normals = np.nan_to_num(asset.face_normals / np.maximum(n, epsilon), nan=0., posinf=0., neginf=0.)


_MAP = {
    'trim': AugmentTrim,
    'affine': AugmentAffine,
    'normalize': AugmentNormalize,
}


def get_augments(*args) -> List[Augment]:
    augments = []
    for config in args:
        target = (config or {}).get('__target__')
        if target not in _MAP:
            continue
        c = dict(config)
        c.pop('__target__', None)
        augments.append(_MAP[target].parse(**c))
    return augments

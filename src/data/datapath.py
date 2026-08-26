from abc import abstractmethod, ABC
from collections import defaultdict
from dataclasses import dataclass, field
from numpy import ndarray
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import os

from ..rig_package.info.asset import Asset
from .spec import ConfigSpec


def parse_obj_mesh(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Pure Python/NumPy Wavefront OBJ parser. No external dependencies."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                face_idx = []
                for p in parts:
                    prefix = p.split("/")[0]
                    if prefix.lstrip("-").isdigit():
                        idx = int(prefix)
                        if idx > 0:
                            face_idx.append(idx - 1)
                        elif idx < 0:
                            face_idx.append(len(verts) + idx)
                if len(face_idx) >= 3:
                    for i in range(1, len(face_idx) - 1):
                        faces.append([face_idx[0], face_idx[i], face_idx[i + 1]])

    vertices_np = np.asarray(verts, dtype=np.float32) if verts else np.zeros((0, 3), dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    return vertices_np, faces_np


@dataclass
class LazyAsset(ABC):
    """store datapath and load upon requiring"""
    path: str
    cls: Optional[str] = None

    @abstractmethod
    def load(self) -> 'Asset':
        raise NotImplementedError()


@dataclass
class ObjLazyAsset(LazyAsset):
    """Native high-performance OBJ loader."""
    def load(self) -> 'Asset':
        vertices, faces = parse_obj_mesh(self.path)
        asset = Asset(
            vertices=vertices,
            faces=faces,
            mesh_names=[Path(self.path).stem],
            cls=self.cls,
            path=self.path,
        )
        return asset


@dataclass
class NpzLazyAsset(LazyAsset):
    def load(self) -> 'Asset':
        d = np.load(self.path, allow_pickle=True)
        asset = Asset(
            vertices=d['vertices'],
            faces=d['faces'],
            mesh_names=d.get('mesh_names', None),
            joint_names=d.get('joint_names', None),
            parents=d.get('parents', None),
            lengths=d.get('lengths', None),
            matrix_world=d.get('matrix_world', None),
            matrix_local=d.get('matrix_local', None),
            armature_name=d.get('armature_name', None),
            skin=d.get('skin', None),
            cls=self.cls,
            path=self.path,
        )
        asset.cls = self.cls
        asset.path = self.path
        return asset


@dataclass
class UniRigLazyAsset(LazyAsset):
    """map unirig's data correctly"""
    def load(self) -> 'Asset':
        def bn(x):
            if isinstance(x, ndarray) and x.ndim == 0:
                return x.item()
            return x

        d = np.load(self.path, allow_pickle=True)
        parents = bn(d.get('parents', None))
        if parents is not None:
            parents = [-1 if x is None else x for x in parents]
            parents = np.array(parents)
        matrix_local = bn(d.get('matrix_local', None))
        joints = bn(d.get('joints', None))
        if matrix_local is not None and matrix_local.ndim != 3 and joints is not None:
            matrix_local = np.zeros((joints.shape[0], 4, 4))
            matrix_local[...] = np.eye(4)
            matrix_local[:, :3, 3] = joints
        asset = Asset(
            vertices=d['vertices'],
            faces=d['faces'],
            joint_names=bn(d.get('names', None)),
            parents=parents,
            lengths=bn(d.get('lengths', None)),
            matrix_world=bn(d.get('matrix_world', None)),
            matrix_local=matrix_local,
            armature_name=bn(d.get('armature_name', None)),
            skin=bn(d.get('skin', None)),
            cls=self.cls,
            path=self.path,
        ).change_dtype(float_dtype=np.float32, int_dtype=np.int32)
        asset.cls = self.cls
        asset.path = self.path
        return asset


# Alias for backward compatibility
TrimeshLazyAsset = ObjLazyAsset
BpyLazyAsset = ObjLazyAsset


@dataclass
class Datapath(ConfigSpec):
    """handle input data paths"""
    filepaths: List[str]
    input_dataset_dir: str = ''
    cls_name: Optional[List[str]] = None
    cls_bias: Optional[List[int]] = None
    cls_length: Optional[List[int]] = None
    num_files: Optional[int] = None
    use_prob: bool = False
    cls_weight: Optional[List[float]] = None
    loader: type[LazyAsset] = ObjLazyAsset
    data_name: Optional[str] = None
    ignore_check: bool = False

    vertex_groups: Dict[str, ndarray] = field(default_factory=dict)
    sampled_vertices: Optional[ndarray] = None
    sampled_normals: Optional[ndarray] = None
    sampled_vertex_groups: Optional[Dict[str, ndarray]] = None

    @classmethod
    def parse(cls, **kwargs) -> 'Datapath':
        MAP = {
            None: ObjLazyAsset,
            'obj': ObjLazyAsset,
            'trimesh': ObjLazyAsset,
            'npz': NpzLazyAsset,
            'unirig': UniRigLazyAsset,
        }
        input_dataset_dir = kwargs.get('input_dataset_dir', '')
        num_files = kwargs.get('num_files', None)
        use_prob = kwargs.get('use_prob', False)
        data_name = kwargs.get('data_name', 'raw_data.npz')
        loader_name = kwargs.get('loader', None)
        loader_cls = MAP.get(loader_name, ObjLazyAsset)
        ignore_check = kwargs.get('ignore_check', False)

        _filepaths = kwargs.get('filepaths', [])
        if isinstance(_filepaths, list):
            filepaths = _filepaths
            cls_name = None
            cls_bias = None
            cls_length = None
            cls_weight = None
        elif isinstance(_filepaths, dict):
            filepaths = []
            cls_name = []
            cls_bias = []
            cls_length = []
            cls_weight = []
            for k, v in _filepaths.items():
                assert isinstance(v, list), "items in the dict must be a list of paths"
                cls_name.append(k)
                cls_bias.append(len(filepaths))
                cls_length.append(len(v))
                cls_weight.append(1.0)
                filepaths.extend(v)
        else:
            filepaths = []
            cls_name = None
            cls_bias = None
            cls_length = None
            cls_weight = None

        return Datapath(
            filepaths=filepaths,
            input_dataset_dir=input_dataset_dir,
            cls_name=cls_name,
            cls_bias=cls_bias,
            cls_length=cls_length,
            num_files=num_files,
            use_prob=use_prob,
            cls_weight=cls_weight,
            loader=loader_cls,
            data_name=data_name,
            ignore_check=ignore_check,
        )

    def make(self, path: str, cls: str | None) -> LazyAsset:
        return self.loader(path=path, cls=cls)

    def __getitem__(self, index: int) -> LazyAsset:
        path = os.path.join(self.input_dataset_dir, self.filepaths[index])
        if self.data_name is not None:
            path = os.path.join(path, self.data_name)
        name = None
        if self.cls_name is not None and self.cls_bias is not None and self.cls_length is not None:
            for i in range(len(self.cls_bias)):
                start = self.cls_bias[i]
                end = start + self.cls_length[i]
                if start <= index < end:
                    name = self.cls_name[i]
                    break
        return self.make(path=path, cls=name)

    def get_data(self) -> List[LazyAsset]:
        return [self[i] for i in range(len(self))]

    def split_by_cls(self) -> Dict[str | None, 'Datapath']:
        res: Dict[str | None, Datapath] = {}
        if self.cls_name is None:
            res[None] = self
            return res
        if self.cls_bias is None:
            raise ValueError("do not have cls_bias")
        if self.cls_length is None:
            raise ValueError("do not have cls_length")
        d_filepaths = defaultdict(list)
        for (i, cls) in enumerate(self.cls_name):
            s = slice(self.cls_bias[i], self.cls_bias[i] + self.cls_length[i])
            d_filepaths[cls].extend(self.filepaths[s].copy())
        for cls in d_filepaths:
            res[cls] = Datapath(
                filepaths=d_filepaths[cls],
                input_dataset_dir=self.input_dataset_dir,
                cls_name=[cls],
                cls_bias=[0],
                cls_length=[len(d_filepaths[cls])],
                num_files=self.num_files,
                use_prob=self.use_prob,
                cls_weight=None,
                loader=self.loader,
                data_name=self.data_name,
            )
        return res

    def __len__(self):
        return len(self.filepaths)
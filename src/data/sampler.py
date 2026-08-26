from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from ..rig_package.info.asset import Asset
from ..rig_package.utils import sample_vertex_groups
from .spec import ConfigSpec


@dataclass
class SampleResult:
    sampled_vertices: np.ndarray
    sampled_normals: np.ndarray
    sampled_vertex_groups: Dict[str, np.ndarray]
    skin_samples: Optional[int] = None


@dataclass
class Sampler(ConfigSpec):
    num_samples: int = 4096
    num_vertex_samples: Optional[int] = None
    shuffle: bool = True

    @classmethod
    def parse(cls, **kwargs) -> 'Sampler':
        return Sampler(
            num_samples=kwargs.get('num_samples', 4096),
            num_vertex_samples=kwargs.get('num_vertex_samples', None),
            shuffle=kwargs.get('shuffle', True),
        )

    def sample(self, asset: Asset) -> SampleResult:
        if asset.vertices is None or asset.faces is None:
            raise ValueError("Asset missing vertices or faces")
        
        if asset.vertex_normals is None or asset.face_normals is None:
            asset.build_normals()

        vg_dict = {}
        sampled_vertices = None
        sampled_normals = None

        if asset.vertex_groups:
            for name, vg in asset.vertex_groups.items():
                s_verts, s_norms, s_vg = sample_vertex_groups(
                    vertices=asset.vertices,
                    faces=asset.faces,
                    num_samples=self.num_samples,
                    num_vertex_samples=self.num_vertex_samples,
                    vertex_normals=asset.vertex_normals,
                    face_normals=asset.face_normals,
                    vertex_groups=vg,
                    shuffle=self.shuffle,
                    same=True,
                )
                if s_verts.ndim == 3:
                    sampled_vertices = s_verts[:, 0]
                else:
                    sampled_vertices = s_verts
                if s_norms is not None:
                    sampled_normals = s_norms[:, 0] if s_norms.ndim == 3 else s_norms
                vg_dict[name] = s_vg
        else:
            s_verts, s_norms, _ = sample_vertex_groups(
                vertices=asset.vertices,
                faces=asset.faces,
                num_samples=self.num_samples,
                num_vertex_samples=self.num_vertex_samples,
                vertex_normals=asset.vertex_normals,
                face_normals=asset.face_normals,
                vertex_groups=asset.skin,
                shuffle=self.shuffle,
                same=True,
            )
            sampled_vertices = s_verts[:, 0] if s_verts.ndim == 3 else s_verts
            if s_norms is not None:
                sampled_normals = s_norms[:, 0] if s_norms.ndim == 3 else s_norms

        if sampled_normals is None and sampled_vertices is not None:
            sampled_normals = np.zeros_like(sampled_vertices)

        return SampleResult(
            sampled_vertices=sampled_vertices,
            sampled_normals=sampled_normals,
            sampled_vertex_groups=vg_dict,
            skin_samples=None,
        )


def get_sampler(**kwargs) -> Sampler:
    return Sampler.parse(**kwargs)
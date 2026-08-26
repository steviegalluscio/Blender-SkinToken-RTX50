from dataclasses import dataclass
from typing import List, Optional, Tuple
from .spec import ConfigSpec


@dataclass
class Order(ConfigSpec):
    orders: Optional[List[str]] = None
    skeleton_path: Optional[str] = None
    root: Optional[str] = None
    merge_skin: bool = True
    do_not_normalize: bool = False

    @classmethod
    def parse(cls, **kwargs) -> 'Order':
        return Order(
            orders=kwargs.get('orders', None),
            skeleton_path=kwargs.get('skeleton_path', None),
            root=kwargs.get('root', None),
            merge_skin=kwargs.get('merge_skin', True),
            do_not_normalize=kwargs.get('do_not_normalize', False),
        )

    def arrange_names(self, cls: Optional[str], names: List[str], parents: List[int]) -> Tuple[List[str], List[int]]:
        return names, parents
import inspect
from typing import Any, Callable, Dict, Optional, Type, Union

Registrable = Union[Type[Any], Callable[..., Any]]


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._module_dict: Dict[str, Registrable] = {}

    def register(
        self, name: Optional[str] = None
    ) -> Callable[[Registrable], Registrable]:
        def _register(cls_or_func: Registrable) -> Registrable:
            reg_name = name or cls_or_func.__name__
            self._module_dict[reg_name] = cls_or_func
            return cls_or_func

        return _register

    def get(self, name: str) -> Registrable:
        if name not in self._module_dict:
            raise KeyError(f'{name} is not registered in {self.name}')
        return self._module_dict[name]

    def build(self, cfg: Dict[str, Any], **kwargs: Any) -> Any:
        """Factory method."""
        obj_type = cfg.pop('type')
        obj_cls = self.get(obj_type)

        full_args = {**cfg, **kwargs}

        sig = inspect.signature(obj_cls)
        valid_args = {k: v for k, v in full_args.items() if k in sig.parameters}
        return obj_cls(**valid_args)


ENCODERS = Registry('encoders')
DATASETS = Registry('datasets')

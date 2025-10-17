from dataclasses import dataclass
from typing import Any, Dict
import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in path.split('.'):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def load_config(path: str) -> Config:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Config(raw=data)



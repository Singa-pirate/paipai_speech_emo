"""配置文件模塊

提供配置文件的加載、保存和路徑解析功能
用於管理模型訓練和推論的配置參數
"""
import os
from pathlib import Path

import yaml


def load_config(path):
    """加載YAML配置文件

    Args:
        path: 配置文件路徑

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 當配置文件不存在時
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config, path):
    """保存配置字典到YAML文件

    Args:
        config: 配置字典
        path: 保存路徑
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def resolve_path(path_value):
    """解析路徑，展開環境變數

    Args:
        path_value: 路徑值

    Returns:
        解析後的路徑字符串，如果path_value為None則返回None
    """
    if path_value is None:
        return None
    return os.path.expandvars(str(path_value))

"""日誌配置模塊

提供日誌記錄器的設置功能
用於記錄模型訓練和推論過程中的信息
"""
import logging
from pathlib import Path


def setup_logging(log_dir, name="train"):
    """設置日誌記錄器

    創建一個同時輸出到文件和控制台的日誌記錄器
    日誌格式包含時間、日誌級別和消息內容

    Args:
        log_dir: 日誌文件保存目錄
        name: 記錄器名稱，默認為"train"

    Returns:
        配置好的日誌記錄器對象
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

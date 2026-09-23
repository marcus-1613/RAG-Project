"""PDF 去重模块。

通过计算 PDF 文件的 SHA-256 哈希值判断文件是否已经入库，
避免对重复上传的 PDF 重复生成 Embedding。
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_hash(file_path: str | Path) -> str:
    """计算文件的 SHA-256 哈希值。

    Args:
        file_path: 文件路径。

    Returns:
        str: 文件的十六进制 SHA-256 哈希值。
    """
    file_path = Path(file_path)
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        # 分块读取，避免大文件一次性载入内存
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

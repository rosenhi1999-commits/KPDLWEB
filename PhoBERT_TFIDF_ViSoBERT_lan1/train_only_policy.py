"""Read train.csv and locally generated artifacts only."""
import os
from pathlib import Path
import sys


def _audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path.suffix.lower() != '.csv':
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    reading = ('r' in mode or '+' in mode) if isinstance(mode, str) else (flags & os.O_ACCMODE) != os.O_WRONLY
    if not reading or path.name == 'train.csv':
        return
    root = os.environ.get('ABSA_ARTIFACT_ROOT')
    if root and path.is_relative_to(Path(root).resolve()):
        return
    raise PermissionError('Chỉ đọc train.csv và các kết quả do lượt chạy tạo ra.')


sys.addaudithook(_audit)

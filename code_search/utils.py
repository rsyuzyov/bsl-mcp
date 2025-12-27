"""Утилиты форматирования и системные функции."""
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional


def format_time(ts: float) -> str:
    """Форматирует timestamp в HH:MM:SS."""
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def format_duration(sec: float) -> str:
    """Форматирует секунды в MM:SS или HH:MM:SS."""
    sec = int(sec)
    if sec < 3600:
        return f"{sec // 60}:{sec % 60:02d}"
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def find_locking_process(path: Path) -> Optional[int]:
    """Найти PID процесса, блокирующего папку (кроссплатформенно).
    
    Returns:
        PID процесса или None если не найден
    """
    try:
        import psutil
    except ImportError:
        return None
    
    target = str(path).lower()
    current_pid = os.getpid()
    
    for proc in psutil.process_iter(['pid', 'name', 'open_files', 'cmdline']):
        try:
            if proc.info['pid'] == current_pid:
                continue
            
            # Проверяем открытые файлы
            open_files = proc.info.get('open_files') or []
            for f in open_files:
                if target in str(f.path).lower():
                    return proc.info['pid']
            
            # Проверяем cmdline для python процессов
            name = proc.info.get('name', '').lower()
            if 'python' in name:
                cmdline = proc.info.get('cmdline') or []
                cmdline_str = ' '.join(cmdline).lower()
                if 'run.py' in cmdline_str or 'bsl-mcp' in cmdline_str:
                    try:
                        cwd = proc.cwd()
                        if 'bsl-mcp' in cwd.lower():
                            return proc.info['pid']
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    return None


def check_index_lock(index_dir: str) -> Tuple[bool, Optional[int], Optional[str]]:
    """Проверить, заблокирована ли папка индекса (кроссплатформенно).
    
    Args:
        index_dir: Путь к папке индекса
        
    Returns:
        (is_locked, pid, message)
        - is_locked: True если заблокировано
        - pid: PID процесса-владельца (если найден)
        - message: Сообщение об ошибке с подсказкой
    """
    qdrant_path = Path(index_dir) / "qdrant"
    
    if not qdrant_path.exists():
        return False, None, None
    
    # Qdrant создаёт .lock файл
    lock_file = qdrant_path / ".lock"
    
    if not lock_file.exists():
        return False, None, None
    
    # Пробуем открыть lock-файл эксклюзивно
    is_locked = False
    try:
        if sys.platform == "win32":
            import msvcrt
            fd = os.open(str(lock_file), os.O_RDWR)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except (IOError, OSError):
                is_locked = True
            finally:
                os.close(fd)
        else:
            # Linux / macOS
            import fcntl
            fd = os.open(str(lock_file), os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except (IOError, OSError):
                is_locked = True
            finally:
                os.close(fd)
    except Exception:
        # Не смогли проверить — считаем не заблокированным
        return False, None, None
    
    if not is_locked:
        return False, None, None
    
    # Папка заблокирована, ищем владельца
    pid = find_locking_process(qdrant_path)
    
    if pid:
        if sys.platform == "win32":
            kill_cmd = f"taskkill /F /PID {pid}"
        else:
            kill_cmd = f"kill -9 {pid}"
        message = (
            f"Папка индекса заблокирована процессом PID={pid}.\n"
            f"Для завершения выполните: {kill_cmd}"
        )
    else:
        message = (
            f"Папка индекса '{qdrant_path}' заблокирована другим процессом.\n"
            f"Возможно, запущен другой экземпляр приложения."
        )
    
    return True, pid, message

"""Менеджер конфигурации."""
import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .logger import get_logger

logger = get_logger("app.config")

DEFAULT_PORT = 8000
DEFAULT_CHECK_INTERVAL = 300
DEFAULT_CONFIG_FILE = "config.yaml"


def _diff_dicts(old: dict, new: dict, path: str = "") -> list[str]:
    """Рекурсивное сравнение двух словарей, возвращает список изменений."""
    changes = []
    all_keys = set(old.keys()) | set(new.keys())
    
    for key in all_keys:
        full_path = f"{path}.{key}" if path else key
        
        if key not in old:
            changes.append(f"  + {full_path}: {new[key]}")
        elif key not in new:
            changes.append(f"  - {full_path}: {old[key]}")
        elif old[key] != new[key]:
            if isinstance(old[key], dict) and isinstance(new[key], dict):
                changes.extend(_diff_dicts(old[key], new[key], full_path))
            elif isinstance(old[key], list) and isinstance(new[key], list):
                changes.extend(_diff_lists(old[key], new[key], full_path))
            else:
                changes.append(f"  ~ {full_path}: {old[key]} -> {new[key]}")
    
    return changes


def _diff_lists(old: list, new: list, path: str) -> list[str]:
    """Сравнение списков (для ibs)."""
    changes = []
    
    # Для списков ИБ сравниваем по name
    old_by_name = {item.get("name", i): item for i, item in enumerate(old) if isinstance(item, dict)}
    new_by_name = {item.get("name", i): item for i, item in enumerate(new) if isinstance(item, dict)}
    
    for name in set(old_by_name.keys()) | set(new_by_name.keys()):
        item_path = f"{path}[{name}]"
        if name not in old_by_name:
            changes.append(f"  + {item_path}: добавлено")
        elif name not in new_by_name:
            changes.append(f"  - {item_path}: удалено")
        elif old_by_name[name] != new_by_name[name]:
            changes.extend(_diff_dicts(old_by_name[name], new_by_name[name], item_path))
    
    return changes


@dataclass
class IBConfig:
    """Конфигурация одной информационной базы."""
    name: str  # Уникальный идентификатор (slug)
    title: str = ""
    source_dir: str = ""
    index_dir: str = ""
    embedding_model: str = "cointegrated/rubert-tiny2"
    embedding_device: str = "cpu"  # cpu, gpu, dml
    embedding_mode: str = "full"  # full | methods | signatures
    vector_db: str = "qdrant"

    def __post_init__(self):
        if not self.title:
            self.title = self.name


@dataclass
class GlobalConfig:
    """Глобальная конфигурация приложения."""
    port: int = DEFAULT_PORT
    check_interval: int = DEFAULT_CHECK_INTERVAL
    log_level: str = "INFO"
    scan_workers: int = 8
    ibs: list[IBConfig] = field(default_factory=list)


class ConfigManager:
    """Менеджер загрузки и сохранения конфигурации."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_FILE):
        self.config_path = Path(config_path)
        self.config = GlobalConfig()

    def load(self) -> GlobalConfig:
        """Загрузить конфигурацию из файла."""
        if not self.config_path.exists():
            # Use logger if initialized, otherwise print? 
            # Actually logger is not init yet when loading config usually.
            print(f"Конфиг {self.config_path} не найден, создаю дефолтный.")
            self.save()
            return self.config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            global_data = data.get("global", {})
            self.config.port = global_data.get("port", DEFAULT_PORT)
            self.config.check_interval = global_data.get("check_interval", DEFAULT_CHECK_INTERVAL)
            self.config.log_level = global_data.get("log_level", "INFO")
            self.config.scan_workers = global_data.get("scan_workers", 8)

            self.config.ibs = []
            for ib_data in data.get("ibs", []):
                self.config.ibs.append(IBConfig(**ib_data))
            
            return self.config
        except Exception as e:
            print(f"Ошибка загрузки конфига: {e}")
            return self.config

    def save(self):
        """Сохранить конфигурацию в файл."""
        new_data = {
            "global": {
                "port": self.config.port,
                "check_interval": self.config.check_interval,
                "log_level": self.config.log_level,
                "scan_workers": self.config.scan_workers
            },
            "ibs": [asdict(ib) for ib in self.config.ibs]
        }
        
        # Читаем старый конфиг для сравнения
        old_data = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    old_data = yaml.safe_load(f) or {}
            except Exception:
                pass
        
        # Сравниваем и логируем изменения
        changes = _diff_dicts(old_data, new_data)
        if changes:
            logger.info(f"Сохранение конфигурации. Изменения:\n" + "\n".join(changes))
        else:
            logger.debug("Сохранение конфигурации (без изменений)")
        
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(new_data, f, allow_unicode=True, sort_keys=False)

    def add_ib(self, ib: IBConfig, overwrite: bool = False):
        """Добавить новую ИБ."""
        # Проверка дубликатов
        for i, existing in enumerate(self.config.ibs):
            if existing.name == ib.name:
                if not overwrite:
                    raise ValueError(f"ИБ с именем {ib.name} уже существует")
                # Заменяем существующую
                self.config.ibs[i] = ib
                self.save()
                return

        self.config.ibs.append(ib)
        self.save()

    def remove_ib(self, name: str):
        """Удалить ИБ."""
        self.config.ibs = [ib for ib in self.config.ibs if ib.name != name]
        self.save()

    def get_ib(self, name: str) -> IBConfig | None:
        """Получить конфиг ИБ по имени."""
        for ib in self.config.ibs:
            if ib.name == name:
                return ib
        return None

"""Утилиты для работы с метаданными 1С."""
import re
from dataclasses import dataclass
from typing import Optional


# Маппинг английских названий папок на русские типы объектов
OBJECT_TYPE_MAP = {
    "Catalogs": "Справочник",
    "Documents": "Документ",
    "Reports": "Отчёт",
    "DataProcessors": "Обработка",
    "CommonModules": "ОбщийМодуль",
    "InformationRegisters": "РегистрСведений",
    "AccumulationRegisters": "РегистрНакопления",
    "CalculationRegisters": "РегистрРасчета",
    "AccountingRegisters": "РегистрБухгалтерии",
    "Constants": "Константа",
    "Enums": "Перечисление",
    "ChartsOfCharacteristicTypes": "ПВХ",
    "ChartsOfAccounts": "ПланСчетов",
    "ChartsOfCalculationTypes": "ПланВидовРасчета",
    "BusinessProcesses": "БизнесПроцесс",
    "Tasks": "Задача",
    "ExchangePlans": "ПланОбмена",
    "WebServices": "WebСервис",
    "HTTPServices": "HTTPСервис",
    "ScheduledJobs": "РегламентноеЗадание",
    "Sequences": "Последовательность",
    "DocumentJournals": "ЖурналДокументов",
    "FilterCriteria": "КритерийОтбора",
    "SettingsStorages": "ХранилищеНастроек",
    "FunctionalOptions": "ФункциональнаяОпция",
    "FunctionalOptionsParameters": "ПараметрФО",
    "CommonForms": "ОбщаяФорма",
    "CommonCommands": "ОбщаяКоманда",
    "CommandGroups": "ГруппаКоманд",
    "CommonTemplates": "ОбщийМакет",
    "CommonPictures": "ОбщаяКартинка",
    "SessionParameters": "ПараметрСеанса",
    "Roles": "Роль",
    "Subsystems": "Подсистема",
    "StyleItems": "ЭлементСтиля",
    "Styles": "Стиль",
    "Languages": "Язык",
    "ExternalDataSources": "ВнешнийИсточникДанных",
}

# Типы модулей
MODULE_TYPE_MAP = {
    "ObjectModule": "МодульОбъекта",
    "ManagerModule": "МодульМенеджера",
    "RecordSetModule": "МодульНабораЗаписей",
    "ValueManagerModule": "МодульМенеджераЗначения",
    "Module": "МодульФормы",
    "CommandModule": "МодульКоманды",
}


@dataclass
class ObjectMetadata:
    """Метаданные объекта 1С, извлечённые из пути файла."""
    object_type: str          # Тип объекта (русский): "Справочник", "Документ" и т.д.
    object_type_en: str       # Тип объекта (английский): "Catalogs", "Documents"
    object_name: str          # Имя объекта: "Контрагенты", "Заказы"
    module_type: str          # Тип модуля: "МодульОбъекта", "МодульМенеджера"
    form_name: Optional[str]  # Имя формы (если это модуль формы)


def parse_1c_path(rel_path: str) -> Optional[ObjectMetadata]:
    """
    Извлечь метаданные из относительного пути файла в выгрузке 1С.
    
    Примеры:
        "Catalogs\\Контрагенты\\Ext\\ObjectModule.bsl" 
            → ObjectMetadata(object_type="Справочник", object_name="Контрагенты", ...)
        
        "CommonModules\\ОбщегоНазначения\\Ext\\Module.bsl"
            → ObjectMetadata(object_type="ОбщийМодуль", object_name="ОбщегоНазначения", ...)
        
        "Documents\\Заказ\\Forms\\ФормаДокумента\\Ext\\Form\\Module.bsl"
            → ObjectMetadata(object_type="Документ", object_name="Заказ", form_name="ФормаДокумента", ...)
    """
    # Нормализуем разделители
    path = rel_path.replace("/", "\\")
    parts = path.split("\\")
    
    if len(parts) < 2:
        return None
    
    object_type_en = parts[0]
    
    if object_type_en not in OBJECT_TYPE_MAP:
        return None
    
    object_type = OBJECT_TYPE_MAP[object_type_en]
    object_name = parts[1]
    
    # Определяем тип модуля
    module_type = ""
    form_name = None
    
    # Ищем имя файла модуля
    filename = parts[-1] if parts else ""
    if filename.endswith(".bsl"):
        module_name = filename[:-4]  # убираем .bsl
        module_type = MODULE_TYPE_MAP.get(module_name, module_name)
    
    # Проверяем, есть ли это форма
    if "Forms" in parts:
        forms_idx = parts.index("Forms")
        if forms_idx + 1 < len(parts):
            form_name = parts[forms_idx + 1]
            module_type = "МодульФормы"
    
    return ObjectMetadata(
        object_type=object_type,
        object_type_en=object_type_en,
        object_name=object_name,
        module_type=module_type,
        form_name=form_name
    )


def format_object_context(metadata: ObjectMetadata) -> str:
    """
    Форматировать контекст объекта для добавления к эмбеддингу.
    
    Это добавляется к тексту чанка перед эмбеддингом для улучшения семантики.
    """
    parts = [metadata.object_type, metadata.object_name]
    
    if metadata.form_name:
        parts.append(f"Форма: {metadata.form_name}")
    
    if metadata.module_type:
        parts.append(metadata.module_type)
    
    return " | ".join(parts)


# Паттерны директив компиляции для фильтрации
COMPILATION_DIRECTIVES = [
    "&НаКлиенте",
    "&НаСервере",
    "&НаСервереБезКонтекста",
    "&НаКлиентеНаСервереБезКонтекста",
    "&НаКлиентеНаСервере",
]


def is_compilation_directive_match(text: str, query: str) -> bool:
    """
    Проверить, является ли совпадение ложноположительным из-за директивы компиляции.
    
    Пример: запрос "клиент" не должен находить "&НаКлиенте"
    """
    query_lower = query.lower()
    
    # Если запрос — часть директивы компиляции, это ложное совпадение
    for directive in COMPILATION_DIRECTIVES:
        if query_lower in directive.lower():
            # Проверяем, содержит ли текст эту директиву
            if directive.lower() in text.lower():
                return True
    
    return False

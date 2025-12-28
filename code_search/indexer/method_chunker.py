"""Извлечение методов (процедур/функций) из BSL-кода."""
import re
from dataclasses import dataclass


@dataclass
class MethodChunk:
    """Структура извлечённого метода."""
    name: str              # ПолучитьОстаткиТоваров
    name_split: str        # получить остатки товаров
    signature: str         # Функция ПолучитьОстаткиТоваров(Склад, Дата) Экспорт
    annotations: str       # &НаСервере\n&Перед("...")
    doc_comment: str       # // Получает остатки по складу...
    line_number: int       # Номер строки объявления (1-based)
    is_function: bool      # True = Функция, False = Процедура
    is_export: bool        # Экспортный метод
    text: str              # Полный текст для эмбеддинга


def split_camel_case(name: str) -> str:
    """Разбивает CamelCase на отдельные слова (поддержка кириллицы).
    
    Пример: ПолучитьОстаткиТоваров -> получить остатки товаров
    """
    # Вставляем пробел между строчной и заглавной буквой
    result = re.sub(r'([а-яёa-z])([А-ЯЁA-Z])', r'\1 \2', name)
    # Вставляем пробел между заглавными, если за ними строчная (для аббревиатур)
    result = re.sub(r'([А-ЯЁA-Z]+)([А-ЯЁA-Z][а-яёa-z])', r'\1 \2', result)
    return result.lower()


# Regex для поиска объявлений методов
# Поддержка многострочных объявлений, аннотаций, комментариев
_METHOD_PATTERN = re.compile(
    r'''
    # Аннотации перед методом (опционально, несколько строк)
    ((?:&[^\n]+\n)*)
    # Ключевое слово
    (Процедура|Функция|Procedure|Function)\s+
    # Имя метода
    (\w+)\s*
    # Параметры (могут быть на нескольких строках)
    \(([^)]*)\)
    # Экспорт (опционально)
    (\s+Экспорт|\s+Export)?
    ''',
    re.IGNORECASE | re.VERBOSE | re.MULTILINE | re.DOTALL
)


def _extract_doc_comments(content: str, method_start: int, max_lines: int = 10) -> str:
    """Извлекает комментарии перед методом.
    
    Ищет блок комментариев непосредственно перед объявлением метода.
    """
    # Находим начало строки с методом
    line_start = content.rfind('\n', 0, method_start) + 1
    
    # Берём текст ДО метода
    before_method = content[:line_start]
    lines = before_method.split('\n')
    
    # Собираем комментарии снизу вверх
    comment_lines = []
    for line in reversed(lines[-max_lines:]):
        stripped = line.strip()
        if stripped.startswith('//'):
            comment_lines.insert(0, stripped)
        elif stripped.startswith('&'):
            # Аннотации уже захватываются regex, пропускаем
            continue
        elif stripped == '':
            # Пустая строка — продолжаем, если уже есть комментарии
            if comment_lines:
                continue
        else:
            # Не-комментарий — прекращаем
            break
    
    return '\n'.join(comment_lines)


def extract_methods(content: str, include_body_preview: bool = False, body_preview_lines: int = 5) -> list[MethodChunk]:
    """Извлекает все методы из BSL-кода.
    
    Args:
        content: Исходный код BSL
        include_body_preview: Включать ли первые строки тела метода
        body_preview_lines: Сколько строк тела включать
        
    Returns:
        Список MethodChunk с извлечёнными методами
    """
    methods = []
    
    # Разбиваем на строки для подсчёта номеров
    lines = content.split('\n')
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line) + 1)  # +1 для \n
    
    def offset_to_line(offset: int) -> int:
        """Преобразует смещение в номер строки (1-based)."""
        for i, lo in enumerate(line_offsets):
            if lo > offset:
                return i
        return len(lines)
    
    for match in _METHOD_PATTERN.finditer(content):
        annotations_raw = match.group(1).strip()
        keyword = match.group(2)
        name = match.group(3)
        params_raw = match.group(4)
        export_raw = match.group(5)
        
        # Определяем тип
        is_function = keyword.lower() in ('функция', 'function')
        is_export = bool(export_raw and export_raw.strip())
        
        # Формируем сигнатуру (нормализуем пробелы в параметрах)
        params_clean = ' '.join(params_raw.split())
        export_str = ' Экспорт' if is_export else ''
        signature = f"{keyword} {name}({params_clean}){export_str}"
        
        # Номер строки
        line_number = offset_to_line(match.start())
        
        # Doc-комментарии
        # Ищем перед аннотациями, если они есть
        comment_search_start = match.start() - len(annotations_raw) if annotations_raw else match.start()
        doc_comment = _extract_doc_comments(content, comment_search_start)
        
        # Превью тела (опционально)
        body_preview = ""
        if include_body_preview:
            method_end = match.end()
            # Берём следующие N строк после объявления
            end_line = offset_to_line(method_end)
            preview_lines = lines[end_line:end_line + body_preview_lines]
            body_preview = '\n'.join(line.strip() for line in preview_lines if line.strip())
        
        # Формируем текст для эмбеддинга
        name_split = split_camel_case(name)
        
        text_parts = []
        if annotations_raw:
            text_parts.append(annotations_raw)
        text_parts.append(signature)
        text_parts.append(f"// {name_split}")  # Добавляем разбитое имя как псевдо-комментарий
        if doc_comment:
            text_parts.append(doc_comment)
        if body_preview:
            text_parts.append(body_preview)
        
        text = '\n'.join(text_parts)
        
        methods.append(MethodChunk(
            name=name,
            name_split=name_split,
            signature=signature,
            annotations=annotations_raw,
            doc_comment=doc_comment,
            line_number=line_number,
            is_function=is_function,
            is_export=is_export,
            text=text
        ))
    
    return methods


def extract_method_chunks(content: str, mode: str = "methods") -> list[tuple[str, dict]]:
    """Извлекает чанки методов для индексации.
    
    Args:
        content: Исходный код BSL
        mode: Режим извлечения:
            - "methods" — сигнатура + комменты + превью тела
            - "signatures" — только сигнатура + комменты
            
    Returns:
        Список кортежей (text, metadata) для индексации
    """
    include_body = (mode == "methods")
    methods = extract_methods(content, include_body_preview=include_body)
    
    chunks = []
    for m in methods:
        metadata = {
            "func_name": m.name,
            "line_number": m.line_number,
            "is_function": m.is_function,
            "is_export": m.is_export,
        }
        chunks.append((m.text, metadata))
    
    return chunks

# 1C Code Search MCP

Локальный MCP-сервер для семантического поиска по кодовой базе 1С. Поддерживает множество информационных баз, различные модели эмбеддингов и векторные движки.

## Ключевые возможности

*   **Мульти-конфигурация**: Поддержка нескольких баз кода (информационных баз) одновременно.
*   **Гибкий выбор моделей**: Поддержка любых моделей совместимых с HuggingFace/ONNX (по умолчанию `cointegrated/rubert-tiny2`).
*   **Векторный поиск**: Использование Qdrant для быстрого и эффективного поиска.
*   **Веб-интерфейс**: Удобное управление базами и настройками через браузер.
*   **MCP Протокол**: Легкая интеграция с Claude, Cursor, Kiro и другими MCP-клиентами.

## Установка

1.  Клонируйте репозиторий.
2.  Установите зависимости:

```bash
python install.py
```

## Запуск

```bash
python code-search.py
```
или
```bash
python -m code_search
```

Сервер запустится на порту 8000 (по умолчанию).
Откройте **http://localhost:8000** в браузере для настройки баз.

## Конфигурация (`config.yaml`)

Файл `config.yaml` создается автоматически при первом запуске. Вы можете редактировать его вручную или через веб-интерфейс.

Пример конфигурации:

```yaml
global:
  port: 8000
  check_interval: 300

ibs:
  - name: "trade"
    title: "Управление Торговлей"
    source_dir: "C:/Projects/Trade/src"
    index_dir: "./indices/trade"
    embedding_model: "cointegrated/rubert-tiny2"
    vector_db: "qdrant"

  - name: "erp"
    title: "1С:ERP"
    source_dir: "C:/Projects/ERP/src"
    index_dir: "./indices/erp"
    embedding_model: "intfloat/multilingual-e5-small"
    vector_db: "qdrant"
```

### Параметры

**Global:**
*   `port`: Порт веб-сервера.
*   `check_interval`: Интервал фоновой проверки изменений файлов (в секундах).

**IBs (Информационные базы):**
*   `name`: Уникальный ID базы (латиница).
*   `title`: Человекочитаемое название.
*   `source_dir`: Путь к каталогу с выгрузкой конфигурации (XML файлы).
*   `index_dir`: Путь для хранения векторного индекса.
*   `embedding_model`: Имя модели эмбеддингов.
*   `vector_db`: Исполнитель поиска (`qdrant`).

## Поддерживаемые модели и движки

### Модели эмбеддингов

Сервер автоматически загружает и конвертирует в ONNX модели с HuggingFace. Рекомендуемые:

1.  `cointegrated/rubert-tiny2` (Default) — Быстрая, легкая, хорошая точность.
2.  `intfloat/multilingual-e5-small` — Высокая точность, мультиязычность.
3.  `ai-forever/sbert_large_nlu_ru` — Крупная модель для русского языка.

### Векторные движки

*   **Qdrant** — Используется по умолчанию. Работает локально, хранит данные в `index_dir`.
*   *(В планах: LanceDB, ChromaDB)*

## Подключение к клиентам (MCP)

### Kiro / Supermaven

В файл `~/.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "1c-rag": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

### Claude Desktop config

```json
{
  "mcpServers": {
    "1c-rag": {
      "command": "python",
      "args": ["C:\\путь\\к\\repo\\bsl-mcp\\code-search.py"]
    }
  }
}
```
*Примечание: Для stdio подключения может потребоваться доработка entrypoint, рекомендуется использовать SSE подключение если клиент поддерживает его.*

## Лицензия

MIT

# 1C Code Search MCP

Локальный MCP-сервер для поиска по кодовой базе 1С. Работает на CPU, без видеокарты.

## Установка и запуск

```bash
pip install -r requirements.txt
python code-search.py --source ./путь-к-выгрузке-1с
```

Или через install.py:
```bash
python install.py
python code-search.py --source ./путь-к-выгрузке-1с
```

## Подключение к Kiro

Добавь в `~/.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "1c-rag": {
      "url": "http://localhost:8000"
    }
  }
}
```

## Параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--source` | — | Каталог выгрузки 1С (обязательный) |
| `--port` | 8000 | Порт сервера |
| `--index` | ./1c-index | Каталог для индекса |

## Как работает

Индексирует XML-файлы выгрузки через sentence-transformers (all-MiniLM-L6-v2) + FAISS. При повторном запуске использует готовый индекс.

## Характеристики

| Аспект | Python RAG (LlamaIndex+FAISS) |
|--------|-------------------------------|
| Индексация 5ГБ | 2-5мин (параллельно) |
| RAM | 1-2ГБ (FAISS mmap) |
| Поиск | 0.1-0.3с, +XML парсинг |
| Точность 1С | Высокая (NodeParser + metadata) |
| Зависимости | pip (0.5ГБ) |
| MCP setup | 5 строк FastAPI |


## Лицензия

MIT

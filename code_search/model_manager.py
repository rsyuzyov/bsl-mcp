"""Менеджер моделей эмбеддингов."""
import threading
import os
from dataclasses import dataclass
from pathlib import Path


from .logger import get_logger

logger = get_logger("app.models")

@dataclass
class ModelInfo:
    """Информация о загруженной модели."""
    model: object
    tokenizer: object
    name: str
    loading: bool = True
    error: str | None = None


# Папка для кэширования ONNX моделей
ONNX_CACHE_DIR = Path(__file__).parent.parent / ".onnx_cache_dml"


class ModelManager:
    """Синглтон для управления загрузкой моделей."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance.models = {}  # name -> ModelInfo
        return cls._instance

    def get_model(self, model_name: str, device: str = "cpu") -> ModelInfo:
        """Получить модель по имени (запускает загрузку если нет)."""
        cache_key = (model_name, device)
        with self._lock:
            if cache_key not in self.models:
                self.models[cache_key] = ModelInfo(None, None, model_name)
                threading.Thread(target=self._load_model_worker, args=(model_name, device), daemon=True).start()
            return self.models[cache_key]

    def _load_model_worker(self, model_name: str, device: str):
        """Фоновая загрузка модели."""
        import time
        start_time = time.time()
        cache_key = (model_name, device)
        info = self.models[cache_key]
        try:
            logger.info(f"Начало загрузки модели {model_name} на {device}...")
            
            # Сначала пробуем sentence-transformers (проще и надёжнее)
            try:
                from sentence_transformers import SentenceTransformer
                
                # sentence-transformers автоматически поддерживает разные устройства
                if device == "cpu":
                    model = SentenceTransformer(model_name, device="cpu")
                elif device == "cuda" or device == "gpu":
                    model = SentenceTransformer(model_name, device="cuda")
                else:
                    # Для dml используем cpu через sentence-transformers 
                    # или попробуем ONNX напрямую
                    model = self._load_onnx_model(model_name, device)
                    if model is not None:
                        info.model = model
                        info.loading = False
                        logger.info(f"Модель {model_name} загружена (ONNX)")
                        return
                    # Fallback на CPU
                    model = SentenceTransformer(model_name, device="cpu")
                    logger.warning(f"DirectML недоступен, используется CPU")
                
                info.model = model
                info.loading = False
                logger.info(f"Модель {model_name} загружена за {time.time() - start_time:.2f} сек")
                
            except ImportError:
                # Если sentence-transformers нет, используем ONNX напрямую
                model = self._load_onnx_model(model_name, device)
                if model is None:
                    raise RuntimeError("Не удалось загрузить модель")
                info.model = model
                info.loading = False
                logger.info(f"Модель {model_name} загружена (ONNX) за {time.time() - start_time:.2f} сек")

        except Exception as e:
            info.error = str(e)
            info.loading = False
            logger.error(f"Ошибка загрузки модели {model_name}: {e}", exc_info=True)

    def _load_onnx_model(self, model_name: str, device: str):
        """Загрузка модели через ONNX Runtime напрямую."""
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            import numpy as np
            
            # Определяем провайдеры
            providers = []
            if device == "dml":
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            elif device == "cuda" or device == "gpu":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            
            # Проверяем доступные провайдеры
            available = ort.get_available_providers()
            providers = [p for p in providers if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]
            
            logger.info(f"ONNX провайдеры: {providers}")
            
            # Получаем путь к ONNX модели
            onnx_path = self._get_onnx_model_path(model_name)
            if onnx_path is None:
                return None
            
            # Создаём сессию
            sess_options = ort.SessionOptions()
            # Настройки для стабильности DML
            sess_options.enable_mem_pattern = False
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            session = ort.InferenceSession(str(onnx_path), providers=providers, sess_options=sess_options)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            class ONNXWrapper:
                def __init__(self, session, tokenizer):
                    self.session = session
                    self.tokenizer = tokenizer
                
                def encode(self, sentences, batch_size=32, show_progress_bar=False):
                    all_embeddings = []
                    for i in range(0, len(sentences), batch_size):
                        batch = sentences[i : i + batch_size]
                        
                        # Pad batch to fixed size to avoid ONNX Runtime buffer reuse shape mismatches on DML
                        original_len = len(batch)
                        if original_len < batch_size and batch_size > 0:
                            batch = batch + [batch[-1]] * (batch_size - original_len)
                        
                        encoded = self.tokenizer(
                            batch, 
                            padding='max_length', 
                            truncation=True, 
                            max_length=512, 
                            return_tensors='np'
                        )
                        
                        input_names = [i.name for i in self.session.get_inputs()]
                        
                        inputs = {
                            'input_ids': encoded['input_ids'].astype(np.int64),
                            'attention_mask': encoded['attention_mask'].astype(np.int64),
                        }
                        if 'token_type_ids' in encoded:
                             inputs['token_type_ids'] = encoded['token_type_ids'].astype(np.int64)
                        
                        final_inputs = {k: v for k, v in inputs.items() if k in input_names}
                        
                        try:
                            outputs = self.session.run(None, final_inputs)
                        except Exception as e:
                            # DirectML error handler
                            err_str = str(e)
                            if "0xce" in err_str or "0xcf" in err_str or "utf-8" in err_str:
                                logger.error("DML Native Error (likely encoding issue masking real error). Retrying with CPU...")
                                # Fallback logic not easy here without full reload, just re-raise closer to reality 
                                raise RuntimeError("DirectML Native Crash") from e
                            raise e
                            
                        token_embeddings = outputs[0]
                        
                        if original_len < batch_size:
                            token_embeddings = token_embeddings[:original_len]
                            encoded['attention_mask'] = encoded['attention_mask'][:original_len]

                        attention_mask = encoded['attention_mask']
                        
                        input_mask_expanded = np.expand_dims(attention_mask, -1)
                        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
                        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
                        embeddings = sum_embeddings / sum_mask
                        
                        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                        embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
                        
                        all_embeddings.append(embeddings)
                    
                    if len(all_embeddings) > 0:
                        return np.vstack(all_embeddings)
                    return np.array([])
            
            return ONNXWrapper(session, tokenizer)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки ONNX: {e}", exc_info=True)
            return None

    def _get_onnx_model_path(self, model_name: str) -> Path | None:
        """Получить путь к ONNX модели, экспортировать если нужно."""
        ONNX_CACHE_DIR.mkdir(exist_ok=True)
        
        # Имя папки для модели
        safe_name = model_name.replace("/", "_")
        model_dir = ONNX_CACHE_DIR / safe_name
        onnx_path = model_dir / "model.onnx"
        
        if onnx_path.exists():
            return onnx_path
        
        # Экспортируем модель в ONNX
        try:
            logger.info(f"Экспорт модели {model_name} в ONNX...")
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            model_dir.mkdir(exist_ok=True)
            
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            model.eval()
            
            # Dummy input with batch size 1 for dynamic export
            dummy = tokenizer(["test"], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
            
            # Export
            torch.onnx.export(
                model,
                (dummy['input_ids'], dummy['attention_mask']),
                str(onnx_path),
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state'],
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'sequence'},
                    'attention_mask': {0: 'batch', 1: 'sequence'},
                    'last_hidden_state': {0: 'batch', 1: 'sequence'}
                },
                opset_version=13
            )
            
            logger.info(f"Модель экспортирована в {onnx_path}")
            return onnx_path
            
        except Exception as e:
            logger.error(f"Ошибка экспорта в ONNX: {e}")
            return None

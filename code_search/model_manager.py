"""Менеджер моделей эмбеддингов."""
import threading
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Информация о загруженной модели."""
    model: object
    tokenizer: object
    name: str
    loading: bool = True
    error: str | None = None


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

    def get_model(self, model_name: str) -> ModelInfo:
        """Получить модель по имени (запускает загрузку если нет)."""
        with self._lock:
            if model_name not in self.models:
                self.models[model_name] = ModelInfo(None, None, model_name)
                threading.Thread(target=self._load_model_worker, args=(model_name,), daemon=True).start()
            return self.models[model_name]

    def _load_model_worker(self, model_name: str):
        """Фоновая загрузка модели."""
        info = self.models[model_name]
        try:
            print(f"Загрузка модели {model_name} (ONNX)...")
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
            import torch
            import numpy as np

            # Load/Export to ONNX
            model = ORTModelForFeatureExtraction.from_pretrained(
                model_name, 
                export=True,
                provider="CPUExecutionProvider"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            class ONNXWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                
                def encode(self, sentences, batch_size=32, show_progress_bar=False):
                    all_embeddings = []
                    for i in range(0, len(sentences), batch_size):
                        batch = sentences[i : i + batch_size]
                        encoded_input = self.tokenizer(
                            batch, 
                            padding=True, 
                            truncation=True, 
                            max_length=512, 
                            return_tensors='pt'
                        )
                        outputs = self.model(**encoded_input)
                        token_embeddings = outputs.last_hidden_state
                        attention_mask = encoded_input['attention_mask']
                        
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        embeddings = sum_embeddings / sum_mask
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        all_embeddings.append(embeddings.detach().numpy())
                    
                    if len(all_embeddings) > 0:
                        return np.vstack(all_embeddings)
                    return np.array([])

            info.model = ONNXWrapper(model, tokenizer)
            info.loading = False
            print(f"Модель {model_name} загружена")

        except Exception as e:
            info.error = str(e)
            info.loading = False
            print(f"Ошибка загрузки модели {model_name}: {e}")

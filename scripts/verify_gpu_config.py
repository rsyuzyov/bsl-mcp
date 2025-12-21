
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add repo root to path
# Add repo root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from code_search.model_manager import ModelManager

class TestGPUConfig(unittest.TestCase):
    def setUp(self):
        ModelManager._instance = None
    
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    @patch('transformers.AutoTokenizer')
    def test_cpu_provider(self, mock_tokenizer, mock_ort):
        with patch('threading.Thread') as mock_thread_cls:
            mock_thread_cls.side_effect = lambda target, args, daemon: MagicMock(start=lambda: target(*args))
            
            ModelManager._instance = None
            mm = ModelManager()
            mm.get_model("test-model", device="cpu")
            
            mock_ort.from_pretrained.assert_called_with(
                "test-model", 
                export=True, 
                provider="CPUExecutionProvider"
            )

    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    @patch('transformers.AutoTokenizer')
    def test_gpu_provider(self, mock_tokenizer, mock_ort):
        ModelManager._instance = None
        with patch('threading.Thread') as mock_thread_cls:
            mock_thread_cls.side_effect = lambda target, args, daemon: MagicMock(start=lambda: target(*args))
            
            mm = ModelManager()
            mm.get_model("test-model", device="gpu")
            
            mock_ort.from_pretrained.assert_called_with(
                "test-model", 
                export=True, 
                provider="CUDAExecutionProvider"
            )

    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    @patch('transformers.AutoTokenizer')
    def test_dml_provider(self, mock_tokenizer, mock_ort):
        ModelManager._instance = None
        with patch('threading.Thread') as mock_thread_cls:
            mock_thread_cls.side_effect = lambda target, args, daemon: MagicMock(start=lambda: target(*args))
            
            mm = ModelManager()
            mm.get_model("test-model", device="dml")
            
            mock_ort.from_pretrained.assert_called_with(
                "test-model", 
                export=True, 
                provider="DmlExecutionProvider"
            )

if __name__ == '__main__':
    unittest.main()

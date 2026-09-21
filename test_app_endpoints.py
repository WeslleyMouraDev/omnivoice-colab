import sys
import unittest
from unittest.mock import MagicMock

if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()

# Mock de gradio caso não esteja instalado no ambiente local
if "gradio" not in sys.modules:
    mock_gr = MagicMock()
    mock_gr.themes.Soft = MagicMock()
    
    # Simular Blocks e registro de fns
    class MockBlocks:
        def __init__(self, *args, **kwargs):
            self.fns = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def queue(self, *args, **kwargs):
            return self
        def launch(self, *args, **kwargs):
            return self

    class MockButton:
        def __init__(self, *args, **kwargs):
            pass
        def click(self, fn=None, inputs=None, outputs=None, api_name=None, js=None):
            if api_name:
                mock_fn = MagicMock()
                mock_fn.api_name = api_name
                mock_fn.fn = fn
                mock_blocks_inst.fns[api_name] = mock_fn
            return self

    mock_blocks_inst = MockBlocks()
    mock_gr.Blocks = MagicMock(return_value=mock_blocks_inst)
    mock_gr.Button = MockButton
    sys.modules["gradio"] = mock_gr

import app

class TestAppEndpoints(unittest.TestCase):

    def test_endpoints_registered(self):
        registered_api_names = set(app.demo.fns.keys())
        print("Endpoints registrados no Gradio:", registered_api_names)
        self.assertIn("prepare_voice", registered_api_names)
        self.assertIn("generate_tts", registered_api_names)
        self.assertIn("clone_and_generate", registered_api_names)

    def test_api_handlers_exist(self):
        self.assertTrue(callable(app.api_prepare_voice_fn))
        self.assertTrue(callable(app.api_generate_tts_fn))
        self.assertTrue(callable(app.api_clone_and_generate_fn))

if __name__ == "__main__":
    unittest.main()

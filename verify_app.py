import sys
import os
from pathlib import Path

# Add current dir to path
sys.path.append(os.getcwd())

try:
    from code_search.web.app import create_app
    from code_search.app_context import IBManager
    
    print("Importing successful.")
    
    # Mock IBManager
    class MockIBManager:
        def get_all_contexts(self):
            return []
        def get_context(self, name):
            return None
            
    mgr = MockIBManager()
    print("Creating app...")
    app = create_app(mgr)
    print("App created successfully.")
    
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

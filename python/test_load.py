import speed_token
import os

try:
    path = os.path.abspath("data/cl100k_base.tiktoken")
    print(f"Loading vocab from {path}")
    if not os.path.exists(path):
        print("Path doesn't exist!")
    else:
        ft = speed_token.SpeedToken(path, "cl100k_base")
        print("Loaded successfully")
        tokens = ft.encode("hello world")
        print(f"Encoded: {tokens}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

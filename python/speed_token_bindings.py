import ctypes
import os
import sys

class SpeedToken:
    def __init__(self, dll_path=None):
        if dll_path is None:
            dll_name = "libspeed_token.dll" if sys.platform == "win32" else "libspeed_token.so"
            # Try to find in the package directory or current directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(base_dir, "..", dll_name)
            if not os.path.exists(dll_path):
                dll_path = os.path.join(os.getcwd(), dll_name)
        
        dll_path = os.path.abspath(dll_path)
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DLL not found at {dll_path}")

        # On Windows, add the DLL directory to search path for dependencies
        if sys.platform == "win32" and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(os.path.dirname(dll_path))

        try:
            self.lib = ctypes.CDLL(dll_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load DLL from {dll_path}: {e}")

        # Tokenizer* tokenizer_load(const char *path, const char *pattern)
        self.lib.tokenizer_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.tokenizer_load.restype = ctypes.c_void_p

        # void tokenizer_free(Tokenizer *t)
        self.lib.tokenizer_free.argtypes = [ctypes.c_void_p]
        self.lib.tokenizer_free.restype = None

        # size_t tokenizer_encode(Tokenizer *t, const char *text, int32_t *out_tokens, size_t max_tokens)
        self.lib.tokenizer_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
        self.lib.tokenizer_encode.restype = ctypes.c_size_t

        # size_t tokenizer_decode(Tokenizer *t, const int32_t *tokens, size_t num_tokens, char *out_text, size_t max_text_len)
        self.lib.tokenizer_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.tokenizer_decode.restype = ctypes.c_size_t

        # Profiling
        class TokenizerProfile(ctypes.Structure):
            _fields_ = [
                ("pretokenize_cycles", ctypes.c_uint64),
                ("merge_cycles", ctypes.c_uint64),
                ("hash_lookup_cycles", ctypes.c_uint64),
                ("total_cycles", ctypes.c_uint64),
            ]
        self.TokenizerProfile = TokenizerProfile

        self.lib.tokenizer_reset_profile.argtypes = [ctypes.c_void_p]
        self.lib.tokenizer_reset_profile.restype = None

        self.lib.tokenizer_get_profile.argtypes = [ctypes.c_void_p]
        self.lib.tokenizer_get_profile.restype = TokenizerProfile

        self.tokenizer = None

    def load(self, path, pattern):
        self.tokenizer = self.lib.tokenizer_load(path.encode('utf-8'), pattern.encode('utf-8'))
        if not self.tokenizer:
            raise Exception(f"Failed to load tokenizer from {path}")

    def encode(self, text):
        if not self.tokenizer:
            raise Exception("Tokenizer not loaded")
        
        # Manually replace lone surrogates to match tiktoken's behavior exactly
        # and avoid environment-specific replacement characters in encode('utf-8', 'replace')
        text_replaced = "".join([c if not (0xD800 <= ord(c) <= 0xDFFF) else "\ufffd" for c in text])
        encoded_text = text_replaced.encode('utf-8')
        
        # Upper bound for tokens: each byte could be a token
        max_tokens = len(encoded_text) + 100 
        out_tokens = (ctypes.c_int32 * max_tokens)()
        
        count = self.lib.tokenizer_encode(self.tokenizer, encoded_text, out_tokens, max_tokens)
        if count == (2**64 - 1 if sys.maxsize > 2**32 else 2**32 - 1): # (size_t)-1
            raise Exception("Encoding failed")
            
        return [out_tokens[i] for i in range(count)]

    def decode(self, tokens):
        if not self.tokenizer:
            raise Exception("Tokenizer not loaded")
        
        # Rough upper bound for decoded text
        max_text_len = len(tokens) * 64 + 1024 
        out_text = ctypes.create_string_buffer(max_text_len)
        
        int_array = (ctypes.c_int32 * len(tokens))(*tokens)
        
        length = self.lib.tokenizer_decode(self.tokenizer, int_array, len(tokens), out_text, max_text_len)
        if length == (2**64 - 1 if sys.maxsize > 2**32 else 2**32 - 1): # (size_t)-1
            raise Exception("Decoding failed")
            
        return out_text.value.decode('utf-8', errors='replace')

    def reset_profile(self):
        if self.tokenizer:
            self.lib.tokenizer_reset_profile(self.tokenizer)

    def get_profile(self):
        if self.tokenizer:
            p = self.lib.tokenizer_get_profile(self.tokenizer)
            return {
                "pretokenize_cycles": p.pretokenize_cycles,
                "merge_cycles": p.merge_cycles,
                "hash_lookup_cycles": p.hash_lookup_cycles,
                "total_cycles": p.total_cycles,
            }
        return None

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'tokenizer') and self.tokenizer:
            self.lib.tokenizer_free(self.tokenizer)

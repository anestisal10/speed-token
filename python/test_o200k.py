import tiktoken
import speed_token
import os

# 2. Initialize our tokenizer for o200k
ft = speed_token.SpeedToken("data/o200k_base.tiktoken", "o200k_base")
enc = tiktoken.get_encoding("o200k_base")

test_strings = [
    "Hello world", "Python is great.", "Complex Unicode: 😊 🌍 🚀",
    "0.123456789", "  multiple   spaces  "
]

passed = 0
for s in test_strings:
    exp = enc.encode(s)
    act = ft.encode(s)
    if exp == act:
        print(f"MATCHED: {repr(s)}")
        passed += 1
    else:
        print(f"MISMATCH: {repr(s)}")
        print(f"  Exp: {exp}")
        print(f"  Act: {act}")

print(f"Passed {passed}/{len(test_strings)} o200k_base tests.")

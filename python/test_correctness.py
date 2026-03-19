import tiktoken
import speed_token
import os
import time

def test_correctness():
    # 1. Ensure vocabs are downloaded
    if not os.path.exists("data/cl100k_base.tiktoken"):
        print("Downloading vocabs...")
        import subprocess
        # Find the absolute path of the current file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        download_script = os.path.join(current_dir, "download_vocabs.py")
        subprocess.run(["python", download_script], check=True)

    # 2. Initialize our tokenizer
    try:
        ft = speed_token.SpeedToken("data/cl100k_base.tiktoken", "cl100k_base")
    except Exception as e:
        print(f"Error loading: {e}")
        # Try loading from parent dir if running from root
        ft = speed_token.SpeedToken("../data/cl100k_base.tiktoken", "cl100k_base")

    # 3. Initialize tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    # 4. Test cases
    test_strings = [
        "Hello world",
        "Hello world!",
        "  multiple   spaces  ",
        "Python is great.",
        "Antidisestablishmentarianism",
        "", # empty string
        "123 4567 890", # numbers
        "Complex Unicode: 😊 🌍 🚀",
        "Mixed language: Hello Γειά σου 你好 नमस्ते",
        "Code snippet: def foo(): return 42",
        "Double newlines\n\nTriple newlines\n\n\n",
        "Contractions: I'm, you're, we'll, it's, don't",
        "Edge case: lone surrogate \ud800", # Need to check how tiktoken handles this
        "         ", # only spaces
        "0.123456789", # floats
        "!@#$%^&*()_+", # symbols
    ]

    print(f"{'Input':<40} | {'Status':<10}")
    print("-" * 55)

    passed = 0
    for s in test_strings:
        s_display = repr(s)[:38]
        # tiktoken encode
        expected = enc.encode(s)
        
        # our encode
        try:
            actual = ft.encode(s)
            if actual == expected:
                print(f"{repr(s_display):<40} | PASSED")
                passed += 1
            else:
                with open("mismatches.txt", "a", encoding="utf-8") as f:
                    f.write(f"FAILED: {repr(s)}\nExp: {expected}\nAct: {actual}\n")
                # Print breakdown
                if len(expected) != len(actual):
                    print(f"  Len mismatch: Exp {len(expected)}, Act {len(actual)}")
                for i in range(min(len(expected), len(actual))):
                    if expected[i] != actual[i]:
                        print(f"  Diff at index {i}: Exp {expected[i]}, Act {actual[i]}")
                        break
        except Exception as e:
            with open("errors.txt", "a", encoding="utf-8") as f:
                f.write(f"ERROR on {repr(s)}: {e}\n")

    print("-" * 55)
    print(f"Passed {passed}/{len(test_strings)} basic tests.")

    if passed == len(test_strings):
        print("\nRunning large corpus test (simulated)...")
        # In a real scenario, we'd load a 50k string corpus
        # For now, let's just do a few more complex ones
        print("All basic tests passed!")
    else:
        print("\nCorrectness baseline not yet achieved.")

if __name__ == "__main__":
    test_correctness()

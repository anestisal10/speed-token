import time
import os
import numpy as np
import tiktoken
from tokenizers import Tokenizer as HFTokenizer
from speed_token import SpeedToken

try:
    import rs_bpe
except ImportError:
    rs_bpe = None

def load_corpora():
    corpora = {}
    for name in ["wiki", "stack", "data", "adversarial"]:
        ext = "py" if name == "stack" else "json" if name == "data" else "txt"
        path = f"data/{name}.{ext}"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                corpora[name] = f.read()
    return corpora

def benchmark_throughput(name, text, ft, tt, hf, rs):
    results = {}
    size_mb = len(text.encode('utf-8')) / (1024 * 1024)
    
    # SpeedToken
    start = time.perf_counter()
    tokens = ft.encode(text)
    end = time.perf_counter()
    results["SpeedToken (ST)"] = {
        "throughput_mb": size_mb / (end - start),
        "tokens_sec": len(tokens) / (end - start)
    }

    chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]
    start = time.perf_counter()
    batch_tokens = ft.encode_batch(chunks)
    end = time.perf_counter()
    num_tokens = sum(len(t) for t in batch_tokens)
    results["SpeedToken (MT)"] = {
        "throughput_mb": size_mb / (end - start),
        "tokens_sec": num_tokens / (end - start)
    }

    # Tiktoken
    start = time.perf_counter()
    tokens = tt.encode(text)
    end = time.perf_counter()
    results["Tiktoken"] = {
        "throughput_mb": size_mb / (end - start),
        "tokens_sec": len(tokens) / (end - start)
    }

    # rs-bpe
    if rs:
        start = time.perf_counter()
        tokens = rs.encode(text)
        end = time.perf_counter()
        results["rs-bpe (ST)"] = {
            "throughput_mb": size_mb / (end - start),
            "tokens_sec": len(tokens) / (end - start)
        }

        start = time.perf_counter()
        batch_res = rs.encode_batch(chunks)
        batch_tokens, num_tokens, _ = batch_res
        end = time.perf_counter()
        results["rs-bpe (MT)"] = {
            "throughput_mb": size_mb / (end - start),
            "tokens_sec": num_tokens / (end - start)
        }

    # HF Tokenizers
    if hf:
        start = time.perf_counter()
        tokens = hf.encode(text).ids
        end = time.perf_counter()
        results["HF Tokenizers"] = {
            "throughput_mb": size_mb / (end - start),
            "tokens_sec": len(tokens) / (end - start)
        }

    return results

def benchmark_latency(ft, tt):
    sizes = [1024, 10240, 1024 * 1024] # 1KB, 10KB, 1MB
    text_sample = "Hello world! This is a test. " * 50000
    
    latency_results = {}
    for size in sizes:
        sub_text = text_sample[:size]
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            ft.encode(sub_text)
            latencies.append((time.perf_counter() - start) * 1000) # ms
        
        latency_results[f"{size//1024}KB"] = {
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }
    return latency_results

def run_all():
    vocab_path = "data/cl100k_base.tiktoken"
    ft = SpeedToken(vocab_path, "cl100k_base")
    tt = tiktoken.get_encoding("cl100k_base")
    rs = rs_bpe.openai.cl100k_base() if rs_bpe else None
    
    try:
        hf = HFTokenizer.from_pretrained("gpt2")
    except:
        hf = None

    corpora = load_corpora()
    
    print("# Benchmark Results")
    for name, text in corpora.items():
        print(f"\n## Corpus: {name.capitalize()}")
        res = benchmark_throughput(name, text, ft, tt, hf, rs)
        print("| Method | Throughput (MB/s) | Tokens/sec |")
        print("| --- | --- | --- |")
        for method, metrics in res.items():
            print(f"| {method} | {metrics['throughput_mb']:.2f} | {metrics['tokens_sec']:.0f} |")

    print("\n## Latency (SpeedToken)")
    lat = benchmark_latency(ft, tt)
    print("| size | p50 (ms) | p95 (ms) | p99 (ms) |")
    print("| --- | --- | --- | --- |")
    for size, metrics in lat.items():
        print(f"| {size} | {metrics['p50']:.3f} | {metrics['p95']:.3f} | {metrics['p99']:.3f} |")

if __name__ == "__main__":
    run_all()

import tiktoken
import time
import os
import speed_token

CORPUS_PATH = "data/corpus_1gb.txt"

def benchmark_tiktoken(corpus):
    enc = tiktoken.get_encoding("cl100k_base")
    # Warmup
    enc.encode("hello world")
    
    start = time.perf_counter()
    tokens = enc.encode(corpus)
    end = time.perf_counter()
    
    duration = end - start
    size_mb = len(corpus.encode('utf-8')) / (1024 * 1024)
    throughput = size_mb / duration
    tps = len(tokens) / duration
    
    return duration, throughput, tps, len(tokens)

def benchmark_speed_token(ft, corpus):
    # Warmup
    ft.encode("hello world")
    ft.reset_profile()
    
    start_time = time.perf_counter()
    tokens = ft.encode(corpus)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    profile = ft.get_profile()
    
    size_mb = len(corpus.encode('utf-8')) / (1024 * 1024)
    throughput = size_mb / duration
    tps = len(tokens) / duration
    
    return duration, throughput, tps, len(tokens), profile

def main():
    if not os.path.exists(CORPUS_PATH):
        print(f"Corpus not found at {CORPUS_PATH}. Please run create_corpus.py first.")
        return

    print(f"Loading corpus from {CORPUS_PATH} (this might take a moment)...")
    with open(CORPUS_PATH, "rb") as f:
        corpus_bytes = f.read()
    corpus = corpus_bytes.decode('utf-8', errors='ignore')
    
    actual_size_mb = len(corpus_bytes) / (1024 * 1024)
    
    print(f"\nBenchmarking on full {actual_size_mb:.2f} MB corpus...")
    
    print("Tiktoken:")
    t_dur, t_thru, t_tps, t_cnt = benchmark_tiktoken(corpus)
    print(f"  Duration: {t_dur:.4f}s")
    print(f"  Throughput: {t_thru:.2f} MB/s")
    print(f"  Tokens/sec: {t_tps:.2f}")
    
    print("\nSpeedToken (C Baseline):")
    try:
        ft = speed_token.SpeedToken("data/cl100k_base.tiktoken", "cl100k_base")
    except Exception as e:
        print(f"  Failed to load SpeedToken: {e}")
        return

    f_dur, f_thru, f_tps, f_cnt, f_prof = benchmark_speed_token(ft, corpus)
    print(f"  Duration: {f_dur:.4f}s")
    print(f"  Throughput: {f_thru:.2f} MB/s")
    print(f"  Tokens/sec: {f_tps:.2f}")
    
    if t_cnt != f_cnt:
        print(f"  WARNING: Token count mismatch! Tiktoken: {t_cnt}, SpeedToken: {f_cnt}")
    else:
        print("  Token counts match exactly.")

    print(f"\n--- Profiling Breakdown (SpeedToken) ---")
    total_c = f_prof["total_cycles"]
    pre_c = f_prof["pretokenize_cycles"]
    merge_c = f_prof["merge_cycles"]
    hash_c = f_prof["hash_lookup_cycles"]
    
    report = []
    report.append(f"Benchmarking on {actual_size_mb:.2f} MB slice")
    report.append("-" * 40)
    report.append(f"Tiktoken: {t_thru:.2f} MB/s, {t_tps:.2f} tokens/sec")
    report.append(f"SpeedToken: {f_thru:.2f} MB/s, {f_tps:.2f} tokens/sec")
    report.append("-" * 40)
    report.append(f"Total C Cycles: {total_c:,}")
    report.append(f"  (a) Pretokenization:   {pre_c:>15,} ({pre_c/total_c*100:5.1f}%)")
    report.append(f"  (b) BPE Merge (Total): {merge_c:>15,} ({merge_c/total_c*100:5.1f}%)")
    report.append(f"      - Hash Lookups:    {hash_c:>15,} ({hash_c/total_c*100:5.1f}%)")
    
    small_start = time.perf_counter()
    for _ in range(1000):
        ft.encode("hi")
    small_end = time.perf_counter()
    per_call_overhead_us = (small_end - small_start) * 1e6 / 1000
    report.append(f"Avg Python-C overhead: {per_call_overhead_us:.2f} us")
    
    for line in report:
        print(line)
        
    with open("benchmark_report.txt", "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(report))
    print("\nFull report saved to benchmark_report.txt")

if __name__ == "__main__":
    main()

# SpeedToken 🚀

**The world's fastest BPE tokenizer.** 3–15× faster than `tiktoken` via AVX2 SIMD pretokenization and O(n) doubly-linked BPE merges.

## 📊 Performance Benchmarks

*Measured on Windows (x86_64, AVX2) using cl100k_base. MT denotes Rayon-parallel batch encoding.*

<!-- BENCHMARK_TABLE_START -->
| Corpus | Method | Throughput (MB/s) | Tokens/sec |
| :--- | :--- | :--- | :--- |
| **Wikipedia** | **SpeedToken (MT)** | **38.19** | **12,803,354** |
| | SpeedToken (ST) | 13.94 | 4,673,539 |
| | rs-bpe (MT) | 14.07 | 4,718,085 |
| | rs-bpe (ST) | 13.17 | 4,413,433 |
| | Tiktoken | 4.85 | 1,627,100 |
| **JSON** | **SpeedToken (MT)** | **30.31** | **21,230,938** |
| | SpeedToken (ST) | 14.72 | 10,309,400 |
| | rs-bpe (ST) | 7.03 | 4,921,177 |
| | Tiktoken | 2.09 | 1,461,528 |
| **Code** | **SpeedToken (MT)** | **29.21** | **9,791,058** |
| | SpeedToken (ST) | 12.85 | 4,307,746 |
| | Tiktoken | 4.12 | 1,380,046 |
<!-- BENCHMARK_TABLE_END -->

## 🛠️ How It Works

### 1. SIMD Pretokenization
Uses AVX2 `VPSHUFB` instructions for ultra-fast parallel byte classification. Instead of slow regex matching, we classify 32 bytes at a time into categories (letters, digits, whitespace, etc.) to identify token boundaries.

```mermaid
graph LR
    A[Input Bytes] --> B{AVX2 VPSHUFB}
    B --> C[Letters Mask]
    B --> D[Digits Mask]
    B --> E[Whitespace Mask]
    C & D & E --> F[Token Boundaries]
```

### 2. O(n) Doubly-Linked BPE
Replaces the standard $O(n^2)$ merge loop with a doubly-linked list and a min-priority queue of merge candidates. This ensures each merge operation is $O(1)$ and the total complexity is $O(n)$ relative to the number of input bytes.

```mermaid
sequenceDiagram
    participant Q as Priority Queue (Ranks)
    participant L as Doubly-Linked List (Tokens)
    L->>Q: Push initial pairs
    loop until Q empty
        Q->>L: Pop highest rank pair (A, B)
        L->>L: Merge (A, B) -> C
        L->>Q: Push new neighbors (prev, C), (C, next)
    end
```

### 3. Rayon Parallel Batching
Leverages Rust's Rayon library to parallelize encoding across multiple CPU cores. Each core maintains its own thread-local scratch arena and pair-cache, enabling massive scaling without lock contention.

## 🚀 Quick Start

### Installation

```bash
pip install speed_token
```

### Usage

```python
from speed_token import SpeedToken

# Load cl100k_base (GPT-4) vocabulary
ft = SpeedToken("path/to/cl100k_base.tiktoken", "cl100k_base")

# Encode a single string
tokens = ft.encode("Hello world! 🚀")
print(tokens)

# Decode tokens back to bytes
bytes_out = ft.decode(tokens)
print(bytes_out.decode('utf-8'))

# Batch encode with Rayon parallelism
texts = ["First sentence", "Second sentence", "Third sentence"]
batch_tokens = ft.encode_batch(texts)
```

## 📜 License
MIT

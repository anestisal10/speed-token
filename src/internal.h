#ifndef INTERNAL_H
#define INTERNAL_H

#include "tokenizer.h"
#include "hashtable.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef _MSC_VER
#include <intrin.h>
#define RDTSC() __rdtsc()
#define THREAD_LOCAL __declspec(thread)
#else
#include <x86intrin.h>
#define RDTSC() __rdtsc()
#define THREAD_LOCAL _Thread_local
#endif

typedef struct {
    unsigned char *key;
    size_t len;
    int32_t rank;
} VocabEntry;

typedef struct {
    VocabEntry *entries;
    size_t capacity;
    size_t count;
} HashVocab;

// Doubly-linked list node for BPE
typedef struct BNode {
    int32_t rank;
    struct BNode *prev;
    struct BNode *next;
} BNode;

// Arena for BNode allocation
typedef struct {
    BNode *nodes;
    size_t capacity;
    size_t count;
} NodeArena;

// Entry in the priority queue
typedef struct {
    int32_t rank;
    BNode *left;
} HeapEntry;

typedef struct {
    HeapEntry *entries;
    size_t capacity;
    size_t size;
} MergeHeap;

struct Tokenizer {
    HashVocab encoder;
    PairHashMap pair_encoder; // (left_token, right_token) -> rank
    unsigned char **decoder_tokens;
    size_t *decoder_lens;
    size_t vocab_size;
    char pattern[32];
    TokenizerProfile profile;
    
    // Thread-local or per-tokenizer scratch space
    NodeArena arena;
    MergeHeap heap;
};

uint32_t hash_bytes(const unsigned char *bytes, size_t len);
void hash_vocab_insert(HashVocab *v, const unsigned char *bytes, size_t len, int32_t rank);
int32_t hash_vocab_get(HashVocab *v, const unsigned char *bytes, size_t len);

#endif // INTERNAL_H

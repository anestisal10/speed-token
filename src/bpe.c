#include "tokenizer.h"
#include "internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    for (size_t i = 0; i < t->encoder.capacity; i++) {
        if (t->encoder.entries[i].key) free(t->encoder.entries[i].key);
    }
    free(t->encoder.entries);
    pair_hashmap_free(&t->pair_encoder);
    for (size_t i = 0; i < t->vocab_size; i++) {
        if (t->decoder_tokens[i]) free(t->decoder_tokens[i]);
    }
    free(t->decoder_tokens);
    free(t->decoder_lens);
    free(t);
}

void tokenizer_reset_profile(Tokenizer *t) {
    if (t) memset(&t->profile, 0, sizeof(TokenizerProfile));
}

TokenizerProfile tokenizer_get_profile(Tokenizer *t) {
    if (t) return t->profile;
    TokenizerProfile empty = {0};
    return empty;
}

size_t tokenizer_decode(Tokenizer *t, const int32_t *tokens, size_t num_tokens, char *out_text, size_t max_text_len) {
    size_t pos = 0;
    for (size_t i = 0; i < num_tokens; i++) {
        int32_t rank = tokens[i];
        if (rank < 0 || (size_t)rank >= t->vocab_size) continue;
        size_t len = t->decoder_lens[rank];
        if (pos + len >= max_text_len) break;
        memcpy(out_text + pos, t->decoder_tokens[rank], len);
        pos += len;
    }
    if (pos < max_text_len) out_text[pos] = '\0';
    return pos;
}

// Heap operations
static void heap_push(MergeHeap *h, int32_t rank, BNode *left) {
    if (h->size == h->capacity) {
        h->capacity = h->capacity == 0 ? 1024 : h->capacity * 2;
        h->entries = realloc(h->entries, h->capacity * sizeof(HeapEntry));
    }
    size_t i = h->size++;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (h->entries[p].rank <= rank) break;
        h->entries[i] = h->entries[p];
        i = p;
    }
    h->entries[i].rank = rank;
    h->entries[i].left = left;
}

static bool heap_pop(MergeHeap *h, int32_t *rank, BNode **left) {
    if (h->size == 0) return false;
    *rank = h->entries[0].rank;
    *left = h->entries[0].left;
    
    HeapEntry last = h->entries[--h->size];
    if (h->size == 0) return true;

    size_t i = 0;
    while (i * 2 + 1 < h->size) {
        size_t child = i * 2 + 1;
        if (child + 1 < h->size && h->entries[child + 1].rank < h->entries[child].rank) {
            child++;
        }
        if (last.rank <= h->entries[child].rank) break;
        h->entries[i] = h->entries[child];
        i = child;
    }
    h->entries[i] = last;
    return true;
}

static THREAD_LOCAL PairHashMap tl_cache = {NULL, 0, 0};

static inline int32_t get_pair_rank(Tokenizer *t, BNode *left) {
    if (!left || !left->next) return -1;
    uint64_t key = ((uint64_t)left->rank << 32) | (uint32_t)left->next->rank;
    
    if (tl_cache.capacity == 0) {
        pair_hashmap_init(&tl_cache, 4096);
    }

    int32_t r = pair_hashmap_get(&tl_cache, key);
    if (r != -1) return r;

    // Not in cache, try looking up concatenation
    unsigned char pair_bytes[1024];
    size_t l1 = t->decoder_lens[left->rank];
    size_t l2 = t->decoder_lens[left->next->rank];
    if (l1 + l2 > 1024) return -1;

    memcpy(pair_bytes, t->decoder_tokens[left->rank], l1);
    memcpy(pair_bytes + l1, t->decoder_tokens[left->next->rank], l2);

    r = hash_vocab_get(&t->encoder, pair_bytes, l1 + l2);

    if (r != -1) {
        pair_hashmap_insert(&tl_cache, key, r);
    }
    return r;
}

static THREAD_LOCAL NodeArena tl_arena = {NULL, 0, 0};
static THREAD_LOCAL MergeHeap tl_heap = {NULL, 0, 0};

void tokenizer_thread_free(void) {
    if (tl_arena.nodes) {
        free(tl_arena.nodes);
        tl_arena.nodes = NULL;
        tl_arena.capacity = 0;
        tl_arena.count = 0;
    }
    if (tl_heap.entries) {
        free(tl_heap.entries);
        tl_heap.entries = NULL;
        tl_heap.capacity = 0;
        tl_heap.size = 0;
    }
    pair_hashmap_free(&tl_cache);
}

size_t bpe_encode_chunk(Tokenizer *t, const unsigned char *chunk, size_t len, int32_t *out) {
    uint64_t start_total = RDTSC();
    if (len == 0) return 0;
    if (len == 1) {
        out[0] = hash_vocab_get(&t->encoder, chunk, 1);
        // t->profile.total_cycles += RDTSC() - start_total;
        return 1;
    }

    // Prepare arena
    if (tl_arena.capacity < len) {
        tl_arena.capacity = len * 2;
        tl_arena.nodes = realloc(tl_arena.nodes, tl_arena.capacity * sizeof(BNode));
    }
    tl_arena.count = len;
    
    // Reset heap
    tl_heap.size = 0;

    // Initial sequence
    BNode *nodes = tl_arena.nodes;
    for (size_t i = 0; i < len; i++) {
        unsigned char b = chunk[i];
        nodes[i].rank = hash_vocab_get(&t->encoder, &b, 1);
        nodes[i].prev = (i > 0) ? &nodes[i - 1] : NULL;
        nodes[i].next = (i < len - 1) ? &nodes[i + 1] : NULL;
    }

    // Seed heap
    for (size_t i = 0; i < len - 1; i++) {
        int32_t r = get_pair_rank(t, &nodes[i]);
        if (r != -1) {
            heap_push(&tl_heap, r, &nodes[i]);
        }
    }

    BNode *head = &nodes[0];
    // uint64_t start_merge = RDTSC();

    while (tl_heap.size > 0) {
        int32_t r;
        BNode *left;
        if (!heap_pop(&tl_heap, &r, &left)) break;

        // Validation: is this pair still valid?
        if (!left->next) continue;
        int32_t current_r = get_pair_rank(t, left);
        if (current_r != r) continue;

        // Perform merge: left + left->next -> new node (re-use left)
        BNode *to_remove = left->next;
        left->rank = r;
        left->next = to_remove->next;
        if (to_remove->next) to_remove->next->prev = left;
        
        // After merge, look at new potential pairs: (left->prev, left) and (left, left->next)
        if (left->prev) {
            int32_t r_prev = get_pair_rank(t, left->prev);
            if (r_prev != -1) heap_push(&tl_heap, r_prev, left->prev);
        }
        if (left->next) {
            int32_t r_next = get_pair_rank(t, left);
            if (r_next != -1) heap_push(&tl_heap, r_next, left);
        }
    }
    // t->profile.merge_cycles += RDTSC() - start_merge;

    size_t count = 0;
    for (BNode *curr = head; curr; curr = curr->next) {
        out[count++] = curr->rank;
    }

    // t->profile.total_cycles += RDTSC() - start_total;
    return count;
}

// Pretokenizer will call this for each chunk
size_t tokenizer_encode(Tokenizer *t, const char *text, int32_t *out_tokens, size_t max_tokens) {
    // The REAL implementation will use pretokenize.c
    extern size_t pretokenize_and_encode(Tokenizer *t, const char *text, int32_t *out_tokens, size_t max_tokens);
    return pretokenize_and_encode(t, text, out_tokens, max_tokens);
}

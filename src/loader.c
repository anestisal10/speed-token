#include "tokenizer.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char base64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Simple base64 decode for tiktoken files
int base64_decode(const char *in, unsigned char *out, size_t *out_len) {
    int table[256];
    memset(table, -1, sizeof(table));
    for (int i = 0; i < 64; i++) table[(unsigned char)base64_table[i]] = i;

    size_t in_len = strlen(in);
    if (in_len % 4 != 0) return -1;

    size_t count = 0;
    for (size_t i = 0; i < in_len; i += 4) {
        int v0 = table[(unsigned char)in[i]];
        int v1 = table[(unsigned char)in[i+1]];
        int v2 = table[(unsigned char)in[i+2]];
        int v3 = table[(unsigned char)in[i+3]];

        if (v0 < 0 || v1 < 0) return -1;
        
        out[count++] = (v0 << 2) | (v1 >> 4);
        if (in[i+2] != '=') {
            if (v2 < 0) return -1;
            out[count++] = ((v1 & 0xf) << 4) | (v2 >> 2);
        }
        if (in[i+3] != '=') {
            if (v3 < 0) return -1;
            out[count++] = ((v2 & 0x3) << 6) | v3;
        }
    }
    *out_len = count;
    return 0;
}

uint32_t hash_bytes(const unsigned char *bytes, size_t len) {
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= 16777619u;
    }
    return hash;
}

void hash_vocab_insert(HashVocab *v, const unsigned char *bytes, size_t len, int32_t rank) {
    if (v->count * 2 >= v->capacity) {
        size_t old_cap = v->capacity;
        VocabEntry *old_entries = v->entries;
        v->capacity = old_cap == 0 ? 256 : old_cap * 2;
        v->entries = calloc(v->capacity, sizeof(VocabEntry));
        v->count = 0;
        for (size_t i = 0; i < old_cap; i++) {
            if (old_entries[i].key) {
                hash_vocab_insert(v, old_entries[i].key, old_entries[i].len, old_entries[i].rank);
                free(old_entries[i].key);
            }
        }
        free(old_entries);
    }
    uint32_t h = hash_bytes(bytes, len) % v->capacity;
    while (v->entries[h].key) {
        h = (h + 1) % v->capacity;
    }
    v->entries[h].key = malloc(len);
    memcpy(v->entries[h].key, bytes, len);
    v->entries[h].len = len;
    v->entries[h].rank = rank;
    v->count++;
}

int32_t hash_vocab_get(HashVocab *v, const unsigned char *bytes, size_t len) {
    if (v->capacity == 0) return -1;
    uint32_t h = hash_bytes(bytes, len) % v->capacity;
    while (v->entries[h].key) {
        if (v->entries[h].len == len && memcmp(v->entries[h].key, bytes, len) == 0) {
            return v->entries[h].rank;
        }
        h = (h + 1) % v->capacity;
    }
    return -1;
}

// Minimal BPE encode for short strings (vocab entries) to find parents
static int find_bpe_parents(Tokenizer *t, const unsigned char *bytes, size_t len, int32_t *p1, int32_t *p2) {
    if (len <= 1) return 0;
    
    // We use a simple array-based list for short vocab strings
    int32_t tokens[256];
    size_t n = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char b = bytes[i];
        tokens[n++] = hash_vocab_get(&t->encoder, &b, 1);
    }
    
    while (n > 2) {
        int best_idx = -1;
        int32_t best_rank = 0x7FFFFFFF;
        
        for (size_t i = 0; i < n - 1; i++) {
            // Find rank of pair (tokens[i], tokens[i+1])
            // We need to concat their bytes and look up in t->encoder
            unsigned char pair_buf[512];
            size_t l1 = t->decoder_lens[tokens[i]];
            size_t l2 = t->decoder_lens[tokens[i+1]];
            if (l1 + l2 > 512) continue;
            memcpy(pair_buf, t->decoder_tokens[tokens[i]], l1);
            memcpy(pair_buf + l1, t->decoder_tokens[tokens[i+1]], l2);
            
            int32_t r = hash_vocab_get(&t->encoder, pair_buf, l1 + l2);
            if (r != -1 && r < best_rank) {
                best_rank = r;
                best_idx = i;
            }
        }
        
        if (best_idx == -1) break;
        
        // Merge best_idx and best_idx + 1
        tokens[best_idx] = best_rank;
        for (size_t i = best_idx + 1; i < n - 1; i++) {
            tokens[i] = tokens[i+1];
        }
        n--;
    }
    
    if (n == 2) {
        *p1 = tokens[0];
        *p2 = tokens[1];
        return 1;
    }
    return 0;
}

Tokenizer* tokenizer_load(const char *path, const char *pattern) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;

    Tokenizer *t = calloc(1, sizeof(Tokenizer));
    strncpy(t->pattern, pattern, sizeof(t->pattern) - 1);
    
    // Pre-allocate space for decoder
    t->decoder_tokens = calloc(256000, sizeof(unsigned char*));
    t->decoder_lens = calloc(256000, sizeof(size_t));

    pair_hashmap_init(&t->pair_encoder, 256000);

    char b64[4096];
    int32_t rank;
    unsigned char decoded[4096];
    size_t dlen;

    while (fscanf(f, "%s %d", b64, &rank) == 2) {
        if (base64_decode(b64, decoded, &dlen) == 0) {
            // Find parents before inserting the new rank into hash_vocab
            int32_t p1, p2;
            if (find_bpe_parents(t, decoded, dlen, &p1, &p2)) {
                pair_hashmap_insert(&t->pair_encoder, ((uint64_t)p1 << 32) | p2, rank);
            }

            hash_vocab_insert(&t->encoder, decoded, dlen, rank);
            t->decoder_tokens[rank] = malloc(dlen);
            memcpy(t->decoder_tokens[rank], decoded, dlen);
            t->decoder_lens[rank] = dlen;
            if ((size_t)rank >= t->vocab_size) t->vocab_size = (size_t)rank + 1;
        }
    }
    fclose(f);

    // Initialize scratch space
    t->arena.capacity = 16384;
    t->arena.nodes = malloc(t->arena.capacity * sizeof(BNode));
    t->heap.capacity = 16384;
    t->heap.entries = malloc(t->heap.capacity * sizeof(HeapEntry));

    return t;
}

#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Robin Hood hashmap entry for token pairs
typedef struct {
    uint64_t key;   // (left_token << 32) | right_token
    int32_t value;  // rank
    int32_t dist;   // Distance from ideal slot. -1 means empty.
} PairEntry;

typedef struct {
    PairEntry *slots;
    size_t capacity; // Must be power of 2
    size_t count;
} PairHashMap;

void pair_hashmap_init(PairHashMap *map, size_t initial_capacity);
void pair_hashmap_free(PairHashMap *map);
void pair_hashmap_insert(PairHashMap *map, uint64_t key, int32_t value);
int32_t pair_hashmap_get(const PairHashMap *map, uint64_t key);

#endif // HASHTABLE_H

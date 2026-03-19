#include "hashtable.h"
#include <stdlib.h>
#include <string.h>

static uint64_t hash_64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

void pair_hashmap_init(PairHashMap *map, size_t initial_capacity) {
    // Round up to power of 2
    size_t cap = 1;
    while (cap < initial_capacity) cap <<= 1;
    map->capacity = cap;
    map->slots = malloc(cap * sizeof(PairEntry));
    for (size_t i = 0; i < cap; i++) {
        map->slots[i].dist = -1;
    }
    map->count = 0;
}

void pair_hashmap_free(PairHashMap *map) {
    if (map->slots) free(map->slots);
    map->slots = NULL;
    map->capacity = 0;
    map->count = 0;
}

void pair_hashmap_insert(PairHashMap *map, uint64_t key, int32_t value) {
    // Rehash if load factor > 0.7
    if (map->count * 10 >= map->capacity * 7) {
        size_t old_cap = map->capacity;
        PairEntry *old_slots = map->slots;
        pair_hashmap_init(map, old_cap * 2);
        for (size_t i = 0; i < old_cap; i++) {
            if (old_slots[i].dist != -1) {
                pair_hashmap_insert(map, old_slots[i].key, old_slots[i].value);
            }
        }
        free(old_slots);
    }

    uint64_t h = hash_64(key);
    size_t mask = map->capacity - 1;
    size_t idx = h & mask;
    
    PairEntry entry = {key, value, 0};
    
    while (true) {
        if (map->slots[idx].dist == -1) {
            map->slots[idx] = entry;
            map->count++;
            return;
        }
        
        // Already exists? Update value (though for BPE it's usually 1:1)
        if (map->slots[idx].key == key) {
            map->slots[idx].value = value;
            return;
        }

        // Robin Hood: if current entry is further from its ideal than the slot entry, swap
        if (entry.dist > map->slots[idx].dist) {
            PairEntry temp = map->slots[idx];
            map->slots[idx] = entry;
            entry = temp;
        }

        idx = (idx + 1) & mask;
        entry.dist++;
    }
}

int32_t pair_hashmap_get(const PairHashMap *map, uint64_t key) {
    if (map->capacity == 0) return -1;
    
    uint64_t h = hash_64(key);
    size_t mask = map->capacity - 1;
    size_t idx = h & mask;
    int32_t dist = 0;

    while (true) {
        if (map->slots[idx].dist == -1 || dist > map->slots[idx].dist) {
            return -1;
        }
        if (map->slots[idx].key == key) {
            return map->slots[idx].value;
        }
        idx = (idx + 1) & mask;
        dist++;
    }
}

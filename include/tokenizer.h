#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Tokenizer Tokenizer;

typedef struct {
    uint64_t pretokenize_cycles;
    uint64_t merge_cycles;
    uint64_t hash_lookup_cycles;
    uint64_t total_cycles;
} TokenizerProfile;

// Load a .tiktoken file and initialize the tokenizer
// pattern: "cl100k_base" or "o200k_base" to select the pretokenizer regex
Tokenizer* tokenizer_load(const char *path, const char *pattern);

// Free tokenizer resources
void tokenizer_free(Tokenizer *t);

// Encode text into tokens. Returns the number of tokens written.
// out_tokens: array to store result tokens
// max_tokens: size of out_tokens array
size_t tokenizer_encode(Tokenizer *t, const char *text, int32_t *out_tokens, size_t max_tokens);

// Decode tokens back to text. Returns the number of bytes written.
// out_text: buffer to store result string
// max_text_len: size of out_text buffer
size_t tokenizer_decode(Tokenizer *t, const int32_t *tokens, size_t num_tokens, char *out_text, size_t max_text_len);

// Profiling utilities
void tokenizer_reset_profile(Tokenizer *t);
TokenizerProfile tokenizer_get_profile(Tokenizer *t);

#ifdef __cplusplus
}
#endif

#endif // TOKENIZER_H

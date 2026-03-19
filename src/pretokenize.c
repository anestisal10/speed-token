#include "tokenizer.h"
#include "internal.h"
#include "simd_pretokenize.h"
#include <stdbool.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
#include <intrin.h>
static bool check_avx2() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    if (cpuInfo[0] < 7) return false;
    __cpuidex(cpuInfo, 7, 0);
    return (cpuInfo[1] & (1 << 5)) != 0;
}
#else
#include <cpuid.h>
static bool check_avx2() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;
    }
    return false;
}
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
static bool check_neon() { return true; } // NEON is mandatory on ARM64
#endif

typedef struct {
    uint32_t code;
    int len;
} Codepoint;

static Codepoint decode_utf8(const unsigned char *s, size_t max_len) {
    Codepoint cp = {0, 0};
    if (max_len == 0) return cp;
    if (s[0] < 0x80) {
        cp.code = s[0];
        cp.len = 1;
    } else if ((s[0] & 0xE0) == 0xC0 && max_len >= 2) {
        cp.code = ((s[0] & 0x1F) << 6) | (s[1] & 0x3F);
        cp.len = 2;
    } else if ((s[0] & 0xF0) == 0xE0 && max_len >= 3) {
        cp.code = ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        cp.len = 3;
    } else if ((s[0] & 0xF8) == 0xF0 && max_len >= 4) {
        cp.code = ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        cp.len = 4;
    } else {
        // Handle invalid UTF-8 or lone surrogates as single bytes
        cp.code = s[0];
        cp.len = 1;
    }
    return cp;
}

static bool is_letter(uint32_t cp) {
    if (cp < 128) return isalpha(cp);
    // Simplified Unicode letter ranges
    if (cp >= 0x00C0 && cp <= 0x00FF) return cp != 0x00D7 && cp != 0x00F7;
    if (cp >= 0x0100 && cp <= 0x024F) return true;
    if (cp >= 0x0370 && cp <= 0x03FF) return true;
    if (cp >= 0x0400 && cp <= 0x04FF) return true;
    return cp >= 0x00A0 && cp < 0x2000; // Rough heuristic
}

static bool is_number(uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

static bool is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r';
}

static int match_contraction(const char *s, size_t max_len) {
    if (max_len < 2 || s[0] != '\'') return 0;
    char c1 = tolower(s[1]);
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
    if (max_len >= 3) {
        char c2 = tolower(s[2]);
        if (c1 == 'r' && c2 == 'e') return 3;
        if (c1 == 'v' && c2 == 'e') return 3;
        if (c1 == 'l' && c2 == 'l') return 3;
    }
    return 0;
}

// BPE encode helper from bpe.c
extern size_t bpe_encode_chunk(Tokenizer *t, const unsigned char *chunk, size_t len, int32_t *out);

size_t pretokenize_and_encode(Tokenizer *t, const char *text, int32_t *out_tokens, size_t max_tokens) {
    size_t text_len = strlen(text);
    size_t text_pos = 0;
    size_t token_count = 0;
    static int simd_supported = -1; // -1: uninitialized, 0: no, 1: avx2, 2: neon

    if (simd_supported == -1) {
        simd_supported = 0;
#if defined(__x86_64__) || defined(_M_X64)
        if (check_avx2()) simd_supported = 1;
#elif defined(__aarch64__) || defined(_M_ARM64)
        simd_supported = 2;
#endif
    }

    while (text_pos < text_len && token_count < max_tokens) {
        uint64_t start_pre = RDTSC();
        size_t match_len = 0;
        const char *curr = text + text_pos;
        size_t rem = text_len - text_pos;

        // SIMD fast path: identify class boundaries
        if (simd_supported == 1 && rem >= 32) {
            uint8_t classes[32];
            classify_bytes_avx2((const uint8_t*)curr, classes);
            
            // Fast skip for letters (most common case)
            if (classes[0] == CLASS_LETTER) {
                size_t n = 1;
                while (n < 32 && (classes[n] == CLASS_LETTER)) n++;
                match_len = n;
            } 
            // Fast skip for digits
            else if (classes[0] == CLASS_DIGIT) {
                size_t n = 1;
                while (n < 3 && n < 32 && (classes[n] == CLASS_DIGIT)) n++;
                match_len = n;
            }
            // Fast skip for generic whitespace (if it's not a linebreak)
            else if (classes[0] == CLASS_WHITESPACE) {
                size_t n = 1;
                while (n < 32 && (classes[n] == CLASS_WHITESPACE)) n++;
                // In cl100k, spaces at end of string or single spaces are simple.
                // Multi-spaces followed by non-space take n-1.
                if (text_pos + n == text_len) match_len = n;
                else if (n > 1) match_len = n - 1;
                else match_len = 1;
            }
        }

        // 1. Contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (match_len == 0) {
            int cont = match_contraction(curr, rem);
            if (cont > 0) {
                match_len = cont;
            }
        }

        // 2. Letters: [^\r\n\p{L}\p{N}]?\p{L}+
        if (match_len == 0) {
            Codepoint cp = decode_utf8((const unsigned char*)curr, rem);
            // Case A: Just letters
            if (is_letter(cp.code)) {
                match_len = cp.len;
                while (text_pos + match_len < text_len) {
                    Codepoint next_cp = decode_utf8((const unsigned char*)(text + text_pos + match_len), text_len - (text_pos + match_len));
                    if (is_letter(next_cp.code)) match_len += next_cp.len;
                    else break;
                }
            } 
            // Case B: Optional leading char + letters
            else if (cp.code != '\r' && cp.code != '\n' && !is_number(cp.code)) {
                // The character cp is [^\r\n\p{L}\p{N}] because of the else-if conditions
                if (text_pos + cp.len < text_len) {
                    Codepoint next_cp = decode_utf8((const unsigned char*)(text + text_pos + cp.len), text_len - (text_pos + cp.len));
                    if (is_letter(next_cp.code)) {
                        match_len = cp.len + next_cp.len;
                        while (text_pos + match_len < text_len) {
                            Codepoint nnext_cp = decode_utf8((const unsigned char*)(text + text_pos + match_len), text_len - (text_pos + match_len));
                            if (is_letter(nnext_cp.code)) match_len += nnext_cp.len;
                            else break;
                        }
                    }
                }
            }
        }

        // 3. Numbers: \p{N}{1,3}
        if (match_len == 0) {
            Codepoint cp = decode_utf8((const unsigned char*)curr, rem);
            if (is_number(cp.code)) {
                match_len = cp.len;
                int count = 1;
                while (count < 3 && text_pos + match_len < text_len) {
                    Codepoint next_cp = decode_utf8((const unsigned char*)(text + text_pos + match_len), text_len - (text_pos + match_len));
                    if (is_number(next_cp.code)) {
                        match_len += next_cp.len;
                        count++;
                    } else break;
                }
            }
        }

        // 4. Symbols:  ?[^\s\p{L}\p{N}]+[\r\n]*
        if (match_len == 0) {
            size_t temp_len = 0;
            unsigned char c0 = (unsigned char)text[text_pos];
            if (c0 == ' ' && rem > 1) {
                Codepoint next_cp = decode_utf8((const unsigned char*)(text + text_pos + 1), rem - 1);
                if (!is_whitespace(next_cp.code) && !is_letter(next_cp.code) && !is_number(next_cp.code)) {
                    temp_len = 1 + next_cp.len;
                }
            } else {
                Codepoint cp = decode_utf8((const unsigned char*)curr, rem);
                if (!is_whitespace(cp.code) && !is_letter(cp.code) && !is_number(cp.code)) {
                    temp_len = cp.len;
                }
            }

            if (temp_len > 0) {
                while (text_pos + temp_len < text_len) {
                    Codepoint next_cp = decode_utf8((const unsigned char*)(text + text_pos + temp_len), text_len - (text_pos + temp_len));
                    if (!is_whitespace(next_cp.code) && !is_letter(next_cp.code) && !is_number(next_cp.code)) temp_len += next_cp.len;
                    else break;
                }
                while (text_pos + temp_len < text_len && (text[text_pos + temp_len] == '\r' || text[text_pos + temp_len] == '\n')) {
                    temp_len++;
                }
                match_len = temp_len;
            }
        }

        // 5. Whitespace with linebreaks: \s*[\r\n]+
        if (match_len == 0) {
            size_t temp_len = 0;
            while (text_pos + temp_len < text_len && is_whitespace(text[text_pos + temp_len]) && text[text_pos + temp_len] != '\n' && text[text_pos + temp_len] != '\r') {
                temp_len++;
            }
            if (text_pos + temp_len < text_len && (text[text_pos + temp_len] == '\n' || text[text_pos + temp_len] == '\r')) {
                while (text_pos + temp_len < text_len && (text[text_pos + temp_len] == '\n' || text[text_pos + temp_len] == '\r')) {
                    temp_len++;
                }
                match_len = temp_len;
            }
        }

        // 6. Whitespace followed by more whitespace: \s+(?!\S)
        if (match_len == 0 && is_whitespace(text[text_pos])) {
            size_t n = 0;
            while (text_pos + n < text_len && is_whitespace(text[text_pos + n])) {
                n++;
            }
            if (text_pos + n == text_len) {
                match_len = n;
            } else if (n > 1) {
                match_len = n - 1;
            }
        }

        // 7. Generic Whitespace: \s+
        if (match_len == 0 && is_whitespace(text[text_pos])) {
            match_len = 1;
            while (text_pos + match_len < text_len && is_whitespace(text[text_pos + match_len])) {
                match_len++;
            }
        }

        if (match_len == 0) match_len = 1;

        t->profile.pretokenize_cycles += RDTSC() - start_pre;

        token_count += bpe_encode_chunk(t, (const unsigned char*)curr, match_len, out_tokens + token_count);
        text_pos += match_len;
    }

    return token_count;
}

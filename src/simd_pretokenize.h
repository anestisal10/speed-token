#ifndef SIMD_PRETOKENIZE_H
#define SIMD_PRETOKENIZE_H

#include <stdint.h>

// Byte classes
#define CLASS_WHITESPACE 0x01
#define CLASS_LINEBREAK  0x02
#define CLASS_LETTER     0x04
#define CLASS_DIGIT      0x08
#define CLASS_PUNCT      0x10
#define CLASS_UTF8_CONT  0x20
#define CLASS_UTF8_START 0x40

#ifdef __cplusplus
extern "C" {
#endif

// Classify 32 bytes using AVX2
void classify_bytes_avx2(const uint8_t *bytes, uint8_t *classes);

// Classify 16 bytes using NEON
void classify_bytes_neon(const uint8_t *bytes, uint8_t *classes);

#ifdef __cplusplus
}
#endif

#endif // SIMD_PRETOKENIZE_H

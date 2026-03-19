#include "simd_pretokenize.h"
#include <immintrin.h>

#ifdef _MSC_VER
#define ALIGN32_PREFIX __declspec(align(32))
#define ALIGN32_SUFFIX
#else
#define ALIGN32_PREFIX
#define ALIGN32_SUFFIX __attribute__((aligned(32)))
#endif

ALIGN32_PREFIX static const uint8_t LO_LOOKUP_ARR[32] ALIGN32_SUFFIX = {
    0x7d, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7d, 0x76, 0x74, 0x74, 0x76, 0x74, 0x74,
    0x7d, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7d, 0x76, 0x74, 0x74, 0x76, 0x74, 0x74
};

ALIGN32_PREFIX static const uint8_t HI_LOOKUP_ARR[32] ALIGN32_SUFFIX = {
    0x03, 0x00, 0x11, 0x18, 0x14, 0x14, 0x14, 0x14, 0x20, 0x20, 0x20, 0x20, 0x40, 0x40, 0x40, 0x40,
    0x03, 0x00, 0x11, 0x18, 0x14, 0x14, 0x14, 0x14, 0x20, 0x20, 0x20, 0x20, 0x40, 0x40, 0x40, 0x40
};

void classify_bytes_avx2(const uint8_t *bytes, uint8_t *classes) {
    __m256i lo_lookup = _mm256_load_si256((const __m256i*)LO_LOOKUP_ARR);
    __m256i hi_lookup = _mm256_load_si256((const __m256i*)HI_LOOKUP_ARR);
    __m256i v_bytes = _mm256_loadu_si256((const __m256i*)bytes);
    
    __m256i lo_indices = _mm256_and_si256(v_bytes, _mm256_set1_epi8(0x0F));
    // High nibble: shift each 16-bit word by 4, then mask
    __m256i hi_indices = _mm256_and_si256(_mm256_srli_epi16(v_bytes, 4), _mm256_set1_epi8(0x0F));
    
    __m256i lo_shuf = _mm256_shuffle_epi8(lo_lookup, lo_indices);
    __m256i hi_shuf = _mm256_shuffle_epi8(hi_lookup, hi_indices);
    
    __m256i result = _mm256_and_si256(lo_shuf, hi_shuf);
    _mm256_storeu_si256((__m256i*)classes, result);
}

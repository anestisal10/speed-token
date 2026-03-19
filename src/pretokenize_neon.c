#include "simd_pretokenize.h"

#ifdef __ARM_NEON
#include <arm_neon.h>

void classify_bytes_neon(const uint8_t *bytes, uint8_t *classes) {
    static const uint8_t LO_LOOKUP_ARR[16] = {
        0x7d, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7c, 0x7d, 0x76, 0x74, 0x74, 0x76, 0x74, 0x74
    };
    static const uint8_t HI_LOOKUP_ARR[16] = {
        0x03, 0x00, 0x11, 0x18, 0x14, 0x14, 0x14, 0x14, 0x20, 0x20, 0x20, 0x20, 0x40, 0x40, 0x40, 0x40
    };

    uint8x16_t lo_lookup = vld1q_u8(LO_LOOKUP_ARR);
    uint8x16_t hi_lookup = vld1q_u8(HI_LOOKUP_ARR);
    uint8x16_t v_bytes = vld1q_u8(bytes);

    uint8x16_t lo_indices = vandq_u8(v_bytes, vdupq_n_u8(0x0F));
    uint8x16_t hi_indices = vshrq_n_u8(v_bytes, 4);

    uint8x16_t lo_shuf = vqtbl1q_u8(lo_lookup, lo_indices);
    uint8x16_t hi_shuf = vqtbl1q_u8(hi_lookup, hi_indices);

    uint8x16_t result = vandq_u8(lo_shuf, hi_shuf);
    vst1q_u8(classes, result);
}
#else
void classify_bytes_neon(const uint8_t *bytes, uint8_t *classes) {
    (void)bytes; (void)classes;
    // Fallback/No-op for non-ARM
}
#endif

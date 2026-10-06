/* SanitizerCoverage runtime for distfuzz.
 *
 * Code compiled with -fsanitize-coverage=trace-pc-guard calls these hooks. Every instrumented edge gets a
 * guard number at load time; each execution of the edge bumps its counter in `dfcov_map`, saturating at 255.
 * Python reads the map through ctypes after each program and clears it before the next one.
 */
#include <stdint.h>
#include <string.h>

#define DFCOV_MAX (1u << 22)

static uint8_t dfcov_map[DFCOV_MAX];
static uint32_t dfcov_n;

void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {
    if (start == stop || *start)
        return;
    for (uint32_t *g = start; g < stop; g++)
        *g = dfcov_n < DFCOV_MAX - 1 ? ++dfcov_n : 0;
}

void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    uint8_t *c = &dfcov_map[*guard];
    if (*c != 255)
        (*c)++;
}

/* Number of guards handed out; valid indices are 1..dfcov_edges(). Index 0 collects overflow. */
uint32_t dfcov_edges(void) { return dfcov_n; }

uint8_t *dfcov_buffer(void) { return dfcov_map; }

void dfcov_reset(void) { memset(dfcov_map, 0, dfcov_n + 1); }

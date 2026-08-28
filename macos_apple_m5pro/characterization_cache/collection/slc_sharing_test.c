/*
 * slc_sharing_test.c — Minimal SLC CPU↔NPU sharing test for Apple Silicon
 *
 * Design: Paired NPU/CTRL trials across 3 arena sizes (32/64/128 MB).
 *         No cache eviction between trials — prime itself loads arena into SLC.
 *         NPU trials first to avoid polluting CTRL state.
 *         CTRL spin_wait matched to measured NPU inference time.
 *
 * Hypothesis: M4 SLC is shared between CPU and NPU.
 * Prediction: NPU inference evicts CPU-primed SLC lines → latency jump.
 *
 * Build:  clang -O2 -o slc_sharing_test slc_sharing_test.c
 * Usage:  ./slc_sharing_test [model_path] [n_trials] [n_infer]
 */

#include <errno.h>
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

/* ——— Configuration ————————————————————————————————— */
#define N_WARMUP     10        /* NPU timing warmup runs                   */
#define N_TRIALS     50        /* trials per arena × condition (default)   */
#define N_INFER      1         /* NPU inference count per trial (default)  */
#define CACHE_LINE   128       /* Apple Silicon cache line                 */
#define PROBE_KB     128       /* one probe per 128 KB SLC region          */
#define PROBE_STRIDE ((size_t)PROBE_KB * 1024)
#define MAX_ARENA_MB 128
#define N_ARENAS     3

static const int ARENA_SIZES_MB[N_ARENAS] = {32, 64, 128};

#define POC_SERVER_PY "npu_flush_server.py"

#define DEFAULT_MODEL \
    "./" \
    "EvictionModel_Heavy.mlpackage"

/* ——— Globals ——————————————————————————————————————— */
static uint8_t                 *arena;
static mach_timebase_info_data_t tb;
static volatile uint8_t         sink;

/* ——— Timing ——————————————————————————————————————— */
static inline uint64_t now(void) {
    return mach_absolute_time();
}

static inline uint64_t to_ns(uint64_t ticks) {
    return ticks * tb.numer / tb.denom;
}

static inline void fence(void) {
    __asm__ volatile("dsb ish" ::: "memory");
    __asm__ volatile("isb"      ::: "memory");
}

/* ——— Cache operations —————————————————————————————— */
static void prime_arena(int arena_mb, int dir) {
    size_t bytes = (size_t)arena_mb * 1024 * 1024;
    if (dir) {
        for (size_t off = 0; off < bytes; off += CACHE_LINE)
            sink ^= arena[off];
    } else {
        for (size_t off = bytes; off > 0; ) {
            off -= CACHE_LINE;
            sink ^= arena[off];
        }
    }
    fence();
}

static uint64_t reload_probe(int probe_idx) {
    size_t off = (size_t)probe_idx * PROBE_STRIDE;
    fence();
    uint64_t t0 = now();
    sink ^= arena[off];
    __asm__ volatile("" ::: "memory");
    fence();
    return now() - t0;
}

/* ——— NPU server (fork + pipes) ———————————————————— */
typedef struct {
    pid_t pid;
    int   in_fd;
    int   out_fd;
} npu_t;

static int read_line(int fd, char *buf, size_t cap) {
    size_t pos = 0;
    while (pos + 1 < cap) {
        char c;
        ssize_t n = read(fd, &c, 1);
        if (n <= 0) return n == 0 ? (int)pos : -1;
        buf[pos++] = c;
        if (c == '\n') break;
    }
    buf[pos] = '\0';
    return (int)pos;
}

static void write_all(int fd, const char *s) {
    size_t len = strlen(s);
    size_t off = 0;
    while (off < len) {
        ssize_t n = write(fd, s + off, len - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("write");
            exit(1);
        }
        off += (size_t)n;
    }
}

static npu_t npu_start(const char *model) {
    int p2c[2], c2p[2];
    if (pipe(p2c) || pipe(c2p)) { perror("pipe"); exit(1); }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); exit(1); }

    if (pid == 0) {
        close(p2c[1]); close(c2p[0]);
        dup2(p2c[0], STDIN_FILENO);
        dup2(c2p[1], STDOUT_FILENO);
        close(p2c[0]); close(c2p[1]);
        char _cmd[1024]; snprintf(_cmd, sizeof(_cmd), "python3 %s %s 2>/dev/null", POC_SERVER_PY, model); execl("/bin/sh", "sh", "-c", _cmd, NULL);
        _exit(127);
    }

    close(p2c[0]); close(c2p[1]);

    npu_t s = {pid, p2c[1], c2p[0]};

    char line[256];
    if (read_line(s.out_fd, line, sizeof(line)) <= 0 ||
        strstr(line, "READY") == NULL) {
        fprintf(stderr, "NPU server did not become ready (got: %s)\n", line);
        exit(1);
    }
    return s;
}

/* Returns inference duration in ticks */
static uint64_t npu_infer(npu_t *s, int n_infer) {
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "infer %d\n", n_infer);
    uint64_t t0 = now();
    write_all(s->in_fd, cmd);
    char line[256];
    if (read_line(s->out_fd, line, sizeof(line)) <= 0) {
        fprintf(stderr, "NPU inference failed\n");
        exit(1);
    }
    return now() - t0;
}

static void npu_stop(npu_t *s) {
    if (s->pid <= 0) return;
    write_all(s->in_fd, "quit\n");
    close(s->in_fd);
    close(s->out_fd);
    waitpid(s->pid, NULL, 0);
    s->pid = 0;
}

/* ——— Statistics ——————————————————————————————————— */
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

typedef struct {
    int        arena_mb;
    int        n_probes;
    int        n_trials;
    int        n_samples;
    uint64_t  *samples;      /* [n_samples], sorted after finalize      */
    uint64_t  *probe_sums;   /* [n_probes], per-probe latency sum       */
} result_t;

static result_t *result_new(int arena_mb, int n_trials) {
    result_t *r = calloc(1, sizeof(result_t));
    r->arena_mb  = arena_mb;
    r->n_probes  = (int)((size_t)arena_mb * 1024 * 1024 / PROBE_STRIDE);
    r->n_trials  = n_trials;
    r->n_samples = r->n_probes * n_trials;
    r->samples   = malloc((size_t)r->n_samples * sizeof(uint64_t));
    r->probe_sums = calloc((size_t)r->n_probes, sizeof(uint64_t));
    if (!r->samples || !r->probe_sums) {
        fprintf(stderr, "malloc failed\n"); exit(1);
    }
    return r;
}

static void result_finalize(result_t *r) {
    qsort(r->samples, r->n_samples, sizeof(uint64_t), cmp_u64);
}

static double result_mean(result_t *r) {
    double sum = 0;
    for (int i = 0; i < r->n_samples; i++)
        sum += (double)r->samples[i];
    return sum / r->n_samples;
}

static uint64_t result_pct(result_t *r, double pct) {
    int idx = (int)(pct / 100.0 * (r->n_samples - 1) + 0.5);
    if (idx < 0) idx = 0;
    if (idx >= r->n_samples) idx = r->n_samples - 1;
    return r->samples[idx];
}

/* ——— Run trials ——————————————————————————————————— */
static void run_trials(result_t *r, npu_t *npu,
                       int use_npu, uint64_t spin_ticks, int n_infer) {
    int si = 0;
    for (int trial = 0; trial < r->n_trials; trial++) {
        prime_arena(r->arena_mb, trial & 1);

        if (use_npu) {
            (void)npu_infer(npu, n_infer);
        } else {
            uint64_t t0 = now();
            while (now() - t0 < spin_ticks) {}
        }
        fence();

        for (int p = 0; p < r->n_probes; p++) {
            uint64_t lat = reload_probe(p);
            r->samples[si++] = lat;
            r->probe_sums[p] += lat;
        }

        if ((trial + 1) % 25 == 0)
            fprintf(stderr, "    %d/%d\n", trial + 1, r->n_trials);
    }
}

/* ——— Heatmap ——————————————————————————————————————— */
static void print_heatmap(const result_t *npu, const result_t *ctrl) {
    int    n    = npu->n_probes;
    double ntri = (double)npu->n_trials;

    /* compute per-probe delta in ns */
    double *deltas = malloc((size_t)n * sizeof(double));
    double min_d =  1e300;
    double max_d = -1e300;
    for (int i = 0; i < n; i++) {
        double m_npu  = (double)npu->probe_sums[i]  / ntri;
        double m_ctrl = (double)ctrl->probe_sums[i] / ntri;
        deltas[i] = to_ns((uint64_t)(m_npu - m_ctrl > 0 ? m_npu - m_ctrl : 0));
        if (deltas[i] < min_d) min_d = deltas[i];
        if (deltas[i] > max_d) max_d = deltas[i];
    }

    double range = max_d - min_d;
    if (range <= 0) range = 1;

    printf("  %3d MB  ", npu->arena_mb);

    for (int i = 0; i < n; i++) {
        double frac = (deltas[i] - min_d) / range;
        const char *ch;
        if      (frac < 0.15) ch = " ";
        else if (frac < 0.35) ch = "\xe2\x96\x91";   /* U+2591 LIGHT SHADE  */
        else if (frac < 0.55) ch = "\xe2\x96\x92";   /* U+2592 MEDIUM SHADE */
        else if (frac < 0.75) ch = "\xe2\x96\x93";   /* U+2593 DARK SHADE   */
        else                  ch = "\xe2\x96\x88";   /* U+2588 FULL BLOCK   */
        printf("%s", ch);
        if ((i + 1) % 64 == 0 && i + 1 < n)
            printf("\n         ");  /* 9 spaces to align with "  xx MB  " */
    }

    printf("  [%.0f – %.0f ns]\n\n", min_d, max_d);
    free(deltas);
}

/* ——— Histogram ————————————————————————————————————— */
static void print_histogram(const result_t *ctrl, const result_t *npu) {
    int    n      = ctrl->n_probes;
    double ntri   = (double)ctrl->n_trials;
    int    nbins  = 20;
    int    bin_ns = 20;
    int    c_bins[20] = {0}, n_bins[20] = {0};

    for (int i = 0; i < n; i++) {
        int cb = (int)(to_ns(ctrl->probe_sums[i] / ntri) / bin_ns);
        int nb = (int)(to_ns(npu->probe_sums[i]  / ntri) / bin_ns);
        if (cb >= nbins) cb = nbins - 1;
        if (nb >= nbins) nb = nbins - 1;
        c_bins[cb]++; n_bins[nb]++;
    }

    int mx = 1;
    for (int i = 0; i < nbins; i++) {
        if (c_bins[i] > mx) mx = c_bins[i];
        if (n_bins[i] > mx) mx = n_bins[i];
    }

    printf("  Latency histogram (32 MB arena, %d probes):\n\n", n);
    printf("  %-10s  %-42s %-42s\n", "ns", "CTRL", "NPU");
    printf("  %-10s  %-42s %-42s\n", "----", "----", "----");

    for (int i = 0; i < nbins; i++) {
        printf("  %3d–%3d   ", i * bin_ns, (i + 1) * bin_ns);
        int cw = c_bins[i] * 40 / mx;
        int nw = n_bins[i] * 40 / mx;
        for (int j = 0; j < 40; j++) printf("%c", j < cw ? '#' : ' ');
        printf(" %3d  ", c_bins[i]);
        for (int j = 0; j < 40; j++) printf("%c", j < nw ? '#' : ' ');
        printf(" %3d\n", n_bins[i]);
    }
    printf("\n");
}

/* ——— Main —————————————————————————————————————————— */
int main(int argc, char **argv) {
    const char *model    = argc > 1 ? argv[1] : DEFAULT_MODEL;
    int         n_trials = argc > 2 ? atoi(argv[2]) : N_TRIALS;
    int         n_infer  = argc > 3 ? atoi(argv[3]) : N_INFER;
    if (n_trials < 1) n_trials = N_TRIALS;
    if (n_infer  < 1) n_infer  = N_INFER;

    mach_timebase_info(&tb);

    /* allocate max arena once, mlock so it stays resident */
    size_t max_bytes = (size_t)MAX_ARENA_MB * 1024 * 1024;
    arena = mmap(NULL, max_bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) { perror("mmap arena"); return 1; }
    if (mlock(arena, max_bytes) != 0)
        fprintf(stderr, "warning: mlock failed: %s\n", strerror(errno));
    memset(arena, 0xAA, max_bytes);

    printf("=== SLC CPU<->NPU Sharing Test -- M4 ===\n");
    printf("Model:  %s\n", model);
    printf("Trials: %d per arena x condition  |  n_infer: %d  |  Probes: 1 per %d KB\n\n",
           n_trials, n_infer, PROBE_KB);

    /* ——— Phase 0: measure NPU inference time ——————— */
    printf("--- Phase 0: measuring NPU inference time ---\n");
    npu_t npu = npu_start(model);

    uint64_t npu_ticks = 0;
    for (int i = 0; i < N_WARMUP; i++) {
        prime_arena(32, i & 1);   /* warm arena so NPU sees populated SLC */
        uint64_t t = npu_infer(&npu, n_infer);
        npu_ticks += t;
        fprintf(stderr, "  warmup %d/%d: %.1f ms\n",
                i + 1, N_WARMUP, to_ns(t) / 1e6);
    }
    uint64_t avg_npu_ticks = npu_ticks / N_WARMUP;
    printf("NPU inference: %.1f ms  ->  CTRL spin_wait matched\n\n",
           to_ns(avg_npu_ticks) / 1e6);

    /* ——— Phase 1: NPU trials (first, so NPU state doesn't leak to CTRL) */
    printf("--- Phase 1: NPU trials ---\n");
    result_t *npu_res[N_ARENAS];
    for (int a = 0; a < N_ARENAS; a++) {
        int mb = ARENA_SIZES_MB[a];
        printf("  Arena %d MB (%d probes):\n", mb, mb * 1024 / PROBE_KB);
        npu_res[a] = result_new(mb, n_trials);
        run_trials(npu_res[a], &npu, 1, 0, n_infer);
        result_finalize(npu_res[a]);
    }

    /* ——— Phase 2: CTRL trials —————————————————————— */
    printf("\n--- Phase 2: CTRL trials (spin_wait=%.1f ms) ---\n",
           to_ns(avg_npu_ticks) / 1e6);
    result_t *ctrl_res[N_ARENAS];
    for (int a = 0; a < N_ARENAS; a++) {
        int mb = ARENA_SIZES_MB[a];
        printf("  Arena %d MB (%d probes):\n", mb, mb * 1024 / PROBE_KB);
        ctrl_res[a] = result_new(mb, n_trials);
        run_trials(ctrl_res[a], &npu, 0, avg_npu_ticks, n_infer);
        result_finalize(ctrl_res[a]);
    }

    npu_stop(&npu);

    /* ——— Results table ————————————————————————————— */
    printf("\n");
    printf("================================================================================\n");
    printf("  RESULTS\n");
    printf("================================================================================\n\n");
    printf("  %-7s  %9s  %9s  %9s  %7s  %15s  %15s\n",
           "ARENA", "CTRL_mu", "NPU_mu", "DELTA", "RATIO",
           "CTRL_P50/P99", "NPU_P50/P99");
    printf("  %-7s  %9s  %9s  %9s  %7s  %15s  %15s\n",
           "-------", "-------", "------", "------", "------",
           "--------------", "--------------");

    for (int a = 0; a < N_ARENAS; a++) {
        double   c_mean = to_ns((uint64_t)result_mean(ctrl_res[a]));
        double   n_mean = to_ns((uint64_t)result_mean(npu_res[a]));
        double   delta  = n_mean - c_mean;
        double   ratio  = c_mean > 0 ? n_mean / c_mean : 0;
        uint64_t c_p50  = to_ns(result_pct(ctrl_res[a], 50));
        uint64_t c_p99  = to_ns(result_pct(ctrl_res[a], 99));
        uint64_t n_p50  = to_ns(result_pct(npu_res[a], 50));
        uint64_t n_p99  = to_ns(result_pct(npu_res[a], 99));

        printf("  %3d MB   %8.1f   %8.1f   %+8.1f   %5.1fx   "
               "%5llu / %-5llu   %5llu / %-5llu  ns\n",
               ARENA_SIZES_MB[a], c_mean, n_mean, delta, ratio,
               (unsigned long long)c_p50, (unsigned long long)c_p99,
               (unsigned long long)n_p50, (unsigned long long)n_p99);
    }

    /* ——— Heatmaps —————————————————————————————————— */
    printf("\n  Eviction heatmaps (NPU - CTRL delta per %d KB probe):\n", PROBE_KB);
    printf("  Legend: ' '=low  ░  ▒  ▓  █=high\n\n");
    for (int a = 0; a < N_ARENAS; a++)
        print_heatmap(npu_res[a], ctrl_res[a]);

    print_histogram(ctrl_res[0], npu_res[0]);

    /* dump raw probe data for 32 MB arena */
    FILE *df = fopen("/tmp/slc_probe_data.csv", "w");
    fprintf(df, "probe,ctrl_ns,npu_ns\n");
    for (int p = 0; p < ctrl_res[0]->n_probes; p++) {
        fprintf(df, "%d,%.0f,%.0f\n",
                p,
                (double)to_ns((uint64_t)(ctrl_res[0]->probe_sums[p] / (double)ctrl_res[0]->n_trials)),
                (double)to_ns((uint64_t)(npu_res[0]->probe_sums[p]  / (double)npu_res[0]->n_trials)));
    }
    fclose(df);
    fprintf(stderr, "raw data written to /tmp/slc_probe_data.csv\n");


    /* ——— Conclusion ———————————————————————————————— */
    double c32_ns = to_ns((uint64_t)result_mean(ctrl_res[0]));
    double n32_ns = to_ns((uint64_t)result_mean(npu_res[0]));
    double d32 = n32_ns - c32_ns;
    double r32 = c32_ns > 0 ? n32_ns / c32_ns : 0;

    printf("  CONCLUSION: ");
    if (d32 > 30 && r32 > 2.0) {
        printf("SLC IS SHARED between CPU and NPU.\n");
        printf("  NPU inference evicts CPU-primed SLC lines "
               "(%.1fx latency increase at 32 MB arena).\n", r32);
        printf("  Viable CPU<->NPU side channel confirmed.\n");
    } else if (d32 > 10) {
        printf("SLC appears PARTIALLY shared.\n");
        printf("  Moderate eviction effect (%.1fx at 32 MB arena).\n", r32);
    } else {
        printf("No strong evidence of SLC sharing (%.1fx at 32 MB arena).\n", r32);
    }

    /* ——— Cleanup ——————————————————————————————————— */
    for (int a = 0; a < N_ARENAS; a++) {
        free(npu_res[a]->samples);
        free(npu_res[a]->probe_sums);
        free(npu_res[a]);
        free(ctrl_res[a]->samples);
        free(ctrl_res[a]->probe_sums);
        free(ctrl_res[a]);
    }
    munlock(arena, max_bytes);
    munmap(arena, max_bytes);

    printf("\n  Done. sink=%u\n", sink);
    return 0;
}

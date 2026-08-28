#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

#define N_WARMUP 10
#define DEFAULT_TRIALS 50
#define DEFAULT_INFER 1
#define CACHE_LINE 64
#define PROBE_KB 128
#define PROBE_STRIDE ((size_t)PROBE_KB * 1024)
#define MAX_ARENA_MB 128
#define N_ARENAS 3

static const int ARENA_MB[N_ARENAS] = {32, 64, 128};
static uint8_t *arena;
static int arena_fd = -1;
static char arena_shm_name[128];
static volatile uint8_t sink;
static double cycles_per_ns;

typedef struct {
    pid_t pid;
    int in_fd;
    int out_fd;
} npu_server_t;

typedef struct {
    int arena_mb;
    int n_probes;
    int n_trials;
    int n_samples;
    uint64_t *samples;
    uint64_t *probe_sums;
} result_t;

static inline uint64_t rdtscp_ordered(void) {
    unsigned aux;
    _mm_lfence();
    uint64_t value = __rdtscp(&aux);
    _mm_lfence();
    return value;
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void calibrate_tsc(void) {
    struct timespec delay = {.tv_sec = 0, .tv_nsec = 200000000};
    uint64_t n0 = monotonic_ns();
    uint64_t c0 = rdtscp_ordered();
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {}
    uint64_t c1 = rdtscp_ordered();
    uint64_t n1 = monotonic_ns();
    cycles_per_ns = (double)(c1 - c0) / (double)(n1 - n0);
    if (!(cycles_per_ns > 0.1 && cycles_per_ns < 10.0)) {
        fprintf(stderr, "invalid TSC calibration: %.6f cycles/ns\n", cycles_per_ns);
        exit(1);
    }
}

static double cycles_to_ns(double cycles) {
    return cycles / cycles_per_ns;
}

static void pin_cpu(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "warning: cannot pin to CPU %d: %s\n", cpu, strerror(errno));
    }
}

static void prime_arena(int arena_mb, int reverse) {
    size_t bytes = (size_t)arena_mb * 1024 * 1024;
    if (!reverse) {
        for (size_t off = 0; off < bytes; off += CACHE_LINE) sink ^= arena[off];
    } else {
        for (size_t off = bytes; off > 0;) {
            off -= CACHE_LINE;
            sink ^= arena[off];
        }
    }
    _mm_mfence();
}

static uint64_t reload_probe(int probe) {
    size_t off = (size_t)probe * PROBE_STRIDE;
    uint64_t begin = rdtscp_ordered();
    sink ^= arena[off];
    uint64_t end = rdtscp_ordered();
    return end - begin;
}

static int read_line(int fd, char *buffer, size_t capacity) {
    size_t pos = 0;
    while (pos + 1 < capacity) {
        char byte;
        ssize_t count = read(fd, &byte, 1);
        if (count == 0) break;
        if (count < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        buffer[pos++] = byte;
        if (byte == '\n') break;
    }
    buffer[pos] = '\0';
    return (int)pos;
}

static void write_all(int fd, const char *text) {
    size_t length = strlen(text);
    size_t offset = 0;
    while (offset < length) {
        ssize_t count = write(fd, text + offset, length - offset);
        if (count < 0) {
            if (errno == EINTR) continue;
            perror("write");
            exit(1);
        }
        offset += (size_t)count;
    }
}

static npu_server_t start_server(const char *server, const char *model,
                                 const char *shared_path, size_t shared_bytes,
                                 dev_t shared_dev, ino_t shared_ino) {
    int parent_to_child[2], child_to_parent[2];
    if (pipe(parent_to_child) != 0 || pipe(child_to_parent) != 0) {
        perror("pipe");
        exit(1);
    }
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(1);
    }
    if (pid == 0) {
        char bytes_arg[64], dev_arg[64], ino_arg[64];
        snprintf(bytes_arg, sizeof(bytes_arg), "%zu", shared_bytes);
        snprintf(dev_arg, sizeof(dev_arg), "%" PRIuMAX, (uintmax_t)shared_dev);
        snprintf(ino_arg, sizeof(ino_arg), "%" PRIuMAX, (uintmax_t)shared_ino);
        dup2(parent_to_child[0], STDIN_FILENO);
        dup2(child_to_parent[1], STDOUT_FILENO);
        close(parent_to_child[0]);
        close(parent_to_child[1]);
        close(child_to_parent[0]);
        close(child_to_parent[1]);
        execl(server, server, model, shared_path, bytes_arg, dev_arg, ino_arg,
              (char *)NULL);
        perror("exec NPU server");
        _exit(127);
    }
    close(parent_to_child[0]);
    close(child_to_parent[1]);
    npu_server_t result = {
        .pid = pid, .in_fd = parent_to_child[1], .out_fd = child_to_parent[0]
    };
    char line[512] = {0};
    if (read_line(result.out_fd, line, sizeof(line)) <= 0 ||
        strncmp(line, "READY", 5) != 0 ||
        strstr(line, "shared_memory=verified") == NULL) {
        fprintf(stderr, "NPU server failed to start (received: %s)\n", line);
        exit(1);
    }
    fprintf(stderr, "  %s", line);
    return result;
}

static uint64_t infer(npu_server_t *server, int count) {
    char command[64], reply[512];
    snprintf(command, sizeof(command), "infer %d\n", count);
    uint64_t begin = rdtscp_ordered();
    write_all(server->in_fd, command);
    if (read_line(server->out_fd, reply, sizeof(reply)) <= 0 ||
        strncmp(reply, "OK", 2) != 0) {
        fprintf(stderr, "NPU inference failed (received: %s)\n", reply);
        exit(1);
    }
    return rdtscp_ordered() - begin;
}

static void stop_server(npu_server_t *server) {
    if (server->pid <= 0) return;
    write_all(server->in_fd, "quit\n");
    char reply[64];
    (void)read_line(server->out_fd, reply, sizeof(reply));
    close(server->in_fd);
    close(server->out_fd);
    int status = 0;
    waitpid(server->pid, &status, 0);
    server->pid = 0;
}

static result_t *new_result(int arena_mb, int trials) {
    result_t *result = calloc(1, sizeof(*result));
    if (!result) {
        perror("calloc result");
        exit(1);
    }
    result->arena_mb = arena_mb;
    result->n_probes = (int)((size_t)arena_mb * 1024 * 1024 / PROBE_STRIDE);
    result->n_trials = trials;
    result->n_samples = result->n_probes * trials;
    result->samples = malloc((size_t)result->n_samples * sizeof(uint64_t));
    result->probe_sums = calloc((size_t)result->n_probes, sizeof(uint64_t));
    if (!result->samples || !result->probe_sums) {
        perror("allocate result arrays");
        exit(1);
    }
    return result;
}

static int compare_u64(const void *left, const void *right) {
    uint64_t a = *(const uint64_t *)left;
    uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static double result_mean(const result_t *result) {
    long double sum = 0;
    for (int i = 0; i < result->n_samples; ++i) sum += result->samples[i];
    return (double)(sum / result->n_samples);
}

static double result_percentile(const result_t *result, double percentile) {
    int index = (int)llround(percentile / 100.0 * (result->n_samples - 1));
    return (double)result->samples[index];
}

static void run_trials(result_t *result, npu_server_t *server, int use_npu,
                       uint64_t control_cycles, int n_infer, FILE *raw) {
    int sample = 0;
    for (int trial = 0; trial < result->n_trials; ++trial) {
        prime_arena(result->arena_mb, trial & 1);
        if (use_npu) {
            (void)infer(server, n_infer);
        } else {
            uint64_t begin = rdtscp_ordered();
            while (rdtscp_ordered() - begin < control_cycles) _mm_pause();
        }
        _mm_mfence();
        for (int probe = 0; probe < result->n_probes; ++probe) {
            uint64_t latency = reload_probe(probe);
            result->samples[sample++] = latency;
            result->probe_sums[probe] += latency;
            fprintf(raw, "%s,%d,%d,%d,%.3f\n",
                    use_npu ? "npu" : "ctrl", result->arena_mb, trial, probe,
                    cycles_to_ns((double)latency));
        }
        if ((trial + 1) % 10 == 0 || trial + 1 == result->n_trials) {
            fprintf(stderr, "    %d/%d\n", trial + 1, result->n_trials);
        }
    }
    qsort(result->samples, (size_t)result->n_samples,
          sizeof(uint64_t), compare_u64);
}

static void free_result(result_t *result) {
    if (!result) return;
    free(result->samples);
    free(result->probe_sums);
    free(result);
}

static void make_output_dir(const char *path) {
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        perror("mkdir output");
        exit(1);
    }
}

static void create_shared_arena(size_t bytes, char *path, size_t path_capacity,
                                struct stat *status) {
    snprintf(arena_shm_name, sizeof(arena_shm_name), "/slc_x86_%ld",
             (long)getpid());
    arena_fd = shm_open(arena_shm_name, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (arena_fd < 0) {
        perror("shm_open arena");
        exit(1);
    }
    if (ftruncate(arena_fd, (off_t)bytes) != 0) {
        perror("ftruncate arena");
        exit(1);
    }
    arena = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, arena_fd, 0);
    if (arena == MAP_FAILED) {
        perror("mmap shared arena");
        exit(1);
    }
    if (fstat(arena_fd, status) != 0) {
        perror("fstat shared arena");
        exit(1);
    }
    snprintf(path, path_capacity, "/dev/shm/%s",
             arena_shm_name[0] == '/' ? arena_shm_name + 1 : arena_shm_name);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 7) {
        fprintf(stderr,
                "Usage: %s SERVER MODEL [TRIALS=50] [N_INFER=1] [CPU=-1] "
                "[OUTPUT_DIR=results]\n", argv[0]);
        return 2;
    }
    const char *server_path = argv[1];
    const char *model_path = argv[2];
    int trials = argc > 3 ? atoi(argv[3]) : DEFAULT_TRIALS;
    int n_infer = argc > 4 ? atoi(argv[4]) : DEFAULT_INFER;
    int cpu = argc > 5 ? atoi(argv[5]) : -1;
    const char *output_dir = argc > 6 ? argv[6] : "results";
    if (trials < 1 || n_infer < 1) {
        fprintf(stderr, "TRIALS and N_INFER must be positive\n");
        return 2;
    }

    pin_cpu(cpu);
    calibrate_tsc();
    make_output_dir(output_dir);

    char raw_path[4096], summary_path[4096];
    snprintf(raw_path, sizeof(raw_path), "%s/raw_samples.csv", output_dir);
    snprintf(summary_path, sizeof(summary_path), "%s/summary.csv", output_dir);
    FILE *raw = fopen(raw_path, "w");
    FILE *summary = fopen(summary_path, "w");
    if (!raw || !summary) {
        perror("open result file");
        return 1;
    }
    fprintf(raw, "condition,arena_mb,trial,probe,latency_ns\n");
    fprintf(summary,
            "arena_mb,ctrl_mean_ns,npu_mean_ns,delta_ns,ratio,"
            "ctrl_p50_ns,ctrl_p99_ns,npu_p50_ns,npu_p99_ns\n");

    size_t max_bytes = (size_t)MAX_ARENA_MB * 1024 * 1024;
    char shared_path[4096];
    struct stat shared_status;
    create_shared_arena(max_bytes, shared_path, sizeof(shared_path),
                        &shared_status);
    madvise(arena, max_bytes, MADV_HUGEPAGE);
    /* 0x3f3f3f3f is a finite float (~0.747); the first 786432 bytes are also
       the model input tensor mapped by the NPU server. */
    memset(arena, 0x3F, max_bytes);
    if (mlock(arena, max_bytes) != 0) {
        fprintf(stderr, "warning: mlock(%d MiB) failed: %s\n",
                MAX_ARENA_MB, strerror(errno));
    }

    printf("=== CPU <-> NPU cache/memory-hierarchy interference test (x86) ===\n");
    printf("model=%s\nserver=%s\ntrials=%d, n_infer=%d, cpu=%d, "
           "tsc=%.6f cycles/ns\n\n",
           model_path, server_path, trials, n_infer, cpu, cycles_per_ns);
    printf("shared arena=%s, bytes=%zu, dev=%" PRIuMAX ", ino=%" PRIuMAX "\n\n",
           shared_path, max_bytes, (uintmax_t)shared_status.st_dev,
           (uintmax_t)shared_status.st_ino);

    npu_server_t server = start_server(
        server_path, model_path, shared_path, max_bytes,
        shared_status.st_dev, shared_status.st_ino);
    uint64_t inference_cycles = 0;
    for (int i = 0; i < N_WARMUP; ++i) {
        prime_arena(32, i & 1);
        uint64_t duration = infer(&server, n_infer);
        inference_cycles += duration;
        fprintf(stderr, "  warmup %d/%d: %.3f ms\n", i + 1, N_WARMUP,
                cycles_to_ns((double)duration) / 1e6);
    }
    inference_cycles /= N_WARMUP;

    result_t *npu[N_ARENAS] = {0};
    result_t *ctrl[N_ARENAS] = {0};
    for (int index = 0; index < N_ARENAS; ++index) {
        npu[index] = new_result(ARENA_MB[index], trials);
        fprintf(stderr, "  NPU %d MiB\n", ARENA_MB[index]);
        run_trials(npu[index], &server, 1, 0, n_infer, raw);
    }
    for (int index = 0; index < N_ARENAS; ++index) {
        ctrl[index] = new_result(ARENA_MB[index], trials);
        fprintf(stderr, "  CTRL %d MiB\n", ARENA_MB[index]);
        run_trials(ctrl[index], &server, 0, inference_cycles, n_infer, raw);
    }
    stop_server(&server);

    printf("\n%-8s %12s %12s %12s %8s %13s %13s\n",
           "arena", "ctrl mean", "npu mean", "delta", "ratio",
           "ctrl p50/p99", "npu p50/p99");
    for (int index = 0; index < N_ARENAS; ++index) {
        double cmean = cycles_to_ns(result_mean(ctrl[index]));
        double nmean = cycles_to_ns(result_mean(npu[index]));
        double delta = nmean - cmean;
        double ratio = cmean > 0 ? nmean / cmean : NAN;
        double cp50 = cycles_to_ns(result_percentile(ctrl[index], 50));
        double cp99 = cycles_to_ns(result_percentile(ctrl[index], 99));
        double np50 = cycles_to_ns(result_percentile(npu[index], 50));
        double np99 = cycles_to_ns(result_percentile(npu[index], 99));
        printf("%4d MiB %10.2f ns %10.2f ns %+10.2f ns %7.3fx "
               "%5.1f/%-5.1f %5.1f/%-5.1f\n",
               ARENA_MB[index], cmean, nmean, delta, ratio,
               cp50, cp99, np50, np99);
        fprintf(summary, "%d,%.6f,%.6f,%.6f,%.9f,%.6f,%.6f,%.6f,%.6f\n",
                ARENA_MB[index], cmean, nmean, delta, ratio,
                cp50, cp99, np50, np99);
    }
    printf("\nInference/control window: %.3f ms\n",
           cycles_to_ns((double)inference_cycles) / 1e6);
    printf("This measurement shows interference in the shared cache/memory "
           "hierarchy; it does not by itself identify a specific cache level.\n");
    printf("Raw data: %s\nSummary: %s\n", raw_path, summary_path);

    fclose(raw);
    fclose(summary);
    munlock(arena, max_bytes);
    munmap(arena, max_bytes);
    close(arena_fd);
    if (shm_unlink(arena_shm_name) != 0) {
        fprintf(stderr, "warning: shm_unlink(%s) failed: %s\n",
                arena_shm_name, strerror(errno));
    }
    for (int index = 0; index < N_ARENAS; ++index) {
        free_result(npu[index]);
        free_result(ctrl[index]);
    }
    fprintf(stderr, "done, sink=%u\n", sink);
    return 0;
}

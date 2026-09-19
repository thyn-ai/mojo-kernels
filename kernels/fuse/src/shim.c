/*
 * Threading shim for the fusemojo Bitap kernel.
 *
 * The Mojo kernel exports a thread-safe per-range entry point
 * (fusemojo_search_range); this shim owns pthread creation so no function
 * pointers cross the Mojo boundary. Each query is split into disjoint,
 * ascending text ranges executed concurrently; every range writes only its
 * own texts' output slots and its own per-job stash, and fusemojo_search_end
 * concatenates per-job stashes in job order, so results are identical
 * regardless of scheduling.
 *
 * Compiled as libfusemojoshim.{dylib,so} and linked against libfusemojo.
 */
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>

extern int32_t fusemojo_search_range(void *ctx, int32_t job_id, int32_t start, int32_t end);

#define FUSEMOJO_MAX_JOBS 64
/* Below ~2048 texts per job, thread overhead outweighs the parallelism win. */
#define FUSEMOJO_MIN_CHUNK 2048

typedef struct {
  void *ctx;
  int32_t job_id;
  int32_t start;
  int32_t end;
  int32_t rc;
} fusemojo_range_job;

static void *fusemojo_range_main(void *arg) {
  fusemojo_range_job *job = (fusemojo_range_job *)arg;
  job->rc = fusemojo_search_range(job->ctx, job->job_id, job->start, job->end);
  return NULL;
}

/*
 * Run the query over all texts.
 *
 * ctx:         opaque context from fusemojo_search_begin (must have been
 *              created with n_jobs >= the number of jobs used here).
 * n_texts:     number of texts in the index.
 * max_threads: 0 = one job per online CPU; >0 = cap. The number of jobs is
 *              also bounded by FUSEMOJO_MAX_JOBS and a minimum chunk size.
 *
 * Returns 0 on success, or the first non-zero range status.
 */
int32_t fusemojo_search_parallel(void *ctx, int32_t n_texts, int32_t max_threads, int32_t min_chunk) {
  if (!ctx) return 1;
  if (n_texts < 0) return 2;
  if (n_texts == 0) return 0;
  if (min_chunk < 1) min_chunk = FUSEMOJO_MIN_CHUNK;

  long cpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (cpus < 1) cpus = 1;
  int32_t n_jobs = (int32_t)cpus;
  if (max_threads > 0 && max_threads < n_jobs) n_jobs = max_threads;
  if (n_jobs > FUSEMOJO_MAX_JOBS) n_jobs = FUSEMOJO_MAX_JOBS;
  if (n_jobs > n_texts) n_jobs = n_texts;

  int32_t chunk = (n_texts + n_jobs - 1) / n_jobs;
  if (chunk < min_chunk) chunk = min_chunk;
  n_jobs = (n_texts + chunk - 1) / chunk;
  if (n_jobs <= 1) {
    return fusemojo_search_range(ctx, 0, 0, n_texts);
  }

  pthread_t threads[FUSEMOJO_MAX_JOBS];
  fusemojo_range_job jobs[FUSEMOJO_MAX_JOBS];
  int32_t n_pthreads = 0;
  int32_t rc = 0;

  /* Spawn jobs 0..n_jobs-2 on pthreads; run the last job on the caller. */
  for (int32_t j = 0; j < n_jobs - 1; j++) {
    int32_t start = j * chunk;
    int32_t end = start + chunk;
    if (end > n_texts) end = n_texts;
    jobs[j].ctx = ctx;
    jobs[j].job_id = j;
    jobs[j].start = start;
    jobs[j].end = end;
    jobs[j].rc = 0;
    if (pthread_create(&threads[j], NULL, fusemojo_range_main, &jobs[j]) != 0) {
      /* Thread creation failed: finish this and all remaining jobs inline
       * (each with its own job id), then join what was spawned. */
      for (int32_t k = j; k < n_jobs; k++) {
        int32_t kstart = k * chunk;
        int32_t kend = kstart + chunk;
        if (kend > n_texts) kend = n_texts;
        int32_t krc = fusemojo_search_range(ctx, k, kstart, kend);
        if (krc != 0 && rc == 0) rc = krc;
      }
      goto join;
    }
    n_pthreads++;
  }
  {
    int32_t j = n_jobs - 1;
    int32_t start = j * chunk;
    int32_t end = n_texts;
    int32_t last_rc = fusemojo_search_range(ctx, j, start, end);
    if (last_rc != 0 && rc == 0) rc = last_rc;
  }

join:
  for (int32_t j = 0; j < n_pthreads; j++) {
    pthread_join(threads[j], NULL);
    if (jobs[j].rc != 0 && rc == 0) rc = jobs[j].rc;
  }
  return rc;
}

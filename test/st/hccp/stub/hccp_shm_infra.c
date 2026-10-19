/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* hccp_shm_infra.c -- memfd-backed shared memory for the ST framework.
 *
 * Segment lifecycle (t3):
 *   Every segment (RDMA/UB/BUF/NOTIFY/LITE/DEVMEM/H2D) is an anonymous memfd
 *   created by ONE process -- the first
 *   infra-bearing library the runner loads, through hccp_memfd_precreate_ctor
 *   below. The creator fully initializes the segment (locks, counters, magic
 *   last) and only then publishes it: dup2() onto the fixed fd slot +
 *   setenv() of the matching HCCP_ST_MEMFD_* variable. Publication is the
 *   linearization point: attachers that find the env var can trust the
 *   segment -- no magic spin, no creator liveness probe, no stale recycling.
 *   Host children (fork, no exec) and device processes (fork+execv; the
 *   slots carry no CLOEXEC) inherit both the descriptors and the env, so
 *   every descendant attaches through the same channel. Sibling-sharing is
 *   why precreation lives in the common ancestor: RDMA/NOTIFY/DEVMEM are
 *   shared between sibling host children, which cannot inherit descriptors
 *   from each other.
 *
 *   The kernel reclaims a memfd when its last fd closes, i.e. when the last
 *   process of the run exits: a SIGKILLed run leaves nothing behind, and two
 *   concurrent runners hold disjoint anonymous segments. A descendant whose
 *   inherited descriptor does not validate fails fast -- re-creating would
 *   silently split the run in two, the exact bug class this migration kills.
 *
 *   No segment lives on legacy SysV anymore; the transition helpers at the
 *   bottom only remove the dead SysV keys of the migrated segments so
 *   leftovers from pre-memfd runs cannot accumulate.
 *
 * Lock discipline:
 *   - Every segment lock (RDMA/UB/BUF/DEVMEM/LITE) is a robust process-shared
 *     mutex, initialized through hccp_shm_lock_init and taken through
 *     hccp_shm_lock_acquire, whose EOWNERDEAD path marks the mutex consistent
 *     (the protected state self-heals: a disturbed bitmap or slot table costs
 *     at most one slot of capacity).
 *   - The robust flip was one atomic batch by design: the RDMA/UB locks used
 *     to be non-robust while urma.c locked them at three direct sites that
 *     ignored the return value, and flipping the attribute alone would have
 *     turned a dead holder into ENOTRECOVERABLE after an unchecked
 *     EOWNERDEAD -- silently unlocked critical sections, worse than the
 *     deadlock it replaces. The batch flipped the attributes and every lock
 *     site (urma.c rdv trio + the pkt ring helpers below) together. */

#include "hccp_shm_infra.h"

#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000U
#endif

/* --------------------------------------------------------------------------------------------
 * The one lock discipline (helpers declared by hccp_shm_layout.h)
 * ------------------------------------------------------------------------------------------ */

int hccp_shm_lock_init(hccp_shm_lock_t* lock)
{
    pthread_mutexattr_t attr;
    int rc = pthread_mutexattr_init(&attr);
    if (rc != 0) {
        return rc;
    }
    rc = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    if (rc == 0) {
        rc = pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
    }
    if (rc == 0) {
        rc = pthread_mutex_init(lock, &attr);
    }
    (void)pthread_mutexattr_destroy(&attr);
    return rc;
}

int hccp_shm_lock_acquire(hccp_shm_lock_t* lock)
{
    int rc = pthread_mutex_lock(lock);
    if (rc == EOWNERDEAD) {
        /* Dead holder: the protected state self-heals (a disturbed slot bitmap
         * costs at most one slot of capacity), so mark consistent and go on. */
        if (pthread_mutex_consistent(lock) != 0) {
            /* Recovery refused: release the mutex so it settles into
             * ENOTRECOVERABLE and surface the failure to the caller. */
            (void)pthread_mutex_unlock(lock);
            return ENOTRECOVERABLE;
        }
        return 0;
    }
    return rc;
}

int hccp_shm_lock_release(hccp_shm_lock_t* lock) { return pthread_mutex_unlock(lock); }

/* The RDMA/UB segments used to carry a non-robust prefix lock here; the t4
 * flip moved every lock site to the robust helpers, so both segments now
 * initialize through hccp_shm_lock_init (see file header for why the
 * attribute flip and the call-site flip had to land in one batch). */

/* --------------------------------------------------------------------------------------------
 * memfd core
 * ------------------------------------------------------------------------------------------ */

enum {
    HCCP_MEMFD_SEG_RDMA = 0,
    HCCP_MEMFD_SEG_UB,
    HCCP_MEMFD_SEG_BUF,
    HCCP_MEMFD_SEG_NOTIFY,
    HCCP_MEMFD_SEG_LITE,
    HCCP_MEMFD_SEG_DEVMEM,
    HCCP_MEMFD_SEG_H2D,
    HCCP_MEMFD_SEG_COUNT
};

typedef struct {
    const char* name; /* memfd name, visible as memfd:<name> in /proc/<pid>/fd */
    int slot;         /* fixed fd slot the segment is published on */
    const char* env;  /* env var carrying the slot number to descendants */
    size_t size;      /* ftruncate length, layout-frozen */
    uint32_t magic;   /* 0 = no magic (NOTIFY is a born-zero CAS registry) */
    void* fixed;      /* fixed VA window, NULL = kernel-chosen address */
} hccp_memfd_segdesc_t;

static const hccp_memfd_segdesc_t g_hccp_memfd_segs[HCCP_MEMFD_SEG_COUNT] = {
    [HCCP_MEMFD_SEG_RDMA]
    = {"hccp_st_rdma", HCCP_MEMFD_SLOT_RDMA, HCCP_MEMFD_ENV_RDMA, HCCP_SHM_SIZE_RDMA, HCCP_SHM_MAGIC_RDMA, NULL},
    [HCCP_MEMFD_SEG_UB]
    = {"hccp_st_ub", HCCP_MEMFD_SLOT_UB, HCCP_MEMFD_ENV_UB, HCCP_SHM_SIZE_UB, HCCP_SHM_MAGIC_UB, NULL},
    [HCCP_MEMFD_SEG_BUF]
    = {"hccp_st_buf", HCCP_MEMFD_SLOT_BUF, HCCP_MEMFD_ENV_BUF, HCCP_SHM_SIZE_BUF, HCCP_SHM_MAGIC_BUF,
       HCCP_BUF_SHM_ADDR},
    [HCCP_MEMFD_SEG_NOTIFY]
    = {"hccp_st_notify", HCCP_MEMFD_SLOT_NOTIFY, HCCP_MEMFD_ENV_NOTIFY, HCCP_SHM_SIZE_NOTIFY, 0, NULL},
    [HCCP_MEMFD_SEG_LITE]
    = {"hccp_st_lite", HCCP_MEMFD_SLOT_LITE, HCCP_MEMFD_ENV_LITE, HCCP_SHM_SIZE_LITE, HCCP_SHM_MAGIC_LITE, NULL},
    /* The consolidated DEVMEM segment replaces the former 64 per-phy memfds:
     * one object, whole-segment mapping at the fixed DVA window (a per-zone
     * mapping is impossible -- zone stride is 16 pages + 256 bytes, so only
     * every 16th zone offset is page-aligned; see hccp_shm_layout.h). */
    [HCCP_MEMFD_SEG_DEVMEM]
    = {"hccp_st_devmem", HCCP_MEMFD_SLOT_DEVMEM, HCCP_MEMFD_ENV_DEVMEM, HCCP_SHM_SIZE_DEVMEM_SEG,
       HCCP_SHM_MAGIC_DEVMEM_SEG, HCCP_DEVMEM_DVA_ADDR},
    /* H2D is the process/readiness registry: a born-zero table with no lock
     * and no magic, exactly like NOTIFY -- the run's zero page is the state. */
    [HCCP_MEMFD_SEG_H2D] = {"hccp_st_h2d", HCCP_MEMFD_SLOT_H2D, HCCP_MEMFD_ENV_H2D, HCCP_SHM_SIZE_H2D, 0, NULL},
};

/* Raw syscall keeps the core independent of the glibc wrapper version; the fd
 * starts close-on-exec, and publishing dup2()s it onto its fixed slot (dup2
 * clears CLOEXEC on the target), so exec'd device processes inherit exactly
 * the agreed slot set and nothing else. */
static int hccp_memfd_new(const char* name) { return (int)syscall(SYS_memfd_create, name, (unsigned int)MFD_CLOEXEC); }

/* Resolve the published fd slot for a segment. Returns the fd, -1 when not
 * published in this process (only the precreate constructor may create), or
 * -2 when the env entry is corrupt or the descriptor does not validate: a
 * descendant that cannot trust its inheritance fails fast instead. */
static int hccp_memfd_resolve(const hccp_memfd_segdesc_t* seg)
{
    const char* val = getenv(seg->env);
    if (val == NULL || val[0] == '\0') {
        return -1;
    }
    char* end = NULL;
    long slot = strtol(val, &end, 10);
    if (end == val || *end != '\0' || slot < 0 || slot > (1 << 20)) {
        return -2;
    }
    struct stat st;
    if (fstat((int)slot, &st) != 0 || !S_ISREG(st.st_mode) || (size_t)st.st_size != seg->size) {
        return -2;
    }
    return (int)slot;
}

/* Map a published memfd. Fixed-window segments use MAP_FIXED_NOREPLACE so a
 * window already mapped in this process (fork-inherited mapping, or another
 * infra copy's attach) is validated and reused instead of silently replaced;
 * the magic check separates our window from a foreign mapping. */
static void* hccp_memfd_map(int fd, const hccp_memfd_segdesc_t* seg)
{
    void* hint = seg->fixed;
    int flags = MAP_SHARED | ((hint != NULL) ? MAP_FIXED_NOREPLACE : 0);
    void* p = mmap(hint, seg->size, PROT_READ | PROT_WRITE, flags, fd, 0);
    if (p == MAP_FAILED) {
        if (hint != NULL && errno == EEXIST && seg->magic != 0 && *(volatile uint32_t*)hint == seg->magic) {
            return hint;
        }
        return NULL;
    }
    return p;
}

/* Segment-specific initialization. Runs only in the creator, before the
 * segment is published; memfd pages are born zero, so only the header words
 * need writing. Magic goes last, behind a release barrier. */
static void hccp_memfd_seg_init(int segId, void* p)
{
    switch (segId) {
        case HCCP_MEMFD_SEG_RDMA: {
            hccp_rdma_seg_t* s = (hccp_rdma_seg_t*)p;
            (void)hccp_shm_lock_init(&s->lock);
            s->creator_pid = getpid();
            s->next_seq = 1;
            s->next_qp_num = 1;
            __sync_synchronize();
            s->magic = HCCP_SHM_MAGIC_RDMA;
            break;
        }
        case HCCP_MEMFD_SEG_UB: {
            hccp_ub_seg_t* s = (hccp_ub_seg_t*)p;
            (void)hccp_shm_lock_init(&s->lock);
            s->creator_pid = getpid();
            s->next_seq = 1;
            s->next_jetty_id = 1;
            __sync_synchronize();
            s->magic = HCCP_SHM_MAGIC_UB;
            break;
        }
        case HCCP_MEMFD_SEG_BUF: {
            hccp_buf_seg_t* s = (hccp_buf_seg_t*)p;
            (void)hccp_shm_lock_init(&s->lock);
            __sync_synchronize();
            s->magic = HCCP_SHM_MAGIC_BUF;
            break;
        }
        case HCCP_MEMFD_SEG_NOTIFY:
            /* Born-zero CAS registry: nothing to initialize. */
            break;
        case HCCP_MEMFD_SEG_H2D:
            /* Born-zero process/readiness table: nothing to initialize. */
            break;
        case HCCP_MEMFD_SEG_LITE: {
            /* Robust from birth: the only lock site (the rdma_lite rendezvous
             * pairing in ascend_rdma_lite_stub.c) flips to the lock helpers in
             * the same batch, so attribute and site switch atomically (the
             * timing rule in the file header). count/qp_num stay born-zero. */
            hccp_lite_seg_t* s = (hccp_lite_seg_t*)p;
            (void)hccp_shm_lock_init(&s->lock);
            s->creator_pid = getpid();
            __sync_synchronize();
            s->magic = HCCP_SHM_MAGIC_LITE;
            break;
        }
        case HCCP_MEMFD_SEG_DEVMEM: {
            /* One consolidated segment: the master header plus 64 legacy-shaped
             * zones. Every lock (master + zones) is robust -- all lock sites are
             * infra-internal. The master magic goes last, behind every zone's
             * initialization, so a published segment is always fully built. */
            hccp_devmem_seg_t* s = (hccp_devmem_seg_t*)p;
            (void)hccp_shm_lock_init(&s->hdr.lock);
            s->hdr.version = 1;
            for (int phy = 0; phy < HCCP_DEVMEM_MAX_PHY; phy++) {
                hccp_devmem_zone_t* z = (hccp_devmem_zone_t*)((uint8_t*)p + HCCP_DEVMEM_SEG_ZONE_OFF(phy));
                (void)hccp_shm_lock_init(&z->hdr.lock);
                z->hdr.rsv = (uint32_t)phy; /* zone identity pin, structural now */
                __sync_synchronize();
                z->hdr.magic = HCCP_SHM_MAGIC_DEVMEM;
            }
            __sync_synchronize();
            s->hdr.magic = HCCP_SHM_MAGIC_DEVMEM_SEG;
            break;
        }
        default:
            break;
    }
}

/* Publish: dup2 onto the fixed slot (clears CLOEXEC there, survives execv)
 * and record the slot in the env. Call only after full initialization. The
 * raw fd stays open; the caller closes it once publishing is done. */
static int hccp_memfd_publish(int fd, const hccp_memfd_segdesc_t* seg)
{
    char buf[16];
    /* Vacancy probe: dup2 would silently close whatever sits on the target
     * slot. At constructor time (library load) the 900+ block is legitimately
     * empty in any normal launch, so an occupied slot is an anomaly -- fail
     * loudly instead of clobbering an unknown descriptor. */
    if (fcntl(seg->slot, F_GETFD) != -1 || errno != EBADF) {
        fprintf(stderr, "[hccp-st-infra] fd slot %d for %s not vacant; refusing to publish\n", seg->slot, seg->name);
        return -1;
    }
    if (dup2(fd, seg->slot) < 0) {
        return -1;
    }
    (void)snprintf(buf, sizeof(buf), "%d", seg->slot);
    if (setenv(seg->env, buf, 1) != 0) {
        return -1;
    }
    return 0;
}

/* Create, initialize and publish one segment. Returns 0 on success. The
 * creator drops its mapping afterwards: lazy users remap through the env, so
 * the constructor is never the reason a window stays occupied. */
static int hccp_memfd_ensure(int segId)
{
    const hccp_memfd_segdesc_t* seg = &g_hccp_memfd_segs[segId];
    int fd = hccp_memfd_resolve(seg);
    if (fd >= 0) {
        return 0; /* published and validated: lazy users map on demand */
    }
    if (fd == -2) {
        return -1; /* corrupt inheritance: fail fast, never re-create */
    }
    fd = hccp_memfd_new(seg->name);
    if (fd < 0) {
        return -1;
    }
    if (ftruncate(fd, (off_t)seg->size) != 0) {
        (void)close(fd);
        return -1;
    }
    void* p = hccp_memfd_map(fd, seg);
    if (p == NULL) {
        (void)close(fd);
        return -1;
    }
    hccp_memfd_seg_init(segId, p);
    int rc = hccp_memfd_publish(fd, seg);
    (void)close(fd); /* the slot holds its own reference from here on */
    (void)munmap(p, seg->size);
    return (rc == 0) ? 0 : -1;
}

/* Attach to a published segment: map it and belt-check the magic. Creation is
 * the constructor's job alone -- this path never creates, so a missing env can
 * only surface as a visible NULL, never as a second world. */
static void* hccp_memfd_segment(int segId)
{
    const hccp_memfd_segdesc_t* seg = &g_hccp_memfd_segs[segId];
    int fd = hccp_memfd_resolve(seg);
    if (fd < 0) {
        return NULL;
    }
    void* p = hccp_memfd_map(fd, seg);
    if (p == NULL) {
        return NULL;
    }
    if (seg->magic != 0 && *(volatile uint32_t*)p != seg->magic) {
        return NULL; /* unreachable with init-before-publish; fd mix-up belt */
    }
    return p;
}

/* --------------------------------------------------------------------------------------------
 * Precreate constructor: the common ancestor of the whole process tree
 *
 * Lives in every infra-bearing library and therefore runs at the earliest
 * library load in every process:
 *   - runner (binary load of the first infra-bearing .so): env unset, so this
 *     is the single creation point of the run; everything forked later
 *     inherits the published slots and the env.
 *   - device process (post-exec): env inherited, validation only.
 *   - standalone binary (no env): creates its own isolated world.
 * Host children never re-run it: fork does not execute constructors.
 * ------------------------------------------------------------------------------------------ */

__attribute__((constructor(101))) static void hccp_memfd_precreate_ctor(void)
{
    for (int i = 0; i < HCCP_MEMFD_SEG_COUNT; i++) {
        if (hccp_memfd_ensure(i) != 0) {
            /* stderr only: no logging framework exists at constructor time */
            (void)fprintf(stderr, "[hccp_memfd] precreate %s failed\n", g_hccp_memfd_segs[i].name);
        }
    }
    /* DEVMEM rides the same table loop now: one segment, one env, the
     * standard ensure() create-or-validate decision. The former 64-slot
     * family special case (whole-family decision through the shared BASE
     * env) is gone with the family itself. */
}

/* --------------------------------------------------------------------------------------------
 * Compat surface
 * ------------------------------------------------------------------------------------------ */

void* hccp_shm_open(key_t key, size_t size, uint32_t magic)
{
    int segId = -1;
    switch (key) {
        case HCCP_SHM_KEY_RDMA:
            segId = HCCP_MEMFD_SEG_RDMA;
            break;
        case HCCP_SHM_KEY_UB:
            segId = HCCP_MEMFD_SEG_UB;
            break;
        case HCCP_SHM_KEY_LITE:
            segId = HCCP_MEMFD_SEG_LITE;
            break;
        case HCCP_SHM_KEY_H2D:
            segId = HCCP_MEMFD_SEG_H2D;
            break;
        default:
            /* Unknown keys fail fast instead of guessing. */
            return NULL;
    }
    const hccp_memfd_segdesc_t* seg = &g_hccp_memfd_segs[segId];
    if (size != seg->size) {
        return NULL; /* caller compiled against a different layout */
    }
    if (magic != 0 && magic != seg->magic) {
        return NULL; /* magic disagreement: layout drift */
    }
    return hccp_memfd_segment(segId);
}

/* Reset the LITE QP rendezvous for the next case: pairing count back to zero,
 * both published qp_num slots cleared. The rendezvous is per-case state that
 * merely LIVES in a per-run segment: the old SysV world reset it implicitly
 * (the previous case's creator was dead by the next attach, so the kill-probe
 * recycled the segment), and case boundaries are the clean memfd-era
 * equivalent. Call only from the runner between cases -- no lite users exist
 * at that point, the previous case's hosts have been reaped. */
void hccp_lite_rendezvous_reset(void)
{
    hccp_lite_seg_t* s = (hccp_lite_seg_t*)hccp_memfd_segment(HCCP_MEMFD_SEG_LITE);
    if (s == NULL) {
        return;
    }
    if (hccp_shm_lock_acquire(&s->lock) != 0) {
        return; /* segment not recoverable: pairing will fail visibly anyway */
    }
    s->count = 0;
    s->qp_num[0] = 0;
    s->qp_num[1] = 0;
    __sync_synchronize();
    (void)hccp_shm_lock_release(&s->lock);
}

int hccp_pkt_enqueue(
    pthread_mutex_t* lock, hccp_pkt_slot_t* slots, uint32_t nslots, uint64_t* next_seq, uint32_t dst_id,
    uint32_t src_id, const uint8_t* data, uint32_t len)
{
    if (len > HCCP_PKT_MAX_MSG)
        len = HCCP_PKT_MAX_MSG;

    if (hccp_shm_lock_acquire(lock) != 0) {
        return -1; /* ring not recoverable: fail visibly instead of racing */
    }
    int found = -1;
    for (uint32_t i = 0; i < nslots; i++) {
        if (!slots[i].in_use) {
            found = (int)i;
            break;
        }
    }
    if (found < 0) {
        (void)hccp_shm_lock_release(lock);
        return -1;
    }
    hccp_pkt_slot_t* s = &slots[found];
    s->dst_id = dst_id;
    s->src_id = src_id;
    s->byte_len = len;
    s->seq = (*next_seq)++;
    if (len > 0 && data != NULL)
        memcpy(s->data, data, len);
    __sync_synchronize();
    s->in_use = 1;
    (void)hccp_shm_lock_release(lock);
    return 0;
}

int hccp_pkt_dequeue(
    pthread_mutex_t* lock, hccp_pkt_slot_t* slots, uint32_t nslots, uint32_t dst_id, hccp_pkt_slot_t* out)
{
    if (hccp_shm_lock_acquire(lock) != 0) {
        return 0; /* ring not recoverable: report "no packet" */
    }
    int pick = -1;
    uint64_t min_seq = (uint64_t)-1;
    for (uint32_t i = 0; i < nslots; i++) {
        hccp_pkt_slot_t* s = &slots[i];
        if (s->in_use && s->dst_id == dst_id && s->seq < min_seq) {
            min_seq = s->seq;
            pick = (int)i;
        }
    }
    if (pick < 0) {
        (void)hccp_shm_lock_release(lock);
        return 0;
    }
    hccp_pkt_slot_t* s = &slots[pick];
    out->dst_id = s->dst_id;
    out->src_id = s->src_id;
    out->byte_len = s->byte_len;
    out->seq = s->seq;
    if (s->byte_len > 0)
        memcpy(out->data, s->data, s->byte_len);
    s->in_use = 0;
    s->byte_len = 0;
    (void)hccp_shm_lock_release(lock);
    return 1;
}

uint32_t hccp_alloc_unique_id(volatile uint32_t* next_id) { return __sync_fetch_and_add(next_id, 1); }

/* --------------------------------------------------------------------------------------------
 * BUF pool (fixed window 0x600000000000, robust lock)
 * ------------------------------------------------------------------------------------------ */

static hccp_buf_seg_t* g_buf_pool = NULL;

void hccp_buf_pool_init(void)
{
    if (g_buf_pool == NULL) {
        g_buf_pool = (hccp_buf_seg_t*)hccp_memfd_segment(HCCP_MEMFD_SEG_BUF);
    }
}

void* hccp_shm_alloc(size_t len)
{
    (void)len;
    if (g_buf_pool == NULL) {
        hccp_buf_pool_init();
    }
    if (g_buf_pool == NULL)
        return NULL;
    if (hccp_shm_lock_acquire(&g_buf_pool->lock) != 0)
        return NULL; /* pool not recoverable */
    void* ret = NULL;
    for (int i = 0; i < HCCP_BUF_SLOTS; i++) {
        if (!g_buf_pool->buf_in_use[i]) {
            g_buf_pool->buf_in_use[i] = 1;
            memset(g_buf_pool->slots[i], 0, HCCP_BUF_SIZE);
            ret = g_buf_pool->slots[i];
            break;
        }
    }
    (void)hccp_shm_lock_release(&g_buf_pool->lock);
    return ret;
}

void hccp_shm_free(void* va)
{
    if (va == NULL)
        return;
    /* Lazy attach, not just lazy init: this copy of the infra may never have
     * allocated, but a free from another copy is still valid for the shared
     * pool (the fixed window makes the VA process-global). */
    if (g_buf_pool == NULL) {
        hccp_buf_pool_init();
    }
    if (g_buf_pool == NULL)
        return;
    uint8_t* slots0 = &g_buf_pool->slots[0][0];
    uint8_t* p = (uint8_t*)va;
    if (p < slots0)
        return;
    size_t off = (size_t)(p - slots0);
    if (off % HCCP_BUF_SIZE != 0)
        return;
    int idx = (int)(off / HCCP_BUF_SIZE);
    if (idx < 0 || idx >= HCCP_BUF_SLOTS)
        return;
    if (hccp_shm_lock_acquire(&g_buf_pool->lock) != 0)
        return; /* pool not recoverable */
    if (g_buf_pool->buf_in_use[idx]) {
        g_buf_pool->buf_in_use[idx] = 0;
    }
    (void)hccp_shm_lock_release(&g_buf_pool->lock);
}

/* --------------------------------------------------------------------------------------------
 * DEVMEM consolidated segment (single memfd, whole-segment mapping at the
 * fixed DVA window 0x500000000000, robust master + zone locks)
 *
 * One segment holds the master header plus 64 legacy-shaped zones; a zone is
 * addressed structurally (SEG_ZONE_OFF), never by a per-zone mapping: zone
 * stride is 16 pages + 256 bytes, so only every 16th zone offset is
 * page-aligned (hccp_shm_layout.h). Every DVA therefore shifts against the
 * old per-phy window layout -- consumers classify/route addresses through
 * hccp_devmem_dva_decode() instead of the legacy [DVA+HDR, DVA+end) range.
 * ------------------------------------------------------------------------------------------ */

static hccp_devmem_seg_t* g_devmem_seg = NULL;

/* Whole-segment window attach. Idempotent: a second call resolves to the
 * already-mapped window (MAP_FIXED_NOREPLACE + master-magic reuse). */
static hccp_devmem_seg_t* hccp_devmem_seg_attach(void)
{
    if (g_devmem_seg != NULL) {
        return g_devmem_seg;
    }
    g_devmem_seg = (hccp_devmem_seg_t*)hccp_memfd_segment(HCCP_MEMFD_SEG_DEVMEM);
    return g_devmem_seg;
}

static hccp_devmem_zone_t* hccp_devmem_zone(int phyId)
{
    if (phyId < 0 || phyId >= HCCP_DEVMEM_MAX_PHY) {
        return NULL;
    }
    hccp_devmem_seg_t* s = hccp_devmem_seg_attach();
    if (s == NULL) {
        return NULL;
    }
    return (hccp_devmem_zone_t*)((uint8_t*)s + HCCP_DEVMEM_SEG_ZONE_OFF(phyId));
}

void* hccp_devmem_alloc(int phyId, size_t len)
{
    if (len == 0 || len > HCCP_DEVMEM_SIZE) {
        return NULL;
    }
    hccp_devmem_zone_t* z = hccp_devmem_zone(phyId);
    if (z == NULL) {
        return NULL;
    }
    void* ret = NULL;
    if (hccp_shm_lock_acquire(&z->hdr.lock) != 0) {
        return NULL; /* zone not recoverable */
    }
    for (int i = 0; i < HCCP_DEVMEM_SLOTS; i++) {
        if (!z->hdr.buf_in_use[i]) {
            z->hdr.buf_in_use[i] = 1;
            uint8_t* slot = (uint8_t*)z + HCCP_DEVMEM_SLOT_OFF(i);
            memset(slot, 0, HCCP_DEVMEM_SIZE);
            ret = slot;
            break;
        }
    }
    (void)hccp_shm_lock_release(&z->hdr.lock);
    return ret;
}

void hccp_devmem_free(int phyId, void* dva)
{
    if (phyId < 0 || phyId >= HCCP_DEVMEM_MAX_PHY || dva == NULL) {
        return;
    }
    /* Decode-first: the layout predicate is the single authority for slot
     * arithmetic; rejecting foreign zones keeps the caller's own-zone free
     * semantics of the per-phy era. */
    int phy = 0;
    int slot = 0;
    if (!hccp_devmem_dva_decode((uint64_t)(uintptr_t)dva, &phy, &slot) || phy != phyId) {
        return;
    }
    hccp_devmem_zone_t* z = hccp_devmem_zone(phyId);
    if (z == NULL) {
        return;
    }
    if (hccp_shm_lock_acquire(&z->hdr.lock) != 0) {
        return; /* zone not recoverable */
    }
    z->hdr.buf_in_use[slot] = 0;
    (void)hccp_shm_lock_release(&z->hdr.lock);
}

void* hccp_devmem_map_host(int phyId, size_t* lenOut)
{
    /* The host view of the device-memory window: the whole segment is mapped
     * at the fixed DVA base, so the window base itself is the mapping
     * consumers translate through (hva = base + (dva - DVA_ADDR)). phyId only
     * decides whether this process may ask at all.
     * lenOut keeps the legacy per-phy slot-area span (one zone's slots, not
     * the whole window) purely for unmap bookkeeping -- the sole consumer
     * uses it as metadata for the (intentional no-op) unmap call. Do NOT use
     * it as a validity bound for the returned base. */
    if (phyId < 0 || phyId >= HCCP_DEVMEM_MAX_PHY) {
        return NULL;
    }
    if (hccp_devmem_seg_attach() == NULL) {
        return NULL;
    }
    if (lenOut != NULL) {
        *lenOut = (size_t)HCCP_DEVMEM_SLOTS * HCCP_DEVMEM_SIZE;
    }
    return (void*)HCCP_DEVMEM_DVA_ADDR;
}

void hccp_devmem_unmap_host(void* hva)
{
    /* Intentional no-op: the old private anywhere-mapping is gone; the base
     * handed out is the process-lifetime DVA window shared by every devmem
     * user, and unmapping it would pull the floor from under them. The
     * kernel reclaims it at process exit. */
    (void)hva;
}

void hccp_devmem_pool_attach(int phyId) { (void)hccp_devmem_zone(phyId); }

void hccp_devmem_destroy_all(void)
{
    /* memfd pools need no destruction (kernel refcount); the dead legacy SysV
     * keys are removed once per run by the transition helpers below. */
}

/* --------------------------------------------------------------------------------------------
 * Segment-role naming (urma.c owns the register/write/read trio)
 * ------------------------------------------------------------------------------------------ */

const char* hccp_seg_rw_name(hccp_seg_rw_kind_t k)
{
    switch (k) {
        case HCCP_SEG_RW_WRITE:
            return "WRITE";
        case HCCP_SEG_RW_READ:
            return "READ";
        case HCCP_SEG_RW_SEND:
        default:
            return "SEND";
    }
}

/* --------------------------------------------------------------------------------------------
 * NOTIFY registry (born-zero CAS slots; segment lives on memfd)
 * ------------------------------------------------------------------------------------------ */

static hccp_notify_reg_t* hccp_notify_tab_get(void)
{
    static hccp_notify_reg_t* tab = NULL;
    if (tab == NULL) {
        tab = (hccp_notify_reg_t*)hccp_memfd_segment(HCCP_MEMFD_SEG_NOTIFY);
    }
    return tab;
}

int hccp_notify_publish(pid_t hostPid, uint64_t notifyVa)
{
    if (hostPid <= 0 || notifyVa == 0) {
        return -1;
    }
    hccp_notify_reg_t* tab = hccp_notify_tab_get();
    if (tab == NULL) {
        return -1;
    }
    for (int attempt = 0; attempt < 2; attempt++) {
        for (int i = 0; i < HCCP_NOTIFY_MAX_HOST; i++) {
            pid_t owner = tab[i].hostPid;
            if (owner == hostPid) {
                /* Refresh the entry already published by this process. */
                tab[i].notifyVa = notifyVa;
                __sync_synchronize();
                return 0;
            }
            /* Free slot, or slot left behind by a dead process of an earlier
             * test (the registry survives across tests of one binary run). */
            int reclaimable = (owner == 0) || (kill(owner, 0) != 0 && errno == ESRCH);
            if (!reclaimable) {
                continue;
            }
            /* The two host test processes publish concurrently, so the slot
             * must be claimed atomically. */
            if (__sync_bool_compare_and_swap(&tab[i].hostPid, owner, hostPid)) {
                tab[i].notifyVa = notifyVa;
                __sync_synchronize();
                return 0;
            }
        }
    }
    return -1;
}

uint64_t hccp_notify_query(pid_t peerPid)
{
    if (peerPid <= 0) {
        return 0;
    }
    hccp_notify_reg_t* tab = hccp_notify_tab_get();
    if (tab == NULL) {
        return 0;
    }
    for (int i = 0; i < HCCP_NOTIFY_MAX_HOST; i++) {
        /* notifyVa != 0 skips entries whose claim is still in flight. */
        if (tab[i].hostPid == peerPid && tab[i].notifyVa != 0) {
            return (uint64_t)tab[i].notifyVa;
        }
    }
    return 0;
}

/* --------------------------------------------------------------------------------------------
 * Legacy SysV transition helpers
 *
 * No segment lives on SysV anymore; these helpers exist only to RMID the dead
 * keys of the migrated segments as transition hygiene, so leftovers from
 * pre-memfd runs cannot accumulate.
 * ------------------------------------------------------------------------------------------ */

int hccp_shm_destroy(key_t key)
{
    int shmid = shmget(key, 0, 0666);
    if (shmid < 0) {
        return 0;
    }
    return (shmctl(shmid, IPC_RMID, NULL) == 0) ? 1 : 0;
}

/* Remove the SysV keys that no live code uses anymore (their segments moved
 * to memfd): plain hygiene against pre-migration leftovers, run once per run. */
static void hccp_shm_legacy_rmid_dead_keys(void)
{
    (void)hccp_shm_destroy(HCCP_SHM_KEY_RDMA);
    (void)hccp_shm_destroy(HCCP_SHM_KEY_UB);
    (void)hccp_shm_destroy(HCCP_SHM_KEY_BUF);
    (void)hccp_shm_destroy(HCCP_SHM_KEY_NOTIFY);
    (void)hccp_shm_destroy(HCCP_SHM_KEY_LITE);
    (void)hccp_shm_destroy(HCCP_SHM_KEY_H2D);
    for (int phy = 0; phy < HCCP_DEVMEM_MAX_PHY; phy++) {
        (void)hccp_shm_destroy(HCCP_DEVMEM_LEGACY_KEY(phy));
    }
}

void hccp_shm_destroy_all(void)
{
    /* Every segment is memfd now: nothing to destroy at run end -- the kernel
     * reclaims the segments when the run's last process exits. What remains
     * is the dead-key hygiene against pre-migration SysV leftovers. */
    hccp_shm_legacy_rmid_dead_keys();
}

void hccp_shm_purge_all(void)
{
    /* Same hygiene at run start: the memfd segments are born zero per run and
     * need no run-boundary wipe. */
    hccp_shm_legacy_rmid_dead_keys();
}

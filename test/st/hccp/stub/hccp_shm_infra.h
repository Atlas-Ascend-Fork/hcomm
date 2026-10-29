/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_SHM_INFRA_H
#define HCCP_SHM_INFRA_H

/* hccp_shm_infra.h -- runtime API over the ST shared-memory segments.
 *
 * The segment LIVES on an anonymous memfd since t3: the process tree's common
 * ancestor (the first infra-bearing library loaded by the runner, via the
 * precreate constructor in hccp_shm_infra.c) creates every segment, fully
 * initializes it, and publishes it on a fixed fd slot plus a matching
 * HCCP_ST_MEMFD_* env var. Host children inherit both through plain fork,
 * device processes through fork+execv (the published slots carry no CLOEXEC).
 * The kernel drops a segment when its last fd closes, so liveness probing,
 * stale recycling and run-boundary purges are gone for the migrated segments.
 *
 * Byte layouts, SysV transition keys and the lock discipline live in the
 * single layout authority, hccp_shm_layout.h, which this header includes:
 * the key/size/magic macros and the shared typedefs below resolve to it. */

#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libgen.h>
#include <limits.h>
#include <pthread.h>

#include "hccp_shm_layout.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* --------------------------------------------------------------------------------------------
 * memfd slot conventions (runtime identity of the anonymous segments)
 *
 * The precreate constructor publishes each segment with dup2() onto the fixed
 * slot below -- dup2 clears close-on-exec on the target descriptor, so the
 * exec'd device processes inherit exactly the agreed slot set -- and records
 * the slot number in the matching HCCP_ST_MEMFD_* env var. The env var is the
 * runtime authority; the fixed slot is the convention a debugger can rely on
 * (/proc/<pid>/fd shows every slot as memfd:<name>).
 *
 * Every segment of the numbering table is live: the migration completed
 * batch by batch (LITE, DEVMEM, then H2D), and the block now carries the
 * full runtime identity of a run.
 * ------------------------------------------------------------------------------------------ */

/* Compact block 900..969 (aligned with the t1 spike report): the scalar
 * segments first, then the consolidated DEVMEM segment, 907..969 reserved
 * (they held the pre-consolidation per-phy DEVMEM family). The constructor
 * runs at library-load time, when the process holds only the standard streams
 * plus loader-transient descriptors, so the block is far above anything open
 * at that point; dup2 keeps every number occupied for the process's whole
 * lifetime, so later runtime opens can never collide. The low ceiling is
 * equally load-bearing: every number must survive under RLIMIT_NOFILE=1024
 * (verified in the wild), which rules out any 1000+phyId scheme -- phy >= 24
 * would already push dup2 past the soft limit. */
#define HCCP_MEMFD_SLOT_H2D 900 /* process/readiness registry, born-zero table */
#define HCCP_MEMFD_SLOT_RDMA 901
#define HCCP_MEMFD_SLOT_UB 902
#define HCCP_MEMFD_SLOT_BUF 903
#define HCCP_MEMFD_SLOT_LITE 904
#define HCCP_MEMFD_SLOT_NOTIFY 905
#define HCCP_MEMFD_SLOT_DEVMEM 906 /* single consolidated segment: 64 zones in one mapping */

#define HCCP_MEMFD_ENV_H2D "HCCP_ST_MEMFD_H2D"
#define HCCP_MEMFD_ENV_RDMA "HCCP_ST_MEMFD_RDMA"
#define HCCP_MEMFD_ENV_UB "HCCP_ST_MEMFD_UB"
#define HCCP_MEMFD_ENV_BUF "HCCP_ST_MEMFD_BUF"
#define HCCP_MEMFD_ENV_LITE "HCCP_ST_MEMFD_LITE"
#define HCCP_MEMFD_ENV_NOTIFY "HCCP_ST_MEMFD_NOTIFY"
#define HCCP_MEMFD_ENV_DEVMEM "HCCP_ST_MEMFD_DEVMEM"

/* --------------------------------------------------------------------------------------------
 * Run identity (HCCP_ST_RUN_ID): which concurrent runner this tree belongs to
 *
 * Two runners may share one machine when their HDC port blocks and log trees
 * are disjoint. N selects both: ports move to 9999+phyId+N*10000 (see
 * ascend_hal_stub.h) and logs move to <root>/ascend/log/run_<N> (resolved by
 * hccp_st_log_root below -- THE single construction point, shared by the dlog
 * writer and the runner's verdict scan).
 *
 * Parsing lives here so the port and log sides follow ONE rule set. The env
 * is stable after process start -- every setenv in this framework happens in
 * constructors or before fork -- so plain getenv stays race-free. static
 * inline keeps every .so self-contained (the dlog stub does not link the shm
 * infra) and does not trip -Wunused-function in TUs that only want the
 * layouts. Documented valid range is 0..2: N*10000+9999+phyId must stay
 * below the typical Linux ip_local_port_range floor (32768), and N>=6 would
 * overflow the 16-bit port space outright. Malformed or out-of-range values
 * warn (once per TU) and behave as 0: the run-lock guard then serializes the
 * accidental collision instead of a silent port ambush.
 * ------------------------------------------------------------------------------------------ */

static inline int hccp_st_run_id(void)
{
    const char* v = getenv("HCCP_ST_RUN_ID");
    if (v == NULL || v[0] == '\0') {
        return 0;
    }
    char* end = NULL;
    long n = strtol(v, &end, 10);
    if (end == v || *end != '\0' || n < 0 || n > 2) {
        static int warned = 0;
        if (!warned) {
            warned = 1;
            (void)fprintf(stderr, "[hccp_st] HCCP_ST_RUN_ID='%s' invalid (expect 0..2), using 0\n", v);
        }
        return 0;
    }
    return (int)n;
}

/* THE single construction point of the ST log root. The runner's verdict scan
 * (main.cc) and the dlog writer (dlog_stub.c) both resolve their tree through
 * this function: two private copies could drift apart, and a drifted pair
 * silently blinds the log-error verdict component (logs written where the
 * scanner never looks). Slot suffix is decimal run_<N> without leading zeros.
 * The tree itself is created lazily by the first written log line, so an
 * absent root means "nothing was ever logged". Returns 0 on success, -1 when
 * the executable path cannot be resolved or the result would not fit. */
static inline int hccp_st_log_root(char* out, size_t sz, int runId)
{
    char exePath[PATH_MAX];
    char* exeDir = NULL;
    ssize_t count;
    int ret;

    count = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (count <= 0) {
        return -1;
    }
    exePath[count] = '\0';
    exeDir = dirname(exePath);
    if (runId > 0) {
        ret = snprintf(out, sz, "%s/../ascend/log/run_%d", exeDir, runId);
    } else {
        ret = snprintf(out, sz, "%s/../ascend/log", exeDir);
    }
    if (ret < 0 || (size_t)ret >= sz) {
        return -1;
    }
    return 0;
}

/* --------------------------------------------------------------------------------------------
 * Segment access (compat surface; the key parameter selects the segment)
 * ------------------------------------------------------------------------------------------ */

/* Role of a registered memory segment in the urma data plane (urma.c owns the
 * register/write/read trio and maps these onto the urma completion kinds). */
typedef enum {
    HCCP_SEG_RW_SEND = 0,
    HCCP_SEG_RW_WRITE = 1,
    HCCP_SEG_RW_READ = 2,
} hccp_seg_rw_kind_t;

/* Attach to the segment selected by <key> (HCCP_SHM_KEY_RDMA / _UB / _LITE).
 * The size/magic arguments are cross-checked against the layout authority: a
 * mismatch means the caller was compiled against a different layout and gets
 * NULL instead of a corrupted segment. Returns the mapped segment or NULL. */
void* hccp_shm_open(key_t key, size_t size, uint32_t magic);

/* Reset the LITE QP rendezvous (count + published qp_num slots) for the next
 * case. The rendezvous is per-case state in a per-run segment; the runner
 * calls this from the test fixture's SetUp, where the previous case's hosts
 * are already reaped and no lite user exists. */
void hccp_lite_rendezvous_reset(void);

int hccp_pkt_enqueue(
    pthread_mutex_t* lock, hccp_pkt_slot_t* slots, uint32_t nslots, uint64_t* next_seq, uint32_t dst_id,
    uint32_t src_id, const uint8_t* data, uint32_t len);

int hccp_pkt_dequeue(
    pthread_mutex_t* lock, hccp_pkt_slot_t* slots, uint32_t nslots, uint32_t dst_id, hccp_pkt_slot_t* out);

uint32_t hccp_alloc_unique_id(volatile uint32_t* next_id);

void hccp_buf_pool_init(void);

void* hccp_shm_alloc(size_t len);

void hccp_shm_free(void* va);

/* Device-memory pools: one consolidated memfd segment (master header + 64
 * legacy-shaped zones) mapped whole at the fixed DVA window. Slot DVAs are
 * classified and routed through hccp_devmem_dva_decode() (hccp_shm_layout.h);
 * map_host hands out the window base itself, so a host address is always
 * base + (dva - DVA_ADDR), and unmap is a no-op (the window is
 * process-lifetime infrastructure). */
void* hccp_devmem_alloc(int phyId, size_t len);

void hccp_devmem_free(int phyId, void* dva);

void* hccp_devmem_map_host(int phyId, size_t* lenOut);

void hccp_devmem_unmap_host(void* hva);

void hccp_devmem_pool_attach(int phyId);

void hccp_devmem_destroy_all(void);

const char* hccp_seg_rw_name(hccp_seg_rw_kind_t k);

uint32_t hccp_seg_register(uint64_t va, size_t len);

int hccp_seg_write(uint32_t lkey, uint64_t dst_va, const void* src, uint32_t len);

int hccp_seg_read(uint32_t lkey, void* dst, uint64_t src_va, uint32_t len);

/* --------------------------------------------------------------------------------------------
 * Legacy SysV transition helpers (no live SysV segment remains)
 *
 * Every segment is memfd now. The purge/destroy entry points exist only to
 * remove the dead SysV keys of the migrated segments (RDMA/UB/BUF/NOTIFY/
 * LITE/H2D/DEVMEM) so leftovers from pre-memfd runs cannot accumulate.
 * ------------------------------------------------------------------------------------------ */

int hccp_shm_destroy(key_t key);

void hccp_shm_destroy_all(void);

void hccp_shm_purge_all(void);

/* Publish the local notify page base address into the shm registry.
 * Call before the QP connection is established. */
int hccp_notify_publish(pid_t hostPid, uint64_t notifyVa);

/* Look up the notify page base address published by the given host process.
 * Returns 0 when the peer has not published (or no registry exists). */
uint64_t hccp_notify_query(pid_t peerPid);

#if defined(__cplusplus)
}
#endif

#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* hccp_shm_layout.h -- single layout authority for every shared-memory segment of the
 * hccp ST framework: H2D, RDMA, UB, BUF, LITE, DEVMEM, NOTIFY.
 *
 * Today those layouts live as duck-typed copies scattered over the stub files:
 *   h2d_info_t              ascend_hal_stub.h     (ibv_h2d_info_t in ibverbs_stub.c)
 *   rdma_shm_t              ibverbs_stub.c
 *   urma_shm_t              urma.c                (urma_rdv_entry_t)
 *   hccp_buf_pool_t         hccp_shm_infra.c
 *   hccp_devmem_hdr_t       hccp_shm_infra.c
 *   rdma_lite_rendezvous_t  ascend_rdma_lite_stub.c
 *   hccp_pkt_slot_t         hccp_shm_infra.h
 *   hccp_notify_reg_t       hccp_shm_infra.h
 * A layout mismatch between two attaching processes silently corrupts cross-process
 * locks, so this header converges them: every definition below is byte-frozen against
 * the live copy it replaces, and the static_assert lattice turns any drift into a
 * compile error instead of a runtime surprise.
 *
 * Migration contract (consumers switch segment by segment, t3/t4):
 *  - The key/size/magic/dimension macros and the two shared typedefs (hccp_pkt_slot_t,
 *    hccp_notify_reg_t) mirror hccp_shm_infra.h token-for-token, so retiring the infra
 *    copies is a pure deletion. Until that swap lands, do NOT include this header into
 *    a translation unit that also includes hccp_shm_infra.h: the mirrored typedefs
 *    would be duplicate (distinct-type) definitions and fail to compile -- the drift
 *    alarm doing its job one last time.
 *  - Segment structs carry new canonical names (hccp_h2d_entry_t, hccp_rdma_seg_t, ...);
 *    the duck-typed originals stay untouched until their consumer switches over.
 *  - The DEVMEM single-segment 64-zone layout is reserved for the 64-key consolidation
 *    (t4, memfd era): one segment, one zone per phyId, each zone byte-identical to the
 *    legacy per-phy segment. The legacy key derivation stays as a transition macro.
 *
 * Platform freeze: every frozen size is byte-frozen per architecture against glibc.
 * The single arch-dependent fact is the size of pthread_mutex_t: 40 bytes on x86_64,
 * 48 bytes on aarch64 (8-byte alignment on both), which shifts every post-lock field
 * by 8 on aarch64. The canary assert below fires on any other platform; porting to a
 * new architecture means measuring the real layout and extending the frozen-value
 * block below deliberately -- never widen an assert to make it pass.
 */

#ifndef HCCP_SHM_LAYOUT_H
#define HCCP_SHM_LAYOUT_H

#include <stddef.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/types.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* Layout alarm: fires in every TU that includes this header, C and C++ alike. */
#if defined(__cplusplus)
#define HCCP_SHM_LAYOUT_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define HCCP_SHM_LAYOUT_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

/* Per-architecture frozen values. Every number below was measured against the real
 * struct layout of the named glibc/architecture pair; the asserts further down pin
 * the live structs to these constants. Keep both columns byte-exact. */
#if defined(__x86_64__)
#define HCCP_FROZEN_MUTEX_SZ 40 /* sizeof(pthread_mutex_t), x86_64 glibc */
#define HCCP_FROZEN_RDMA_NEXT_SEQ 48
#define HCCP_FROZEN_RDMA_NEXT_QP 56
#define HCCP_FROZEN_RDMA_SLOTS 64
#define HCCP_FROZEN_RDMA_SZ 4195904
#define HCCP_FROZEN_UB_NEXT_JETTY 56
#define HCCP_FROZEN_UB_RDV 60
#define HCCP_FROZEN_UB_SLOTS 1344
#define HCCP_FROZEN_UB_SZ 4197184
#define HCCP_FROZEN_BUF_IN_USE 48
#define HCCP_FROZEN_BUF_SLOTS 176
#define HCCP_FROZEN_BUF_SZ 131248
#define HCCP_FROZEN_LITE_COUNT 48
#define HCCP_FROZEN_LITE_QP 52
#define HCCP_FROZEN_LITE_SZ 64
#define HCCP_FROZEN_DEVMEM_HDR_SZ 112
#define HCCP_FROZEN_DEVMEM_IN_USE 48
#define HCCP_FROZEN_SEG_HDR_RSV 52 /* uint32_t rsv[] count padding the master header */
#elif defined(__aarch64__)
#define HCCP_FROZEN_MUTEX_SZ 48 /* sizeof(pthread_mutex_t), aarch64 glibc */
#define HCCP_FROZEN_RDMA_NEXT_SEQ 56
#define HCCP_FROZEN_RDMA_NEXT_QP 64
#define HCCP_FROZEN_RDMA_SLOTS 72
#define HCCP_FROZEN_RDMA_SZ 4195912
#define HCCP_FROZEN_UB_NEXT_JETTY 64
#define HCCP_FROZEN_UB_RDV 68
#define HCCP_FROZEN_UB_SLOTS 1352
#define HCCP_FROZEN_UB_SZ 4197192
#define HCCP_FROZEN_BUF_IN_USE 56
#define HCCP_FROZEN_BUF_SLOTS 184
#define HCCP_FROZEN_BUF_SZ 131256
#define HCCP_FROZEN_LITE_COUNT 56
#define HCCP_FROZEN_LITE_QP 60
#define HCCP_FROZEN_LITE_SZ 72
#define HCCP_FROZEN_DEVMEM_HDR_SZ 120
#define HCCP_FROZEN_DEVMEM_IN_USE 56
#define HCCP_FROZEN_SEG_HDR_RSV 50
#else
#error "hccp_shm_layout.h: shared-memory layout not frozen for this platform"
#endif

HCCP_SHM_LAYOUT_ASSERT(
    sizeof(pthread_mutex_t) == HCCP_FROZEN_MUTEX_SZ,
    "pthread_mutex_t size drifted from the per-arch glibc frozen value");

/* --------------------------------------------------------------------------------------------
 * SysV key space (frozen, hand-assigned, no run-id dimension)
 *   0x1234ABCD  H2D      0x1234ABCE  RDMA     0x1234ABCF  UB
 *   0x1234ABD0  BUF      0x1234ABD1  LITE
 *   0x1234ABD2 .. 0x1234AC11  DEVMEM[phyId 0..63]  (legacy per-phy pool)
 *   0x1234AC12 .. 0x1234AC1F  reserved gap
 *   0x1234AC20  NOTIFY   (must stay outside the devmem pool range)
 * ------------------------------------------------------------------------------------------ */

#define HCCP_SHM_KEY_H2D ((key_t)0x1234ABCD)
#define HCCP_SHM_KEY_RDMA ((key_t)0x1234ABCE)
#define HCCP_SHM_KEY_UB ((key_t)0x1234ABCF)
#define HCCP_SHM_KEY_BUF ((key_t)0x1234ABD0)
#define HCCP_SHM_KEY_LITE ((key_t)0x1234ABD1)

#define HCCP_SHM_SIZE_H2D ((size_t)7168)
#define HCCP_SHM_MAGIC_H2D ((uint32_t)0)

#define HCCP_SHM_SIZE_RDMA ((size_t)HCCP_FROZEN_RDMA_SZ)
#define HCCP_SHM_MAGIC_RDMA ((uint32_t)0x52444D41u)

#define HCCP_SHM_SIZE_UB ((size_t)HCCP_FROZEN_UB_SZ)
#define HCCP_SHM_MAGIC_UB ((uint32_t)0x55524D41u)

#define HCCP_SHM_SIZE_BUF ((size_t)HCCP_FROZEN_BUF_SZ)
#define HCCP_SHM_MAGIC_BUF ((uint32_t)0x55524D42u)

#define HCCP_SHM_SIZE_LITE ((size_t)HCCP_FROZEN_LITE_SZ)
#define HCCP_SHM_MAGIC_LITE ((uint32_t)0x524C4954u)

/* --------------------------------------------------------------------------------------------
 * The one lock type
 *
 * Every segment lock from now on is a process-shared robust mutex, and the helpers
 * below are the only sanctioned way to touch it (implementation landed with the
 * memfd core, t3). As of the t4 lock-flip batches every segment lock -- RDMA/UB/LITE
 * prefixes and the devmem zones included -- goes through these helpers: the layout
 * ships no non-robust lock sites anymore.
 * ------------------------------------------------------------------------------------------ */

typedef pthread_mutex_t hccp_shm_lock_t;

/* Initialize a fresh lock: PTHREAD_PROCESS_SHARED + PTHREAD_MUTEX_ROBUST.
 * Returns 0 on success, a pthread error number otherwise. */
int hccp_shm_lock_init(hccp_shm_lock_t* lock);

/* Acquire with dead-holder recovery. EOWNERDEAD is absorbed by marking the mutex
 * consistent again: the state these locks protect is self-healing (a slot bitmap
 * disturbed by a dead holder costs at most one slot of capacity), so the caller
 * simply sees success. ENOTRECOVERABLE (a previous recovery gave up) is returned
 * unchanged and means "treat the segment as unusable for this operation". */
int hccp_shm_lock_acquire(hccp_shm_lock_t* lock);

/* Release. Returns 0 on success, a pthread error number otherwise. */
int hccp_shm_lock_release(hccp_shm_lock_t* lock);

/* --------------------------------------------------------------------------------------------
 * Packet slot (shared by the RDMA and UB rings)
 *
 * Byte-identical to hccp_pkt_slot_t in hccp_shm_infra.h: 24-byte head (in_use flags the
 * slot under the prefix lock, seq orders dequeue by enqueue time) + 64KiB payload.
 * ------------------------------------------------------------------------------------------ */

#define HCCP_PKT_MAX_MSG (64 * 1024)

typedef struct {
    int in_use;
    uint32_t dst_id;
    uint32_t src_id;
    uint32_t byte_len;
    uint64_t seq;
    uint8_t data[HCCP_PKT_MAX_MSG];
} hccp_pkt_slot_t;

HCCP_SHM_LAYOUT_ASSERT(sizeof(hccp_pkt_slot_t) == 65560, "hccp_pkt_slot_t must stay 65560 bytes (24B head + 64KiB)");
HCCP_SHM_LAYOUT_ASSERT(offsetof(hccp_pkt_slot_t, seq) == 16, "hccp_pkt_slot_t.seq must stay at offset 16");
HCCP_SHM_LAYOUT_ASSERT(offsetof(hccp_pkt_slot_t, data) == 24, "hccp_pkt_slot_t.data must stay at offset 24");

/* --------------------------------------------------------------------------------------------
 * H2D -- process registry and readiness handshake (no magic, no lock)
 *
 * [host row 0..H2D_MAX_HOST-1][phy column 0..H2D_MAX_DEV-1]; only row HOST_ID_0 is used
 * today, host identity is the phy column. Frozen against h2d_info_t (ascend_hal_stub.h,
 * which already asserts 28B/7168) and ibv_h2d_info_t (ibverbs_stub.c, same bytes).
 * ------------------------------------------------------------------------------------------ */

#define HCCP_H2D_MAX_HOST 4 /* hal stub HOST_OS_MAX == ibverbs IBV_HOST_OS_MAX */
#define HCCP_H2D_MAX_DEV 64 /* hal stub MAX_DEV_ID == ibverbs IBV_H2D_MAX_DEV */

typedef struct {
    pid_t hostPid;            /* host child process, right after fork */
    pid_t devPid;             /* device child process, before exec */
    unsigned int hostLogicId; /* InitHdcId fills the logic<->phy map */
    unsigned int devLogicId;
    unsigned int hostPhyId;
    unsigned int devPhyId;
    volatile unsigned int deviceReady; /* device process sets 1 once the stack is up */
} hccp_h2d_entry_t;

typedef struct {
    hccp_h2d_entry_t table[HCCP_H2D_MAX_HOST][HCCP_H2D_MAX_DEV];
} hccp_h2d_seg_t;

HCCP_SHM_LAYOUT_ASSERT(sizeof(hccp_h2d_entry_t) == 28, "hccp_h2d_entry_t must stay 28 bytes (7 x 4B fields)");
HCCP_SHM_LAYOUT_ASSERT(offsetof(hccp_h2d_entry_t, hostPid) == 0, "hccp_h2d_entry_t.hostPid must stay at offset 0");
HCCP_SHM_LAYOUT_ASSERT(
    offsetof(hccp_h2d_entry_t, deviceReady) == 24, "hccp_h2d_entry_t.deviceReady must stay at offset 24");
HCCP_SHM_LAYOUT_ASSERT(sizeof(hccp_h2d_seg_t) == HCCP_SHM_SIZE_H2D, "H2D segment must stay 7168 bytes ([4][64] x 28B)");

/* --------------------------------------------------------------------------------------------
 * RDMA -- ibverbs data plane (ring only; QP/CQ/MR tables are process-local)
 *
 * Frozen against rdma_shm_t (ibverbs_stub.c): prefix of HCCP_FROZEN_RDMA_SLOTS bytes
 * (64 on x86_64 / 72 on aarch64), then the packet ring.
 * ------------------------------------------------------------------------------------------ */

#define HCCP_RDMA_RING_SLOTS 64 /* ibverbs RDMA_RING_SLOTS */

typedef struct {
    uint32_t magic;       /* HCCP_SHM_MAGIC_RDMA, written last by the creator */
    pid_t creator_pid;    /* creator liveness was the legacy stale probe */
    hccp_shm_lock_t lock; /* serializes enqueue/dequeue over the ring */
    uint64_t next_seq;    /* cross-process monotonic sequence allocator */
    uint32_t next_qp_num; /* cross-process QP number allocator */
    hccp_pkt_slot_t slots[HCCP_RDMA_RING_SLOTS];
} hccp_rdma_seg_t;

HCCP_SHM_LAYOUT_ASSERT(offsetof(hccp_rdma_seg_t, lock) == 8, "hccp_rdma_seg_t.lock must stay at offset 8");
HCCP_SHM_LAYOUT_ASSERT(
    offsetof(hccp_rdma_seg_t, next_seq) == HCCP_FROZEN_RDMA_NEXT_SEQ, "hccp_rdma_seg_t.next_seq offset drifted");
HCCP_SHM_LAYOUT_ASSERT(
    offsetof(hccp_rdma_seg_t, next_qp_num) == HCCP_FROZEN_RDMA_NEXT_QP, "hccp_rdma_seg_t.next_qp_num offset drifted");
HCCP_SHM_LAYOUT_ASSERT(
    offsetof(hccp_rdma_seg_t, slots) == HCCP_FROZEN_RDMA_SLOTS, "hccp_rdma_seg_t.slots offset drifted");
HCCP_SHM_LAYOUT_ASSERT(sizeof(hccp_rdma_seg_t) == HCCP_SHM_SIZE_RDMA, "RDMA segment size drifted (prefix + 64 slots)");

/* --------------------------------------------------------------------------------------------
 * UB -- urma data plane (rendezvous table + ring)
 *
 * Frozen against urma_shm_t / urma_rdv_entry_t (urma.c). The rdv entry duck-types the
 * urma API types as plain 4-byte values: urma_transport_mode_t is an enum and
 * urma_token_t wraps a single uint32_t, so the frozen wire shape is 5 x 4B either way.
 * ------------------------------------------------------------------------------------------ */

#define HCCP_UB_MAX_RDV 64    /* urma URMA_MAX_RDV */
#define HCCP_UB_RING_SLOTS 64 /* urma URMA_RING_SLOTS */

typedef struct {
    int in_use;          /* claimed under the segment lock */
    uint32_t jetty_id;   /* cross-process unique id from next_jetty_id */
    uint32_t trans_mode; /* urma_transport_mode_t (4-byte enum, urma_types.h) */
    uint32_t token;      /* urma_token_t.token (struct wrapping one uint32_t) */
    pid_t owner_pid;     /* publish owner; dead-owner entries are reclaimable */
} hccp_ub_rdv_entry_t;

typedef struct {
    uint32_t magic; /* HCCP_SHM_MAGIC_UB, written last by the creator */
    pid_t creator_pid;
    hccp_shm_lock_t lock;   /* serializes rdv table + ring + id allocators */
    uint64_t next_seq;      /* cross-process monotonic sequence allocator */
    uint32_t next_jetty_id; /* cross-process unique jetty id allocator */
    hccp_ub_rdv_entry_t rdv[HCCP_UB_MAX_RDV];
    hccp_pkt_slot_t slots[HCCP_UB_RING_SLOTS];
} hccp_ub_seg_t;

HCCP_SHM_LAYOUT_ASSERT(sizeof(hccp_ub_rdv_entry_t) == 20, "hccp_ub_rdv_entry_t must stay 20 bytes (5 x 4B fields)");
HCCP_SHM_LAYOUT_ASSERT(
    offsetof(hccp_ub_rdv_entry_t, owner_pid) == 16, "hccp_ub_rdv_entry_t.owner_pid must stay at offset 16");
HCCP_SHM_LAYOUT_ASSERT(
    offsetof(hccp_ub_seg_t, next_jetty_id) == HCCP_FROZEN_UB_NEXT_JETTY, "hccp_ub_seg_t.next_jetty_id offset drifted");
HCCP_SHM_LAYOUT_ASSERT(offsetof(hccp_ub_seg_t, rdv) == HCCP_FROZEN_UB_RDV, "hccp_ub_seg_t.rdv offset drifted");
HCCP_SHM_LAYOUT_ASSERT(offsetof(hccp_ub_seg_t, slots) == HCCP_FROZEN_UB_SLOTS, "hccp_ub_seg_t.slots offset drifted");
HCCP_SHM_LAYOUT_ASSERT(sizeof(hccp_ub_seg_t) == HCCP_SHM_SIZE_UB, "UB segment size drifted (prefix + rdv + 64 slots)");

/* --------------------------------------------------------------------------------------------
 * BUF -- global 4KiB buffer pool, mapped at the fixed VA 0x600000000000
 *
 * Frozen against hccp_buf_pool_t (hccp_shm_infra.c). The only segment whose lock is
 * already robust (dead-holder recovery via EOWNERDEAD + pthread_mutex_consistent, the
 * pattern hccp_shm_lock_acquire generalizes to every other segment).
 * ------------------------------------------------------------------------------------------ */

#define HCCP_BUF_SHM_ADDR ((void*)0x600000000000ULL)
#define HCCP_BUF_SLOTS 32
#define HCCP_BUF_SIZE 4096

typedef struct {
    uint32_t magic;                      /* HCCP_SHM_MAGIC_BUF, written last */
    hccp_shm_lock_t lock;                /* process-shared + ROBUST */
    uint32_t buf_in_use[HCCP_BUF_SLOTS]; /* slot bitmap under the lock */
    uint8_t slots[HCCP_BUF_SLOTS][HCCP_BUF_SIZE];
} hccp_buf_seg_t;

HCCP_SHM_LAYOUT_ASSERT(offsetof(hccp_buf_seg_t, lock) == 8, "hccp_buf_seg_t.lock must stay at offset 8");
HCCP_SHM_LAYOUT_ASSERT(
    offsetof(hccp_buf_seg_t, buf_in_use) == HCCP_FROZEN_BUF_IN_USE, "hccp_buf_seg_t.buf_in_use offset drifted");
HCCP_SHM_LAYOUT_ASSERT(offsetof(hccp_buf_seg_t, slots) == HCCP_FROZEN_BUF_SLOTS, "hccp_buf_seg_t.slots offset drifted");
HCCP_SHM_LAYOUT_ASSERT(sizeof(hccp_buf_seg_t) == HCCP_SHM_SIZE_BUF, "BUF segment size drifted (head + 32 x 4KiB)");

/* --------------------------------------------------------------------------------------------
 * LITE -- dual-host QP rendezvous for the rdma_lite path
 *
 * Frozen against rdma_lite_rendezvous_t (ascend_rdma_lite_stub.c): the trailing 4 bytes
 * are struct padding rounded up to the 8-byte lock alignment; the segment is allocated
 * as the full frozen size HCCP_FROZEN_LITE_SZ (64 bytes on x86_64 / 72 bytes on aarch64).
 * ------------------------------------------------------------------------------------------ */

#define HCCP_LITE_MAX_HOST 2 /* lite RDMA_LITE_MAX_HOST */

typedef struct {
    uint32_t magic; /* HCCP_SHM_MAGIC_LITE, written last by the creator */
    pid_t creator_pid;
    hccp_shm_lock_t lock;                /* serializes the count++ rendezvous pairing */
    uint32_t count;                      /* pair index allocator: idx in {0, 1} */
    uint32_t qp_num[HCCP_LITE_MAX_HOST]; /* my qp_num at idx, spin for the peer */
} hccp_lite_seg_t;

HCCP_SHM_LAYOUT_ASSERT(
    offsetof(hccp_lite_seg_t, count) == HCCP_FROZEN_LITE_COUNT, "hccp_lite_seg_t.count offset drifted");
HCCP_SHM_LAYOUT_ASSERT(
    offsetof(hccp_lite_seg_t, qp_num) == HCCP_FROZEN_LITE_QP, "hccp_lite_seg_t.qp_num offset drifted");
HCCP_SHM_LAYOUT_ASSERT(
    sizeof(hccp_lite_seg_t) == HCCP_SHM_SIZE_LITE, "LITE segment size drifted (fields + tail padding)");

/* --------------------------------------------------------------------------------------------
 * DEVMEM -- per-phy device-memory simulation pool, fixed DVA window 0x500000000000
 *
 * Legacy layout (live today): 64 independent SysV segments, one per phyId, each a
 * 256-byte header region followed by 16 x 4096B slots. A zone below is byte-identical
 * to one legacy segment, so the 64-key -> 1-segment consolidation (t4) is a pure
 * addressing change: zone phyId starts at HCCP_DEVMEM_SEG_ZONE_OFF(phyId).
 *
 * Single-segment layout (reserved for the memfd era): a 256-byte master header, then
 * HCCP_DEVMEM_MAX_PHY zones. It deliberately has no SysV key -- it is born under memfd.
 * The fixed DVA window keeps handing out device addresses as
 * HCCP_DEVMEM_DVA_ADDR + HCCP_DEVMEM_SLOT_OFF(slot) inside the mapped segment.
 * ------------------------------------------------------------------------------------------ */

#define HCCP_SHM_KEY_DEVMEM_BASE ((key_t)0x1234ABD2)
#define HCCP_SHM_MAGIC_DEVMEM ((uint32_t)0x4456454Du)
#define HCCP_DEVMEM_MAX_PHY 64
#define HCCP_DEVMEM_SLOTS 16
#define HCCP_DEVMEM_SIZE 4096
#define HCCP_DEVMEM_HDR 256
#define HCCP_SHM_SIZE_DEVMEM ((size_t)(HCCP_DEVMEM_HDR + HCCP_DEVMEM_SLOTS * HCCP_DEVMEM_SIZE))
#define HCCP_DEVMEM_DVA_ADDR ((void*)0x500000000000ULL)

/* Legacy per-phy pool header. Frozen against hccp_devmem_hdr_t (hccp_shm_infra.c); the
 * legacy lock is not robust yet -- t4 moves it onto hccp_shm_lock_acquire. */
typedef struct {
    uint32_t magic; /* HCCP_SHM_MAGIC_DEVMEM */
    uint32_t rsv;
    hccp_shm_lock_t lock;                   /* serializes the slot bitmap */
    uint32_t buf_in_use[HCCP_DEVMEM_SLOTS]; /* slot bitmap under the lock */
} hccp_devmem_zone_hdr_t;

/* One legacy segment, lifted into a type: header region (padded to HCCP_DEVMEM_HDR)
 * followed by the slot area. The pad bound is also the structural size check: a zone
 * header grown past HCCP_DEVMEM_HDR no longer compiles. */
typedef struct {
    hccp_devmem_zone_hdr_t hdr;
    uint8_t pad[HCCP_DEVMEM_HDR - sizeof(hccp_devmem_zone_hdr_t)];
    uint8_t slots[HCCP_DEVMEM_SLOTS][HCCP_DEVMEM_SIZE];
} hccp_devmem_zone_t;

HCCP_SHM_LAYOUT_ASSERT(
    sizeof(hccp_devmem_zone_hdr_t) == HCCP_FROZEN_DEVMEM_HDR_SZ, "hccp_devmem_zone_hdr_t size drifted");
HCCP_SHM_LAYOUT_ASSERT(
    offsetof(hccp_devmem_zone_hdr_t, buf_in_use) == HCCP_FROZEN_DEVMEM_IN_USE,
    "hccp_devmem_zone_hdr_t.buf_in_use offset drifted");
HCCP_SHM_LAYOUT_ASSERT(
    offsetof(hccp_devmem_zone_t, slots) == HCCP_DEVMEM_HDR, "devmem slot area must stay at zone offset 256");
HCCP_SHM_LAYOUT_ASSERT(
    sizeof(hccp_devmem_zone_t) == HCCP_SHM_SIZE_DEVMEM, "devmem zone must stay 65792 bytes (256B hdr + 16 x 4KiB)");

/* Master header of the consolidated single segment. The lock is reserved for
 * segment-level coordination (creator handshake, zone census); each zone keeps its own
 * lock as the per-phy serialization point. */
#define HCCP_DEVMEM_SEG_HDR 256
#define HCCP_SHM_MAGIC_DEVMEM_SEG ((uint32_t)0x44565347u) /* 'DVSG' */

typedef struct {
    uint32_t magic;   /* HCCP_SHM_MAGIC_DEVMEM_SEG */
    uint32_t version; /* layout version, 1 for now */
    hccp_shm_lock_t lock;
    uint32_t rsv[HCCP_FROZEN_SEG_HDR_RSV]; /* pads the master header to HCCP_DEVMEM_SEG_HDR bytes */
} hccp_devmem_seg_hdr_t;

typedef struct {
    hccp_devmem_seg_hdr_t hdr;
    hccp_devmem_zone_t zones[HCCP_DEVMEM_MAX_PHY]; /* zone == legacy per-phy segment */
} hccp_devmem_seg_t;

#define HCCP_SHM_SIZE_DEVMEM_SEG ((size_t)(HCCP_DEVMEM_SEG_HDR + HCCP_DEVMEM_MAX_PHY * HCCP_SHM_SIZE_DEVMEM))

HCCP_SHM_LAYOUT_ASSERT(
    sizeof(hccp_devmem_seg_hdr_t) == HCCP_DEVMEM_SEG_HDR, "devmem master header must stay 256 bytes");
HCCP_SHM_LAYOUT_ASSERT(
    offsetof(hccp_devmem_seg_t, zones) == HCCP_DEVMEM_SEG_HDR, "devmem zones must start at offset 256");
HCCP_SHM_LAYOUT_ASSERT(
    sizeof(hccp_devmem_seg_t) == HCCP_SHM_SIZE_DEVMEM_SEG,
    "devmem single segment must stay 4210944 bytes (256B + 64 zones)");

/* Mapping constraint (why the consolidated segment is mapped whole, never per zone):
 * mmap file offsets must be page-aligned, but sizeof(hccp_devmem_zone_t) is 16 pages +
 * 256 bytes, so HCCP_DEVMEM_SEG_ZONE_OFF(phyId) is page-aligned only for phyId == 15
 * mod 16 -- a per-zone mapping at the fixed DVA window is impossible for 60 of 64
 * zones. The segment must be mapped at offset 0 over its full length at
 * HCCP_DEVMEM_DVA_ADDR. Consequence: every DVA shifts against the legacy per-phy
 * layout (zone 0 slot 0 moves from DVA+256 to DVA+512), so window-bound arithmetic
 * must rebase through the macros below instead of the legacy DVA+HDR constant. */

/* Transition addressing macros. */

/* Legacy SysV identity of the per-phy pools: 0x1234ABD2 + phyId, phyId 0..63. */
#define HCCP_DEVMEM_LEGACY_KEY(phyId) ((key_t)(HCCP_SHM_KEY_DEVMEM_BASE + (key_t)(phyId)))

/* Slot i of the pool based at <base> (a legacy segment base or a zone base): the fixed
 * DVA window uses the same arithmetic on HCCP_DEVMEM_DVA_ADDR. */
#define HCCP_DEVMEM_SLOT_OFF(slotId) ((size_t)HCCP_DEVMEM_HDR + (size_t)(slotId) * HCCP_DEVMEM_SIZE)

/* Zone phyId of the single segment. phyId must already be validated 0..MAX_PHY-1. */
#define HCCP_DEVMEM_SEG_ZONE_OFF(phyId) \
    (offsetof(hccp_devmem_seg_t, zones) + (size_t)(phyId) * sizeof(hccp_devmem_zone_t))

/* DVA decoder for the consolidated single segment (the one validity predicate shared
 * by every window-check site). Answers "is this address a valid slot DVA" -- inside
 * the fixed window AND landing on some zone's slot area -- and, when requested, hands
 * back the owning zone and slot: DVA == DVA_ADDR + SEG_ZONE_OFF(phy) + SLOT_OFF(slot).
 * Header addresses are rejected by construction: a master/zone-header VA is either
 * below the decode base or folds into the [SLOTS*SIZE, sizeof(zone)) band of the
 * preceding zone, never onto a slot. The predicate answers address validity ONLY:
 * local dereference vs pvm routing stays a separate owner-pid decision, so cross-node
 * paths keep exercising the transport.
 * Returns 1 and fills *phyId and *slotId (either may be NULL for pure classification),
 * or 0 for any non-slot address. */
static inline int hccp_devmem_dva_decode(uint64_t dva, int* phyId, int* slotId)
{
    /* Zone 0's first slot (master header + zone header) is the decode base. */
    const uint64_t slot0
        = (uint64_t)(uintptr_t)HCCP_DEVMEM_DVA_ADDR + HCCP_DEVMEM_SEG_ZONE_OFF(0) + (uint64_t)HCCP_DEVMEM_HDR;
    if (dva < slot0) {
        return 0;
    }
    const uint64_t off = dva - slot0;
    const uint64_t phy = off / (uint64_t)sizeof(hccp_devmem_zone_t);
    const uint64_t rem = off % (uint64_t)sizeof(hccp_devmem_zone_t);
    const uint64_t slotArea = (uint64_t)HCCP_DEVMEM_SLOTS * (uint64_t)HCCP_DEVMEM_SIZE;
    if (phy >= (uint64_t)HCCP_DEVMEM_MAX_PHY || rem >= slotArea || rem % (uint64_t)HCCP_DEVMEM_SIZE != 0) {
        return 0;
    }
    if (phyId != NULL) {
        *phyId = (int)phy;
    }
    if (slotId != NULL) {
        *slotId = (int)(rem / (uint64_t)HCCP_DEVMEM_SIZE);
    }
    return 1;
}

/* --------------------------------------------------------------------------------------------
 * NOTIFY -- host notify page registry (no segment magic; slots are CAS-claimed)
 *
 * Byte-identical to hccp_notify_reg_t in hccp_shm_infra.h: every host test process
 * publishes its notify page base so the peer ibverbs stub can aim process_vm writes
 * that emulate the WRITE_WITH_NOTIFY hardware side effect.
 * ------------------------------------------------------------------------------------------ */

#define HCCP_SHM_KEY_NOTIFY ((key_t)0x1234AC20)

#define HCCP_NOTIFY_MAX_HOST 64

typedef struct {
    volatile pid_t hostPid;     /* owning host process, 0 = free slot */
    volatile uint64_t notifyVa; /* notify page base address of that process */
} hccp_notify_reg_t;

#define HCCP_SHM_SIZE_NOTIFY ((size_t)(HCCP_NOTIFY_MAX_HOST * sizeof(hccp_notify_reg_t)))

HCCP_SHM_LAYOUT_ASSERT(sizeof(hccp_notify_reg_t) == 16, "hccp_notify_reg_t must stay 16 bytes (4B pid + pad + 8B va)");
HCCP_SHM_LAYOUT_ASSERT(offsetof(hccp_notify_reg_t, notifyVa) == 8, "hccp_notify_reg_t.notifyVa must stay at offset 8");
HCCP_SHM_LAYOUT_ASSERT(HCCP_SHM_SIZE_NOTIFY == (size_t)1024, "NOTIFY registry must stay 1024 bytes (64 x 16B)");

#if defined(__cplusplus)
}
#endif

#endif

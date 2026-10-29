# HcclOpExpansionMode

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T06:39:36.455Z pushedAt=2026-09-28T08:29:49.763Z -->

## Description

Defines the expansion mode of a communication operator.

## Prototype

```c
typedef enum {
    HCCL_OP_EXPANSION_MODE_INVALID  = -1,  /* Invalid mode, uninitialized or reserved. */
    HCCL_OP_EXPANSION_MODE_AI_CPU   = 0,   /* Expanded on the AI CPU on the device side. */
    HCCL_OP_EXPANSION_MODE_AIV      = 1,   /* Expanded on the Vector Core (AIV) on the device side. */
    HCCL_OP_EXPANSION_MODE_HOST     = 2,   /* Expanded on the CPU on the host side. The device side automatically selects the scheduler based on the hardware model. */
    HCCL_OP_EXPANSION_MODE_HOST_TS  = 3,   /* Expanded on the CPU on the host side. The host dispatches tasks to the device task scheduler, and the device side performs scheduling and execution. */
    HCCL_OP_EXPANSION_MODE_CCU_MS   = 4,   /* Expanded on the CCU (collective communication acceleration unit) on the device side, using the Memory Slice (MS) mode. */
    HCCL_OP_EXPANSION_MODE_CCU_SCHED = 5,  /* Expanded on the CCU on the device side, using the scheduling mode (the CCU acts as a scheduler to dispatch tasks to the UB engine). */
    HCCL_OP_EXPANSION_AIV_ONLY      = 6,   /* Expanded only on the Vector Core (AIV) on the device side, without mode switching as the data volume changes. */
    HCCL_OP_EXPANSION_MODE_CCU_MS_ONLY   = 7, /* Expanded only on the CCU on the device side, using the Memory Slice (MS) mode, without mode switching as the data volume or resource status changes. */
    HCCL_OP_EXPANSION_MODE_CCU_SCHED_ONLY = 8, /* Expanded only on the CCU on the device side, using the scheduling mode, without mode switching as the data volume or resource status changes. */
} HcclOpExpansionMode;

typedef HcclOpExpansionMode HcclConfigTypeOpExpansionMode;
```

# HcclOpExpansionMode

<!-- md-trans-meta sourceCommit=e169993072c8b4f8692fd6915a9e65337ecf402f translatedAt=2026-10-08T11:35:28.514Z pushedAt=2026-10-12T06:12:35.565Z -->

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
    HCCL_OP_EXPANSION_MODE_CCU_MS   = 4,   /* Expanded on the CCU (Collective Communication Acceleration Unit) on the device side, using CcuBuffer for memory read/write. */
    HCCL_OP_EXPANSION_MODE_CCU_SCHED = 5,  /* Expanded on the CCU on the device side, using the scheduling mode (the CCU acts as a scheduler to dispatch tasks to the UB engine). */
    HCCL_OP_EXPANSION_AIV_ONLY      = 6,   /* Expanded only on the Vector Core (AIV) on the device side, without mode switching as the data size changes. */
} HcclOpExpansionMode;

typedef HcclOpExpansionMode HcclConfigTypeOpExpansionMode;
```

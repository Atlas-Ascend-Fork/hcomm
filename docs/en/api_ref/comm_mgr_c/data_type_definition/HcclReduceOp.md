# HcclReduceOp

<!-- md-trans-meta sourceCommit=7494246995d6d252dbb49e1c6304dffdf77e6d53 translatedAt=2026-09-28T06:41:04.325Z pushedAt=2026-09-28T08:36:22.691Z -->

## Description

Defines the types of reduction operations in collective communication.

## Prototype

```c
typedef enum {
    HCCL_REDUCE_SUM = 0,    /* sum */
    HCCL_REDUCE_PROD = 1,   /* prod */
    HCCL_REDUCE_MAX = 2,    /* max */
    HCCL_REDUCE_MIN = 3,    /* min */
    HCCL_REDUCE_RESERVED = 255 /* reserved */
} HcclReduceOp;
```

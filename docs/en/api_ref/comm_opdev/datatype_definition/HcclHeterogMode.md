# HcclHeterogMode

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-09-28T08:57:15.056Z pushedAt=2026-10-08T05:56:31.865Z -->

## Description

Defines the heterogeneous networking mode.

## Prototype

```c
typedef enum {
    HCCL_HETEROG_MODE_INVALID = -1,    /* Invalid/uninitialized. */
    HCCL_HETEROG_MODE_HOMOGENEOUS = 0, /* Homogeneous networking: a single AI processor type. */
    HCCL_HETEROG_MODE_MIX_A2_A3,       /* Heterogeneous networking: a mix of Atlas A2 products and Atlas A3 products. This mode is not supported currently. */
} HcclHeterogMode;
```

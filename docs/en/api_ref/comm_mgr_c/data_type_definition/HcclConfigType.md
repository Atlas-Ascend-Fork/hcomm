# HcclConfigType

<!-- md-trans-meta sourceCommit=3c68954b90171e154fef0fbc573a36675525893f translatedAt=2026-09-28T06:36:14.281Z pushedAt=2026-09-28T08:21:39.678Z -->

## Description

Configures the expansion mode of the communication operator, the communication algorithm configuration string, and the number of UB multi-channels.

## Prototype

```c
typedef enum {
    HCCL_CONFIG_TYPE_INVALID               = -1,   /* Invalid configuration item type. */
    HCCL_CONFIG_TYPE_OP_EXPANSION_MODE     = 0,    /* Operator expansion mode, corresponding to the type hcclOpExpansionMode. */
    HCCL_CONFIG_TYPE_HCCL_ALGO             = 1,    /* Communication algorithm configuration string, corresponding to a char array of length HCCL_COMM_ALGO_MAX_LENGTH. Newly added enum fields are backward compatible and do not affect code of earlier versions. */
    HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM  = 2,    /* Number of UB multi-channels, corresponding to the type uint32_t. */
} HcclConfigType;
```

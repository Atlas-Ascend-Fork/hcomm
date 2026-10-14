# HcclConfig

<!-- md-trans-meta sourceCommit=e9bc04fe437f954aad83bc43ca7ada08ed671622 translatedAt=2026-10-08T11:35:39.491Z pushedAt=2026-10-12T06:12:35.566Z -->

## Description

Defines the configuration related to collective communication.

## Prototype

```c
typedef enum {
    HCCL_DETERMINISTIC = 0, /* 0: non-deterministic, 1: deterministic, 2: strict(order-preserving) */
    HCCL_CONFIG_RESERVED
} HcclConfig;
```

## Parameters

- **HCCL_DETERMINISTIC**: Whether to enable deterministic computation.

  - **0**: Disables deterministic computation.
  - **1**: Enables deterministic computation.
  - **2**: Enables the order-preservation function (supported only on Atlas A2 products).

- **HCCL_CONFIG_RESERVED**: Reserved parameter.

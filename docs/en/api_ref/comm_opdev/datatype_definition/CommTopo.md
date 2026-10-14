# CommTopo

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:53:49.425Z pushedAt=2026-10-08T03:59:16.340Z -->

## Description

Defines the communication topology types.

## Prototype

```c
typedef enum {
    COMM_TOPO_RESERVED = -1,  /* Reserved topology. */
    COMM_TOPO_CLOS = 0,       /* CLOS interconnection topology. */
    COMM_TOPO_1DMESH = 1,     /* 1DMesh interconnection topology. */
    COMM_TOPO_910_93 = 2,     /* Interconnection topology of Atlas A3 products (with SIO). */
    COMM_TOPO_310P = 3,       /* Interconnection topology of Atlas inference products. */
    COMM_TOPO_A2AXSERVER = 4, /* A2_AX_SERVER */
    COMM_TOPO_CUSTOM = 5      /* Custom. */
} CommTopo;
```

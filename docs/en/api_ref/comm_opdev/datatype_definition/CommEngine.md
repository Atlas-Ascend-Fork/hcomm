# CommEngine

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:52:32.973Z pushedAt=2026-10-08T03:56:21.059Z -->

## Description

Enumeration of communication engine types.

## Prototype

```c
typedef enum {
    COMM_ENGINE_RESERVED = -1,    //< Reserved communication engine.
    COMM_ENGINE_CPU = 0,          //< HOST CPU engine.
    COMM_ENGINE_CPU_TS = 1,       //< HOST CPU TS engine.
    COMM_ENGINE_AICPU = 2,        //< AICPU engine.
    COMM_ENGINE_AICPU_TS = 3,     //< AICPU TS engine.
    COMM_ENGINE_AIV = 4,          //< AIV engine.
    COMM_ENGINE_CCU = 5,          //< CCU engine.
} CommEngine;
```

## Supported Products

<!-- npu="950" id1 -->
For the Ascend 950PR&950DT products, the supported communication engines are as follows:

  - COMM_ENGINE_CPU
  - COMM_ENGINE_AICPU_TS
  - COMM_ENGINE_AIV
  - COMM_ENGINE_CCU
<!-- end id1 -->

<!-- npu="A3" id2 -->
For Atlas A3 products, the supported communication engines are as follows:

  - COMM_ENGINE_AICPU_TS
<!-- end id2 -->

<!-- npu="910b" id3 -->
For Atlas A2 products, the supported communication engines are as follows:

  - COMM_ENGINE_CPU_TS
<!-- end id3 -->

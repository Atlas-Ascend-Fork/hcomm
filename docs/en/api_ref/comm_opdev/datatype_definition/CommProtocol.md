# CommProtocol

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:53:33.480Z pushedAt=2026-10-08T03:58:43.784Z -->

## Description

Defines the communication protocol type enumeration.

## Prototype

```c
typedef enum {
    COMM_PROTOCOL_RESERVED = -1,  /* Reserved protocol type. */
    COMM_PROTOCOL_HCCS = 0,       /* HCCS protocol. */
    COMM_PROTOCOL_ROCE = 1,       /* RDMA over Converged Ethernet */
    COMM_PROTOCOL_PCIE = 2,       /* PCIE protocol. */
    COMM_PROTOCOL_SIO = 3,        /* SIO protocol. */
    COMM_PROTOCOL_UB_CTP = 4,     /* Huawei unified bus UB_CTP. */
    COMM_PROTOCOL_UBC_CTP = COMM_PROTOCOL_UB_CTP, /* Compatible with the legacy protocol name UBC_CTP. */
    COMM_PROTOCOL_UBC_TP = 5,     /* Legacy compatible protocol UBC_TP, not recommended for new development. */
    COMM_PROTOCOL_UB_MEM = 6,     /* UB_MEM */
    COMM_PROTOCOL_UBOE = 7,       /* UBoE */
    COMM_PROTOCOL_HCCS_ONLY = 8,  /* HCCS used for two dies on one card. */
    COMM_PROTOCOL_UB_RTP = 9,     /* UB_RTP */
    COMM_PROTOCOL_UBG = COMM_PROTOCOL_UB_RTP, /* Compatible with the legacy protocol name UBG. */
} CommProtocol;
```

## Supported Products

<!-- npu="950" id1 -->
For the Ascend 950PR&950DT products, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_CPU
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_UB_CTP
  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_UBOE
    - COMM_PROTOCOL_UB_CTP
    - COMM_PROTOCOL_UB_MEM
    - COMM_PROTOCOL_UB_RTP
  - COMM_ENGINE_AIV
    - COMM_PROTOCOL_UB_CTP
    - COMM_PROTOCOL_UB_MEM
    - COMM_PROTOCOL_ROCE
  - COMM_ENGINE_CCU
    - COMM_PROTOCOL_UB_CTP
<!-- end id1 -->

<!-- npu="A3" id2 -->
For Atlas A3 products, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_AICPU_TS
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_HCCS
    - COMM_PROTOCOL_HCCS_ONLY
<!-- end id2 -->

<!-- npu="910b" id3 -->
For Atlas A2 products, the communication protocols supported by each communication engine are as follows:

  - COMM_ENGINE_CPU_TS
    - COMM_PROTOCOL_ROCE
    - COMM_PROTOCOL_HCCS
    - COMM_PROTOCOL_HCCS_ONLY
<!-- end id3 -->

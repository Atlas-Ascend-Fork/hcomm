# HcclP2pKernelParam

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T06:40:40.321Z pushedAt=2026-09-28T08:36:04.386Z -->

## Description

Describes the kernel function parameters of a point-to-point (P2P) communication task, including the send/receive communication thread handle and the operation parameter buffer. This structure is used to pass the context information required for P2P communication when the AI CPU kernel function is launched.

## Prototype

```c
const uint32_t P2P_MAX_ARG_SIZE = 8192U;

typedef struct {
    ThreadHandle sendRecvThread;
    uint8_t opParams[P2P_MAX_ARG_SIZE];
} HcclP2pKernelParam;
```

## Parameters

- **sendRecvThread**: send/receive communication thread handle, used to identify the communication thread that performs point-to-point send/receive operations. See [ThreadHandle](../../comm_opdev/datatype_definition/ThreadHandle.md) for the type definition.
- **opParams**: operation parameter buffer, used to store the parameter data required for kernel function execution, with a maximum length of `P2P_MAX_ARG_SIZE` (8192 bytes). The actual parameter length used is determined by the specific communication operation.

# HcclCMDType

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T06:32:17.690Z pushedAt=2026-09-28T07:00:57.375Z -->

## Description

Identifies the HCCL communication command type, distinguishing between different collective communication and point-to-point communication operations. This enum is used as the `cmdType` field in [HcclOpP2pDesc](./HcclOpP2pDesc.md) to specify the command type of a point-to-point communication task.

> [!NOTE] Note
> When used in [HcclOpP2pDesc](./HcclOpP2pDesc.md), `cmdType` should be set to `HCCL_CMD_SEND` or `HCCL_CMD_RECEIVE`.

## Prototype

```c
typedef enum {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_GATHER,
    HCCL_CMD_SCATTER,
    HCCL_CMD_BATCH_SEND_RECV,
    HCCL_CMD_BATCH_PUT,
    HCCL_CMD_BATCH_GET,
    HCCL_CMD_ALLGATHER_V,
    HCCL_CMD_REDUCE_SCATTER_V,
    HCCL_CMD_BATCH_WRITE,
    HCCL_CMD_HALF_ALLTOALLV = 20,
    HCCL_CMD_ALL,
    HCCL_CMD_FINALIZE = 100,
    HCCL_CMD_INTER_GROUP_SYNC,
    HCCL_CMD_INIT,
    HCCL_CMD_BARRIER,
    HCCL_CMD_MAX
} HcclCMDType;
```

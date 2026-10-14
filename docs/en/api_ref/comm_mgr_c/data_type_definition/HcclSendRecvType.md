# HcclSendRecvType

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-09-28T06:42:10.652Z pushedAt=2026-09-28T08:39:20.652Z -->

## Description

Used for batch point-to-point communication operations to identify whether the current task type is sending or receiving.

## Prototype

```c
typedef enum {
    HCCL_SEND = 0,    /* The current task is a send task. */
    HCCL_RECV = 1,    /* The current task is a receive task. */
    HCCL_SEND_RECV_RESERVED     /* Reserved field. */
} HcclSendRecvType;
```

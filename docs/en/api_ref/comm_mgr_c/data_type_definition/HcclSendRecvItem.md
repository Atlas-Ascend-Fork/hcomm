# HcclSendRecvItem

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-09-28T06:41:57.922Z pushedAt=2026-09-28T08:38:59.835Z -->

## Description

Used for batch point-to-point communication operations to define the basic information of each communication task, including the task type, data address, number of data elements, data type, and peer rank ID.

## Prototype

```c
typedef struct HcclSendRecvItemDef {
    HcclSendRecvType sendRecvType;    /* Indicates whether the current task type is send or receive. */
    void *buf;        /* Buffer address for data send/receive. */
    uint64_t count;   /* Number of data elements to send/receive. */
    HcclDataType dataType;   /* Data type of the data to send/receive. */
    uint32_t remoteRank;     /* Peer rank ID of the data receive/send end in the communicator. */
} HcclSendRecvItem;
```

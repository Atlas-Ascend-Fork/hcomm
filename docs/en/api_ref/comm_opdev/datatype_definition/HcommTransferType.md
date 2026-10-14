# HcommTransferType

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T09:02:03.107Z pushedAt=2026-10-08T06:26:45.352Z -->

## Description

Defines the transfer type enumeration used in batch transfer operations. It is used for the **transType** field in the [HcommBatchTransferDesc](HcommBatchTransferDesc.md) structure to specify the operation type performed by a single transfer descriptor.

## Prototype

```c
typedef enum {
    HCOMM_TRANSFER_TYPE_INVALID = -1,                       /* Invalid transfer type. */
    HCOMM_TRANSFER_TYPE_WRITE = 0,                          /* One-sided write, corresponding to transferInfo.write. */
    HCOMM_TRANSFER_TYPE_WRITE_REDUCE = 1,                   /* One-sided write with reduction, corresponding to transferInfo.reduce. */
    HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY = 2,              /* One-sided write with notification, corresponding to transferInfo.writeWithNotify. */
    HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY = 3,       /* One-sided write with reduction and notification, corresponding to transferInfo.writeReduceWithNotify. */
    HCOMM_TRANSFER_TYPE_READ = 4,                           /* One-sided read, corresponding to transferInfo.read. */
    HCOMM_TRANSFER_TYPE_READ_REDUCE = 5,                    /* One-sided read with reduction, corresponding to transferInfo.reduce. */
    HCOMM_TRANSFER_TYPE_NOTIFY_RECORD = 6                   /* Record notification event, corresponding to transferInfo.notifyRecord. */
} HcommTransferType;
```

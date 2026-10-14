# HcommBatchTransferDesc

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T08:58:08.042Z pushedAt=2026-10-08T06:03:27.570Z -->

## Description

Batch transfer descriptor structure, used to describe the parameters of a single transfer task. The transfer type is specified by **transType**, and the detailed parameters of the corresponding type are provided by the **transferInfo** union.

## Prototype

```c
typedef struct {
    HcommTransferType transType;    /* Transfer type. See HcommTransferType. */
    uint8_t reserved[4];            /* Reserved field. */
    union {
        uint8_t raws[56];           /* Raw byte access, used for zero-copy serialization and other scenarios. */
        struct {
            uint64_t len;           /* Data length (in bytes). */
            void *dst;              /* Remote destination address. */
            void *src;              /* Local source address. */
        } write;
        struct {
            uint64_t len;           /* Data length (in bytes). */
            void *dst;              /* Local destination address. */
            void *src;              /* Remote source address. */
        } read;
        struct {
            uint64_t count;         /* Number of elements. */
            void *dst;              /* Remote destination address. */
            void *src;              /* Local source address. */
            HcommReduceOp reduceOp; /* Reduction operation type. */
            HcommDataType dataType; /* Data type. */
        } reduce;
        struct {
            uint32_t notifyIdx;     /* Notification index. */
        } notifyRecord;
        struct {
            uint64_t len;           /* Data length (in bytes). */
            void *dst;              /* Remote destination address. */
            void *src;              /* Local source address. */
            uint32_t notifyIdx;     /* Remote notification index to notify after the write completes. */
        } writeWithNotify;
        struct {
            uint64_t count;         /* Number of elements. */
            void *dst;              /* Remote destination address. */
            void *src;              /* Local source address. */
            HcommReduceOp reduceOp; /* Reduction operation type. */
            HcommDataType dataType; /* Data type. */
            uint32_t notifyIdx;     /* Remote notification index to notify after the write reduction completes. */
        } writeReduceWithNotify;
    } transferInfo;
} HcommBatchTransferDesc;
```

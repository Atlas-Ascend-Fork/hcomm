# HcclChannelDesc

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:56:24.261Z pushedAt=2026-10-08T05:52:57.777Z -->

## Description

Defines communication channel parameters.

## Prototype

```c
typedef struct {
    CommAbiHeader header;
    uint32_t remoteRank;              /* Remote rankId */
    CommProtocol channelProtocol;     /* Communication protocol. */
    EndpointDesc localEndpoint;       /* Local network device endpoint description. Supported only on Ascend 950PR&950DT products. */
    EndpointDesc remoteEndpoint;      /* Remote network device endpoint description. Supported only on Ascend 950PR&950DT products. */
    uint32_t notifyNum;               /* Number of synchronization signals used on the channel, ranging from 0 to 64, defaulting to 0. */
    HcclMemHandle *memHandles;        /* Memory handles to be exchanged that are registered to the communicator. Supported only by the AIV engine of Ascend 950PR&950DT products. */
    uint32_t memHandleNum;            /* Number of memory handles to be exchanged that are registered to the communicator. Supported only by the AIV engine of Ascend 950PR&950DT products. */
    union {
        uint8_t raws[128];            /* General-purpose buffer. */
        struct {
            uint32_t queueNum;        /* Number of QPs. Currently only one QP is supported. */
            uint32_t retryCnt;        /* Maximum number of retransmissions, ranging from 0 to 7, defaulting to 7. */
            uint32_t retryInterval;   /* Retransmission interval, ranging from 5 to 24, defaulting to 20 (corresponding to 4.096*2^20 us). */
            uint8_t tc;               /* Traffic class (QoS), ranging from 0 to 255, defaulting to 132. Supported only on Atlas A2 products and Atlas A3 products. */
            uint8_t sl;               /* Service level (QoS), ranging from 0 to 7, defaulting to 4. Supported only on Atlas A2 products and Atlas A3 products. */
        } roceAttr;
        struct {
            uint8_t pathMode;         /* UB_MEM access path mode. Value range: 0, 1, 2, and 0xFF (default value 0; when configured as 0xFF, it is processed as 0). 0: automatic mode (prefer single path; use multi-path if unavailable), 1: forced single-path mode, 2: forced multi-path mode. */
        } ubMemAttr;
    };
} HcclChannelDesc;
```

# HcommChannelDesc

<!-- md-trans-meta sourceCommit=451436849f153e7d5464345f85627b4f6fb2a8c9 translatedAt=2026-09-28T09:00:07.660Z pushedAt=2026-10-08T10:46:24.268Z -->

## Description

Defines inter-component channel parameters.

## Prototype

```c
typedef struct {
    CommAbiHeader header;             /* ABI header, containing version and other information. */
    EndpointDesc remoteEndpoint;      /* Remote network device endpoint description. */
    uint32_t notifyNum;               /* Number of synchronization signals used on the channel. */

    // When exchangeAllMems is True, memHandle does not need to be configured.
    bool exchangeAllMems;             /* Whether to exchange the memory information registered on the local network device. */
    HcommMemHandle *memHandles;       /* Memory handle to be exchanged registered to the communicator; invalid when exchangeAllMems is True. */
    uint32_t memHandleNum;            /* Number of memory handles to be exchanged registered to the communicator; invalid when exchangeAllMems is True. */
    HcommSocket socket;               /* Socket handle. */
    HcommSocketRole role;             /* Local role (SERVER or CLIENT). */
    uint16_t port;                    /* Socket listening on the specified port. */
    union {
        uint8_t raws[128];            /* General-purpose buffer. */
        struct {
            uint32_t queueNum;        /* Number of QPs. Value range: [1, 32]. Recommended range: [1, 8]. Defaults to 1 when not configured (0xFFFFFFFF). */
            uint32_t retryCnt;        /* Maximum number of retransmissions. Range: 0 to 7. Default: 7. */
            uint32_t retryInterval;   /* Retransmission interval. Range: 5 to 24. Default: 20 (corresponding to 4.096*2^20us). */
            uint8_t tc;               /* Traffic class (QoS). Range: 0 to 255. Default: 132. */
            uint8_t sl;               /* Service level (QoS). Range: 0 to 7. Default: 4. */
            uint32_t qpThreshold;     /* Minimum data amount per QP in multi-QP scenarios (B). */
            uint32_t cqAttrFlags;     /* CQ attribute flags, used to configure the flags of ibv_cq_init_attr_ex. Default: 0.
                                         Note: NPU NICs do not support this configuration. Whether it takes effect for third-party NICs depends on the capabilities of each NIC. */
            uint16_t* srcPortList;    /* QP source port number list, used for RoCE network hash-based traffic distribution. NULL means not configured.
                                         A field added in ABI v4. HCOMM sets it to NULL when lower-version callers do not set it.
                                         The array length must be equal to queueNum. The i-th QP uses srcPortList[i % array length] as the UDP source port number.
                                         When a channel is created through HcclChannelAcquire, HCOMM automatically fills this field based on the environment variable
                                         MultiQpSrcPort.cfg configuration file pointed to by HCCL_RDMA_QP_PORT_CONFIG_PATH.
                                         When exchangeAllMems is true (that is, in one-sided communication scenario), this field does not take effect (udpSport is set to 0). */
        } roceAttr;
        struct {
            uint32_t qos;             /* HCCS QoS */
        } hccsAttr;
        struct {
            uint32_t sqDepth;         /* UB queue depth. 0 and 0xffffffff mean using the default value. */
            uint32_t scqDepth;        /* UB SCQ queue depth. 0 and 0xffffffff mean using the default value. With a valid value, JFC is created in exclusive mode, which supports only the HOST and AICPU scenarios and is not supported by AIV. The value range is [64, 32768] for the HOST scenario and [64, 16384] for the AI CPU scenario. */
        } ubAttr;
        struct {
            uint8_t pathMode;         /* UB_MEM access path mode. Value range: 0, 1, 2, and 0xFF (the default value is 0, and 0xFF is processed as 0). 0: automatic mode (single path preferred, or multi-path if unavailable). 1: forced single-path mode. 2: forced multi-path mode. */
        } ubMemAttr;
    };
    uint32_t qos;             /* Channel QoS (decoupled from the protocol; in collective communication scenarios, it usually comes from the communicator hcclQos). Value range: [0, 7], or 0xFFFFFFFF meaning not configured.
                                 When not configured: RoCE does not rewrite roceAttr.sl/tc; the internal default QoS (4) is used for UB-type protocols. */
    const char *channelName;  /* Channel service matching identifier, which must be the same on both ends; NULL indicates an anonymous channel. */
} HcommChannelDesc;
```

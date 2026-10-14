# HcclTeamCreateDesc

<!-- md-trans-meta sourceCommit=94a831c461618c9c7c40a063ef2768860b855fee translatedAt=2026-09-28T08:57:40.909Z pushedAt=2026-10-08T06:01:53.489Z -->

## Description

Team creation descriptor, used by the [HcclTeamCreate](../control_plane_api/comms_domain_resource_mgmt/HcclTeamCreate.md) API to describe information such as members, network layer, communication protocol, communication engine, **syncMem** requirements, and shared queue.

## Prototype

```c
typedef struct {
    CommAbiHeader header;
    const uint32_t* rankIds;
    uint32_t rankNum;
    uint32_t selfRankId;
    uint32_t netLayer;
    CommProtocol protocol;
    HcommTeamSyncMemRequirement requirement;
    CommEngine engine;
    uint32_t notifyNum;
    uint32_t channelCnt;
    uint32_t isSharedQueue;
    char sharedQueueTag[HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN];
    uint32_t reserved[8];
} HcclTeamCreateDesc;
```

## Field Description

| Field | Description |
| --- | --- |
| header | ABI header, initialized by [HcclTeamCreateDescInit](../control_plane_api/comms_domain_resource_mgmt/HcclTeamCreateDescInit.md). For the definition of the CommAbiHeader type, see [CommAbiHeader](CommAbiHeader.md). |
| rankIds | Array of **rankIds**, with a length of **rankNum**. The value cannot be **NULL**. |
| rankNum | Number of members. The value cannot be **0** or **1**, and cannot be greater than the **rankSize** of the communicator. |
| selfRankId | Actual **rankId** of this rank, which must exist in **rankIds**. |
| netLayer | Desired network layer. The value can only be **0**, **1**, or **2**, where **0** indicates the default selection. |
| protocol | Desired communication protocol. The value cannot be **COMM_PROTOCOL_RESERVED**. A team supports only one protocol. For the definition of the CommProtocol type, see [CommProtocol](CommProtocol.md). |
| requirement | **syncMem** requirements, including the number of signals/counters/barriers. For the definition of the HcommTeamSyncMemRequirement type, see [HcommTeamSyncMemRequirement](HcommTeamSyncMemRequirement.md). |
| engine | Communication engine type, for example, **COMM_ENGINE_AIV**. For the definition of the CommEngine type, see [CommEngine](CommEngine.md). |
| notifyNum | Number of Notify resources required by each channel. The value range is \[0,64\]. |
| channelCnt | Number of channels for each remote member. The value cannot be **0**. |
| isSharedQueue | Whether to use a shared queue. **0** indicates not used, and a non-zero value indicates used. |
| sharedQueueTag | Shared queue tag, a string ending with '\\0', with a maximum length of **HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN (255)**. This field is valid only when **isSharedQueue** is non-zero. |
| reserved[8] | Reserved field. |

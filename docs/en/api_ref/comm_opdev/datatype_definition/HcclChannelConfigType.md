# HcclChannelConfigType

<!-- md-trans-meta sourceCommit=80ded7b5254f8a4cd6070092501f5f0d1c1a7170 translatedAt=2026-09-28T08:56:21.182Z pushedAt=2026-10-08T10:32:44.695Z -->

## Description

Enum of channel configuration attribute types, used to specify the attribute type to be set in the [HcclChannelConfigSetInt](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigSetInt.md) and [HcclChannelConfigSetStr](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigSetStr.md) APIs.

## Type Definition

```c
typedef enum {
    HCCL_CHANNEL_CONFIG_TYPE_INVALID = -1,
    HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE = 0,
    HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG = 1,
} HcclChannelConfigType;
```

## Enum Value Description

| Enum Value | Value | Description | Applicable API |
| --- | --- | --- | --- |
| **HCCL_CHANNEL_CONFIG_TYPE_INVALID** | -1 | Invalid value, used for initialization validation. | - |
| **HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE** | 0 | Whether to enable shared queue mode (bool/int, default **0=false**).<br>When it is set to **true**, multiple created channels share one Jetty.<br>Only the UB network semantic protocol (UB_CTP/UB_RTP) of the AIV engine is supported. | [HcclChannelConfigSetInt](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigSetInt.md) |
| **HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG** | 1 | Tag identifier of the shared queue (string).<br>Valid only when **IS_SHARED_QUEUE=true**; specifies the tag for creating the channel.<br>On repeated calls, channels created with the same tag share one Jetty. | [HcclChannelConfigSetStr](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigSetStr.md) |

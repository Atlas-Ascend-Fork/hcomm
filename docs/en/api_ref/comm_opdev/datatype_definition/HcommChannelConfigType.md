# HcommChannelConfigType

<!-- md-trans-meta sourceCommit=80ded7b5254f8a4cd6070092501f5f0d1c1a7170 translatedAt=2026-09-28T08:59:07.382Z pushedAt=2026-10-08T06:11:00.385Z -->

## Description

Enumeration of channel configuration attribute types, used to specify the attribute type to be set in the [HcommChannelConfigSetInt](../control_plane_api/basic_resource_mgmt/HcommChannelConfigSetInt.md) API.

## Type Definition

```c
typedef enum {
    HCOMM_CHANNEL_CONFIG_TYPE_INVALID = -1,
    HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE = 0,
} HcommChannelConfigType;
```

## Enum Value Description

| Enum Value | Value | Description | Applicable API |
| --- | --- | --- | --- |
| HCOMM_CHANNEL_CONFIG_TYPE_INVALID | -1 | Invalid value, used for initialization validation. | - |
| HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE | 0 | Whether to enable the shared queue mode (bool/int, default **0=false**).<br>When it is set to **true**, multiple channels created with [HcommChannelCreateWithConfig](../control_plane_api/basic_resource_mgmt/HcommChannelCreateWithConfig.md) share a single Jetty.<br>Only the UB network semantic protocol (UB_CTP/UB_RTP) of the AIV engine is supported. | [HcommChannelConfigSetInt](../control_plane_api/basic_resource_mgmt/HcommChannelConfigSetInt.md) |

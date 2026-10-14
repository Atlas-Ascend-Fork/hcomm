# HcommChannelStatus

<!-- md-trans-meta sourceCommit=05dc726adbd2c72ee36c39dc5844045f412eb39d translatedAt=2026-10-08T12:19:09.910Z pushedAt=2026-10-12T06:12:35.668Z -->

## Description

Enumerates channel link establishment statuses. It is used to parse the link establishment status value of each channel in the **statusList** output parameter of the [HcommChannelGetStatus](../control_plane_api/basic_resource_mgmt/HcommChannelGetStatus.md) API.

## Type Definition

```c
typedef enum {
    HCOMM_CHANNEL_STATUS_READY = 0,
    HCOMM_CHANNEL_STATUS_CONNECTING = 1,
    HCOMM_CHANNEL_STATUS_FAILED_INTERNAL = 2,
} HcommChannelStatus;
```

## Enum Value Description

| Enum Value | Value | Description |
| --- | --- | --- |
| HCOMM_CHANNEL_STATUS_READY | 0 | Link establishment is complete, and the channel is ready for communication operations. |
| HCOMM_CHANNEL_STATUS_CONNECTING | 1 | Link establishment is in progress. Continue calling the [HcommChannelGetStatus](../control_plane_api/basic_resource_mgmt/HcommChannelGetStatus.md) API to drive link establishment. |
| HCOMM_CHANNEL_STATUS_FAILED_INTERNAL | 2 | Link establishment failed. |

## Usage

- After a channel is created through [HcommChannelCreate](../control_plane_api/basic_resource_mgmt/HcommChannelCreate.md) or [HcommChannelCreateWithConfig](../control_plane_api/basic_resource_mgmt/HcommChannelCreateWithConfig.md), link establishment is not completed immediately. The caller must call [HcommChannelGetStatus](../control_plane_api/basic_resource_mgmt/HcommChannelGetStatus.md) in a loop and use this enum to check the link establishment progress. Communication operations can be performed only after the channel is ready.
- When the status is **HCOMM_CHANNEL_STATUS_CONNECTING**, link establishment is still in progress, and the API must be called again to drive link establishment. When the status is **HCOMM_CHANNEL_STATUS_READY**, the channel is ready.
- Link establishment is considered as failure if the status value greater than or equal to **HCOMM_CHANNEL_STATUS_FAILED_INTERNAL**. In addition to the enum values above, the current version may also return more granular status values such as link establishment timeout, insufficient local resources, and insufficient remote resources. These values are provided only to help the caller locate the cause of the failure and should not be used as the basis for determination.

# HcommChannelConfig

<!-- md-trans-meta sourceCommit=0b6eec4ecd6c415834808212b0a36d81ff4f3e32 translatedAt=2026-09-28T08:58:41.918Z pushedAt=2026-10-08T06:09:28.576Z -->

## Description

Opaque handle to a channel configuration object, used to pass advanced configurations such as shared Jetty when creating a communication channel through the [HcommChannelCreateWithConfig](../control_plane_api/basic_resource_mgmt/HcommChannelCreateWithConfig.md) API.

## Type Definition

```c
typedef void *HcommChannelConfig;
```

## Usage

- Create it through [HcommChannelConfigCreate](../control_plane_api/basic_resource_mgmt/HcommChannelConfigCreate.md).
- Set attributes through [HcommChannelConfigSetInt](../control_plane_api/basic_resource_mgmt/HcommChannelConfigSetInt.md).
- Destroy it through [HcommChannelConfigDestroy](../control_plane_api/basic_resource_mgmt/HcommChannelConfigDestroy.md) after use.

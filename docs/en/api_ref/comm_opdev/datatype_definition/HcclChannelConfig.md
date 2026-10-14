# HcclChannelConfig

<!-- md-trans-meta sourceCommit=0b6eec4ecd6c415834808212b0a36d81ff4f3e32 translatedAt=2026-09-28T08:55:43.399Z pushedAt=2026-10-08T04:02:34.437Z -->

## Description

Opaque handle to a channel configuration object, used to pass advanced configurations such as shared Jetty when creating a communication channel through the [HcclChannelAcquireWithConfig](../control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquireWithConfig.md) API.

## Type Definition

```c
typedef void *HcclChannelConfig;
```

## Usage

- Create it through [HcclChannelConfigCreate](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigCreate.md).
- Set attributes through [HcclChannelConfigSetInt](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigSetInt.md) and [HcclChannelConfigSetStr](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigSetStr.md).
- Destroy it through [HcclChannelConfigDestroy](../control_plane_api/comms_domain_resource_mgmt/HcclChannelConfigDestroy.md) after use.

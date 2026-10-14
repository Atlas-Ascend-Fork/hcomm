# HcommEndpointFeatureType

<!-- md-trans-meta sourceCommit=73cabe45b7e2f3d9616f1af5c4030feff964f98d translatedAt=2026-09-28T08:59:46.607Z pushedAt=2026-10-08T06:20:14.115Z -->

## Description

Defines the underlying feature types, used to specify the feature type to be queried in the [HcommEndpointCheckFeature](../control_plane_api/basic_resource_mgmt/HcommEndpointCheckFeature.md) API.

## Prototype

```c
typedef enum {
    HCOMM_ENDPOINT_FEATURE_INVALID = -1,       /* Invalid feature type. Configuration is not supported. */
    HCOMM_ENDPOINT_FEATURE_NDA = 0,            /* NPU Direct RDMA Async feature. */
} HcommEndpointFeatureType;
```

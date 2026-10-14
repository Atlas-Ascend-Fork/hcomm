# ThreadResType

<!-- md-trans-meta sourceCommit=3733b0cf5d4d74afc2f5650108a92c34389c8d7f translatedAt=2026-09-28T09:03:01.357Z pushedAt=2026-10-08T06:29:55.231Z -->

## Description

Underlying resource types that can be obtained.

## Prototype

```c
typedef enum {
    THREAD_RES_TYPE_INVALID = -1,
    THREAD_RES_TYPE_STREAM = 0,   // The obtained resource type is stream.
} ThreadResType;
```

## Field Description

| Field | Value | Description |
| --- | --- | --- |
| THREAD_RES_TYPE_INVALID | -1 | Invalid resource type. |
| THREAD_RES_TYPE_STREAM | 0 | Stream resource. The corresponding resource type is [ThreadResTypeStream](ThreadResTypeStream.md), which can be obtained through the [HcclThreadResGetInfo](../control_plane_api/comms_domain_resource_mgmt/HcclThreadResGetInfo.md) API. |

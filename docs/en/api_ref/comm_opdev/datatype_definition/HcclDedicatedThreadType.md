# HcclDedicatedThreadType

<!-- md-trans-meta sourceCommit=459e4836fcaab27928c332213fec68eb6ccd895c translatedAt=2026-09-28T08:56:51.433Z pushedAt=2026-10-08T05:55:43.852Z -->

## Description

Enumeration of dedicated communication thread usage types, used to specify the usage scenario of a dedicated thread in the [HcclDedicatedThreadAcquire](../control_plane_api/comms_domain_resource_mgmt/HcclDedicatedThreadAcquire.md) API.

## Prototype

```c
typedef enum {
    HCCL_DED_THREAD_TYPE_INVALID = -1,
    HCCL_DED_THREAD_TYPE_AICPU_LAUNCH = 0,
    HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE = 1,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE = 2,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH = 3,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE = 4,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE = 5
} HcclDedicatedThreadType;
```

## Member Description

| Member | Description |
| --- | --- |
| HCCL_DED_THREAD_TYPE_INVALID | Invalid dedicated thread type. |
| HCCL_DED_THREAD_TYPE_AICPU_LAUNCH | Dedicated thread used for AI CPU task launch in single operator mode. |
| HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE | Dedicated thread used for AI CPU task launch in graph mode. |
| HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE | Dedicated thread used for AI CPU ordered task launch in single operator mode. |
| HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH | Dedicated thread used for AI CPU ordered task launch in aclgraph mode. |
| HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE | Dedicated thread used for AI CPU ordered task launch in graph mode. An attached stream must be set in advance through **HcomSetAttachedStream**. |
| HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE | Dedicated thread used for AI CPU ordered task launch on the device side. |

## Constraints

1. **HCCL_DED_THREAD_TYPE_INVALID** is an invalid value. Passing it to the [HcclDedicatedThreadAcquire](../control_plane_api/comms_domain_resource_mgmt/HcclDedicatedThreadAcquire.md) API returns a failure.

2. **HCCL_DED_THREAD_TYPE_AICPU_LAUNCH** and **HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE** are non-ordered types. Their threads are cached by **useType** within the communicator.

3. The **HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_*** series are ordered types. Their threads are managed by the process-level **OrderLaunchThreadMgr** and are not bound to a single communicator. For the behavioral differences among the ordered types, see the constraints in [HcclDedicatedThreadAcquire](../control_plane_api/comms_domain_resource_mgmt/HcclDedicatedThreadAcquire.md).

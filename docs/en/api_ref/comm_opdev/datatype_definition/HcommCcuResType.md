# HcommCcuResType

<!-- md-trans-meta sourceCommit=d7dccb74e04a028e6531edec11eb2f6ccff0e9bf translatedAt=2026-09-28T08:58:31.995Z pushedAt=2026-10-08T06:08:27.506Z -->

## Description

CCU resource type enumeration, used to specify the resource type in the resource descriptor ([HcommCcuResDescHandle](HcommCcuResDescHandle.md)) in the [HcommCcuInsResDescSetNum](../control_plane_api/ccu_resource_mgmt/HcommCcuInsResDescSetNum.md) and [HcommCcuInsResDescQueryNum](../control_plane_api/ccu_resource_mgmt/HcommCcuInsResDescQueryNum.md) APIs.

## Prototype

```c
typedef enum {
    HCOMM_CCU_RES_TYPE_INVALID = -1,
    HCOMM_CCU_RES_TYPE_LOOP = 0,
    HCOMM_CCU_RES_TYPE_CCU_BUF = 1,
    HCOMM_CCU_RES_TYPE_VARIABLE = 2,
    HCOMM_CCU_RES_TYPE_ADDRESS = 3,
    HCOMM_CCU_RES_TYPE_EVENT = 4,
    HCOMM_CCU_RES_TYPE_CCU_THREAD = 5,
    HCOMM_CCU_RES_TYPE_INSTRUCTION = 6
} HcommCcuResType;
```

## Field Description

| Field | Value | Description |
| --- | --- | --- |
| HCOMM_CCU_RES_TYPE_INVALID | -1 | Invalid resource type. |
| HCOMM_CCU_RES_TYPE_LOOP | 0 | Loop resource. |
| HCOMM_CCU_RES_TYPE_CCU_BUF | 1 | CCU Buffer resource, that is, the on-chip high-speed scratchpad created through [CcuBuffer](../data_plane_api/ccu/resource_allocation_operation/CcuBuffer.md) within the Kernel. |
| HCOMM_CCU_RES_TYPE_VARIABLE | 2 | Variable resource, that is, the variable resource created through [Variable](../data_plane_api/ccu/resource_allocation_operation/Variable.md) within the kernel. |
| HCOMM_CCU_RES_TYPE_ADDRESS | 3 | Address resource, that is, the address resource created through [Address](../data_plane_api/ccu/resource_allocation_operation/Address.md) within the kernel. |
| HCOMM_CCU_RES_TYPE_EVENT | 4 | Event resource, that is, the event resource created through [Event](../data_plane_api/ccu/resource_allocation_operation/Event.md) within the kernel. |
| HCOMM_CCU_RES_TYPE_CCU_THREAD | 5 | CCU thread resource. |
| HCOMM_CCU_RES_TYPE_INSTRUCTION | 6 | Instruction resource. |

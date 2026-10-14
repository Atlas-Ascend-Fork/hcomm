# HcommCcuResDescHandle

<!-- md-trans-meta sourceCommit=d7dccb74e04a028e6531edec11eb2f6ccff0e9bf translatedAt=2026-09-28T08:58:12.275Z pushedAt=2026-10-08T06:06:42.185Z -->

## Description

CCU resource descriptor handle type (`uint64_t`). A resource descriptor describes the quantity specifications of various resources (loop, buffer, variable, address, event, thread, instruction) required by a CCU instance, and does not contain actual resources. One resource descriptor corresponds to one IO die.

It is created by [HcommCcuInsResDescCreate](../control_plane_api/ccu_resource_mgmt/HcommCcuInsResDescCreate.md) on the host side. After creation, the quantity of each resource type is initialized to 0. The quantity can be set type by type through [HcommCcuInsResDescSetNum](../control_plane_api/ccu_resource_mgmt/HcommCcuInsResDescSetNum.md) and queried through [HcommCcuInsResDescQueryNum](../control_plane_api/ccu_resource_mgmt/HcommCcuInsResDescQueryNum.md). After the settings are complete, it is passed to [HcommCcuInsCreate](../control_plane_api/ccu_resource_mgmt/HcommCcuInsCreate.md) to create a CCU instance.

In addition, it can be used as a carrier for query results:

- [HcommCcuInsQueryResDesc](../control_plane_api/ccu_resource_mgmt/HcommCcuInsQueryResDesc.md): Queries the quantity of resources actually occupied by a CCU instance and writes it into the descriptor.
- [HcommCcuQueryRemainResDesc](../control_plane_api/ccu_resource_mgmt/HcommCcuQueryRemainResDesc.md): Queries the maximum contiguous remaining resource quantity on the corresponding IO die and writes it into the descriptor.
- [HcommCcuKernelQueryResReq](../control_plane_api/ccu_resource_mgmt/HcommCcuKernelQueryResReq.md): Queries the resource requirements of the CCU kernel and writes them into the descriptor.

When the handle is no longer in use, it must be destroyed through [HcommCcuInsResDescDestroy](../control_plane_api/ccu_resource_mgmt/HcommCcuInsResDescDestroy.md). After destruction, the handle becomes invalid.

## Prototype

```c
typedef uint64_t HcommCcuResDescHandle;
```

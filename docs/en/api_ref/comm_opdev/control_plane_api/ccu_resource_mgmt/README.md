# Introduction

<!-- md-trans-meta sourceCommit=ea28a2a84bd47924bd51751e37d9926552e1ad78 translatedAt=2026-09-28T07:19:33.715Z pushedAt=2026-09-29T11:03:15.750Z -->

The CCU resource management API is used to query and describe CCU resource requirements, create CCU instances on demand, query the resource usage of instances, reserve variable/event resources on instances, and destroy instances held by the caller.

## Creating and Destroying a CCU Instance

The typical process for creating a CCU instance on demand is as follows:

1. Call [HcommCcuInsResDescCreate](HcommCcuInsResDescCreate.md) to create a resource descriptor for the target IO die.
2. Call [HcommCcuKernelQueryResReq](HcommCcuKernelQueryResReq.md) to collect the kernel resource requirements, or call [HcommCcuInsResDescSetNum](HcommCcuInsResDescSetNum.md) to directly set the number of resources.
3. Call [HcommCcuInsCreate](HcommCcuInsCreate.md) to create a CCU instance.
4. Call [HcommCcuInsQueryResDesc](HcommCcuInsQueryResDesc.md) to query the resources actually occupied by the instance if needed.
5. When the instance is no longer needed, call [HcommCcuInsDestroy](HcommCcuInsDestroy.md) to destroy it. If the instance has been successfully bound to a communicator through [HcclCommAssignCcuIns](../comms_domain_resource_mgmt/HcclCommAssignCcuIns.md), its lifecycle is managed by the communicator.

## Reserving Variable/Event Resources

When a group of scalar registers/completion event units needs to be shared across kernels, or shared with a target module outside the CCU, resources can be reserved on the instance:

1. Call [HcommCcuVariableAlloc](HcommCcuVariableAlloc.md) or [HcommCcuEventAlloc](HcommCcuEventAlloc.md) to reserve a physically contiguous segment of variables (scalar registers) or events (completion event units) from the instance resource pool, and obtain a reservation handle. After reservation, there are two usage modes, which are independent of each other. You can use one of them or both:
   - On the host side, call [HcommCcuVariableGetAddr](HcommCcuVariableGetAddr.md) or [HcommCcuEventGetAddr](HcommCcuEventGetAddr.md) to obtain the mapped virtual address of each resource in the segment, and pass it to the target module as the original value.
   - Pass the reservation handle to the kernel through `kernelArgs`, and bind it to the same segment of resources in the kernel using the reservation handle construction forms of [Variable](../../data_plane_api/ccu/resource_allocation_operation/Variable.md), [Event](../../data_plane_api/ccu/resource_allocation_operation/Event.md), or [Array](../../data_plane_api/ccu/resource_allocation_operation/Array.md).
2. There is no separate API for releasing reserved resources. The reservation handle becomes invalid when the instance is destroyed by [HcommCcuInsDestroy](HcommCcuInsDestroy.md).

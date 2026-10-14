# Introduction

<!-- md-trans-meta sourceCommit=ade3956603841a7c26a6df1554296fff3817ae74 translatedAt=2026-09-28T07:09:17.196Z pushedAt=2026-09-29T07:55:12.666Z -->

This section describes the lifecycle management APIs and the memory token query APIs on the host side of the CCU kernel.

With these APIs, you can register, translate, and launch the kernel, as well as convert process virtual addresses into CCU access tokens.

## Prerequisites

The kernel registration and launch APIs require a `CcuInsHandle` (CCU instance handle). Currently, only collective communication scenarios are supported. Before calling these APIs, you must complete HCCL communicator initialization, CCU link establishment, and CCU instance binding.

1. **Create an HcclComm communicator**: See [HcclCommInitClusterInfo](../../../comm_mgr_c/HcclCommInitClusterInfo.md)
   or [HcclCommInitRootInfo](../../../comm_mgr_c/HcclCommInitRootInfo.md).

2. **Plan resources and create a CCU instance**:

   - Call [HcommCcuInsResDescCreate](../ccu_resource_mgmt/HcommCcuInsResDescCreate.md) to create a resource descriptor for the target IO die.
   - Call [HcommCcuKernelQueryResReq](../ccu_resource_mgmt/HcommCcuKernelQueryResReq.md) to collect kernel resource requirements.
   - Call [HcommCcuInsCreate](../ccu_resource_mgmt/HcommCcuInsCreate.md) to create a CCU instance.

3. **Bind the CCU instance**:

   - Call [HcclCommAssignCcuIns](../comms_domain_resource_mgmt/HcclCommAssignCcuIns.md) to bind the instance to the communicator. After successful binding, the ownership and destruction responsibility of the instance are transferred to the communicator.
   - The caller that created the instance can continue to use the handle returned by `HcommCcuInsCreate` to register and launch the kernel, but must not destroy the instance on its own. When only the communicator handle is held, [HcclCommQueryAssignedCcuIns](../comms_domain_resource_mgmt/HcclCommQueryAssignedCcuIns.md) can be called to obtain the bound instance handle.

   A typical usage of querying the bound instance handle through the communicator is as follows:

    ```c
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    HcclResult ret = HcclCommQueryAssignedCcuIns(comm, &insHandle, &insNum);
    // Currently, a communicator can bind at most one CCU instance, so insNum is 1 on success.
    if (ret != HCCL_SUCCESS || insNum != 1) {
        // Error handling.
    }
    ```

> This API belongs to the HCCL layer (not in the `Hcomm*`/`Ccu*` series), and no standalone API reference page is provided yet.
> For the complete signature, refer to the header file `include/hccl/hccl_ccu_res.h`.
> If the communicator has no bound CCU instance, `HcclCommQueryAssignedCcuIns` returns `HCCL_E_UNAVAIL`.
> In addition, [HcclCommQueryCcuIns](../comms_domain_resource_mgmt/HcclCommQueryCcuIns.md) is used to query the communicator's own CCU instance (creating one if none exists), which is independent of the bound instance queried by this API.

## API Call Sequence

The standard call sequence of kernel registration and launch APIs is as follows:

1. [HcommCcuKernelRegisterStart](HcommCcuKernelRegisterStart.md): Starts a round of kernel registration.
2. [HcommCcuKernelRegister](HcommCcuKernelRegister.md): Registers a kernel function and records its operation sequence.
3. [HcommCcuKernelRegisterEnd](HcommCcuKernelRegisterEnd.md): Ends the current round of registration, translates it into device instructions, and dispatches them.
4. [HcommCcuKernelLaunch](HcommCcuKernelLaunch.md): Launches kernel execution (can be called repeatedly).

Bypass API:

- [HcommCcuGetMemToken](HcommCcuGetMemToken.md): Converts a process virtual address into a memory token usable by the CCU. It has no dependency on the main flow and can be called at any time.

## See Also

- [CCU Quick Start (Including the Complete AllGather Process)](../../../../comm_op_dev_guide/ccu_quick_start.md)
- [CCU Communication Operator Development Guide (Step-by-Step Details)](../../../../comm_op_dev_guide/ccu_comm_op_dev/README.md)

# CcuKernelHandle

<!-- md-trans-meta sourceCommit=a47b7d7f73d45261f35021b3769f2dec0a3c9705 translatedAt=2026-09-28T08:50:21.413Z pushedAt=2026-10-08T03:49:04.892Z -->

## Description

CCU kernel handle, returned after registration by [HcommCcuKernelRegister](../control_plane_api/ccu_kernel_launch_execution/HcommCcuKernelRegister.md) and used to identify a registered kernel. The same handle can be launched multiple times through [HcommCcuKernelLaunch](../control_plane_api/ccu_kernel_launch_execution/HcommCcuKernelLaunch.md).

## Prototype

```c
typedef uint64_t CcuKernelHandle;
```

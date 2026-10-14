# HcclCommSuspend

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:39:12.275Z pushedAt=2026-10-12T06:12:35.574Z -->

> [!NOTE] Note
> This API is reserved and may change in the future. It is not intended for developer use.

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

When an uncorrectable memory error (UCE) occurs on the on-chip memory (the ACL API returns the **ACL_ERROR_RT_DEVICE_MTE_ERROR** error code), you can call this API to set the communicator to the suspended state.

This API suspends the communicator without exiting the host-side process. After the fault is rectified, you can call [HcclCommResume](HcclCommResume.md) to restore the communicator state.

## Function Prototype

```c
HcclResult HcclCommSuspend(HcclComm comm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator to be suspended.<br>For details about the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- This API must be used together with [HcclCommResume](HcclCommResume.md).
- This API cannot be executed concurrently with APIs related to collective communication or point-to-point communication.
- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, a wild pointer, or a destroyed handle.

## Example

```c
// Initialize the device resources.
aclInit(NULL);
aclrtSetDevice(devId);

// Create the communicator.
HcclComm hcclComm;
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);
HcclCommInitRootInfo(8, &rootInfo, 0, &hcclComm);

// Suspend the communicator after an on-chip memory UCE fault is detected.
HcclCommSuspend(hcclComm);

// Resume the communicator after the fault is rectified.
HcclCommResume(hcclComm);

// Destroy the communicator.
HcclCommDestroy(hcclComm);
aclFinalize();
```

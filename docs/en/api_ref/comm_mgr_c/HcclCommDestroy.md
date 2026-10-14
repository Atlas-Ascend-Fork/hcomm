# HcclCommDestroy

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:37:14.739Z pushedAt=2026-10-12T06:12:35.570Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

Destroys a specified HCCL communicator.

## Function Prototype

```c
HcclResult HcclCommDestroy(HcclComm comm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Pointer to the communicator to be destroyed.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- This API supports cross-thread call:
  - When the communicator is in a link establishment stuck state or an unoccupied state, this API can be called across threads to destroy the communicator, and **HCCL_SUCCESS** is returned.

    After the communicator is successfully destroyed, the ongoing communication operators will directly exit with an error without waiting for the timeout period, and an ERROR-level log is printed. The log keyword is "Terminating operation due to external request".

  - When the communicator is in a non-link establishment stuck state, or is in another occupied state (for example, during communicator link establishment or communication operator execution), calling this API across threads returns the **HCCL_E_AGAIN** error, and a WARNING-level log is printed. The log keyword is "[HcclCommDestroy] comm is in use, please try again later".

- In the UB Memory scenario, it is recommended to first call [HcclCommSymWinDeregister](HcclCommSymWinDeregister.md) to deregister the symmetric memory window, and then call this API to destroy the communicator.
  If there are still UB Memory windows that have not been deregistered, this API releases their local resources as a fallback when destroying the communicator. In the A3 and URMA scenarios, it is also not required to deregister the symmetric memory window first: when the communicator is destroyed, internal resource management traverses all registered symmetric memory windows, automatically performs deregistration, and releases the corresponding resources (such as symmetric virtual addresses and device-side window copies). Users do not need to call the deregistration API, and destruction will not fail due to the presence of underegistered windows.

- In multi-thread scenarios, ensure the call sequence of HCCL APIs. After the communicator is destroyed by calling this API, other collective communication APIs are no longer supported.
- The **comm** passed in must be a valid communicator handle that has not been destroyed. Do not pass a wild pointer or a destroyed communicator handle.

## Example

```c
// Initialize device resources.
aclInit(NULL);
uint32_t rankSize = 2;
int32_t devices[rankSize] = {0, 1};
HcclComm comms[rankSize];
// Specify the device used for collective communication operations.
for (uint32_t i = 0; i < rankSize; i++) {
    aclrtSetDevice(devices[i]);
}
// Initialize the communicator.
HcclCommInitAll(rankSize, devices, comms);
// Destroy the communicator.
for (uint32_t i = 0; i < rankSize; i++) {
    HcclCommDestroy(comms[i]);
}
// Deinitialize the device resources.
aclFinalize();
```

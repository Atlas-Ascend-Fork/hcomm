# HcclCommInitAll

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:10:11.099Z pushedAt=2026-10-08T07:39:12.132Z -->

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

In single-server communication scenarios, creates the communicators of multiple devices (one device corresponds to one thread) through one process. During communicator initialization, devices\[0\] serves as the root rank to automatically collect cluster information.

## Function Prototype

```c
HcclResult HcclCommInitAll(uint32_t ndev, int32_t* devices, HcclComm* comms)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| ndev | Input | Number of devices in the communicator. |
| devices | Input | List of devices in the communicator. The value is the logical ID of each device, which can be queried by running the **npu-smi info -m** command. HCCL creates the communicators in the order in which devices are set.<br>Note that the input device list cannot contain duplicate device IDs. |
| comms | Output | Array of generated communicator handles. Its size is ndev * sizeof(HcclComm).<br>For details about the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- This API is supported only in single-server communication scenarios, not in multi-server communication scenarios.
- When multiple threads call communication operation APIs (for example, **HcclAllReduce**), ensure that the time difference between calls to communication operation APIs in different threads does not exceed the time specified by the environment variable [HCCL_CONNECT_TIMEOUT](https://gitcode.com/cann/hccl/blob/master/docs/en/user_guide/hccl_env/HCCL_CONNECT_TIMEOUT.md), avoiding link establishment timeout.
- A single device cannot call multiple communication operation APIs at the same time.

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
// Deinitialize device resources.
aclFinalize();
```

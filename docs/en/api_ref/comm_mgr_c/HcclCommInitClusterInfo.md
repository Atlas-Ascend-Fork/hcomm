# HcclCommInitClusterInfo

<!-- md-trans-meta sourceCommit=1435fccbb795a1ecf7fa4675cb37d8bf3e83f779 translatedAt=2026-10-08T11:38:12.082Z pushedAt=2026-10-12T06:12:35.573Z -->

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

Initializes HCCL based on the rank table and creates an HCCL communicator.

The rank table file is a JSON-format file that configures the NPU resource information involved in collective communication. For details about the rank table file configuration, see the [cluster information configuration](https://gitcode.com/cann/hccl/blob/9.2.0/docs/zh/user_guide/cluster_info_config/README.md).

## Function Prototype

```c
HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| **clusterInfo** | Input | Path of the rank table file (including the file name), with a maximum length of 4096 bytes as a string, including the terminator. |
| **rank** | Input | ID of the current rank.<br>Note that the value of this parameter must be consistent with the value of the corresponding **rank_id** field in the rank table. |
| **comm** | Output | Returns the initialized communicator to the caller as a pointer.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

An error will be reported when initialization is performed repeatedly.

## Example

```c
// Initialize device resources.
aclInit(NULL);
// Path of the rank table configuration file.
char *rankTableFile = "/path/rank_table.json";
// Specify the device ID used for collective communication operations.
uint32_t devId = 0;
aclrtSetDevice(devId);
// Create the communicator.
HcclComm hcclComm;
// In this example, devId is used as the rank ID of the current rank.
HcclCommInitClusterInfo(rankTableFile, devId, &hcclComm);
// Destroy the communicator.
HcclCommDestroy(hcclComm);
// Deinitialize device resources.
aclFinalize();
```

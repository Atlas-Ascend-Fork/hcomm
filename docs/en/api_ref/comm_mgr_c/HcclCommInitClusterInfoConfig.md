# HcclCommInitClusterInfoConfig

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:10:59.598Z pushedAt=2026-09-28T09:31:10.648Z -->

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

Initializes HCCL based on the rank table and creates an HCCL communicator with specific configurations.

## Function Prototype

```c
HcclResult HcclCommInitClusterInfoConfig(const char *clusterInfo, uint32_t rank, HcclCommConfig *config, HcclComm *comm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| clusterInfo | Input | Path to the rank table file (including the file name). As a string, its maximum length is 4096 bytes, including the terminator. |
| rank | Input | ID of the current rank.<br>Note that the value of this parameter must be consistent with the value of the corresponding **rank_id** field in the rank table. |
| config | Input | Communicator configuration items, including the buffer size, deterministic computation switch, communicator name, and communication operator expansion mode. Ensure that the configured parameters are within the valid value range. For details about the meanings and priorities of the parameters in **HcclCommConfig**, see the definition of [HcclCommConfig](./data_type_definition/HcclCommConfig.md).<br>Note: The **config** passed in must be initialized by calling [HcclCommConfigInit](HcclCommConfigInit.md) first. |
| comm | Output | Returns the initialized communicator to the caller as a pointer.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

Repeated initialization of the same communicator is not supported.

## Example

```c
// Initialize device resources.
aclInit(NULL);
// Path to the rank table configuration file.
const char *rankTableFile = "/path/to/rank_table.json";
// Specify the device used for collective communication operations.
uint32_t rankSize = 8;
uint32_t devId = 0;
aclrtSetDevice(devId);
// Create and initialize the communicator configuration items.
HcclCommConfig config;
HcclCommConfigInit(&config);
// Modify the communicator configuration as needed.
config.hcclBufferSize = 50;  // Buffer size of the shared data, in MB. The value must be >= 1, and the default value is 200.
strncpy(config.hcclCommName, "comm_1", COMM_NAME_MAX_LENGTH - 1);
config.hcclCommName[COMM_NAME_MAX_LENGTH - 1] = '\0';
// Initialize the communicator.
HcclComm hcclComm;
// In this example, devId is used as the rank ID of the current rank.
HcclCommInitClusterInfoConfig(rankTableFile, devId, &config, &hcclComm);
// Destroy the communicator.
HcclCommDestroy(hcclComm);
// Deinitialize device resources.
aclFinalize();
```

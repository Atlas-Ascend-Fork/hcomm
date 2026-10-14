# HcclCreateSubCommConfig

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:40:51.736Z pushedAt=2026-10-12T06:12:35.578Z -->

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
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

Creates a sub-communicator with specific configurations from an existing global communicator.

This sub-communicator creation method does not require socket link establishment or rank information exchange, and can be used for rapid communicator creation in the event of service faults.

Note:

If the load is unbalanced among devices in the network, the sub-communicator created by this API may time out during link establishment due to asynchronization among devices. In this case, you can increase the link establishment timeout between devices by using the environment variable **HCCL_CONNECT_TIMEOUT**. Configuration example:

```bash
export HCCL_CONNECT_TIMEOUT=600
```

## Function Prototype

```c
HcclResult HcclCreateSubCommConfig(HcclComm *comm, uint32_t rankNum, uint32_t *rankIds, uint64_t subCommId, uint32_t subCommRankId, HcclCommConfig *config, HcclComm *subComm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Global communicator to be split.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| rankNum | Input | Number of ranks in the sub-communicator to be split. |
| rankIds | Input | Array of IDs of the ranks in the sub-communicator within the global communicator.<br>Note: This array must be ordered. The subscript of each rank in the array is mapped to its rank ID in the sub-communicator. |
| subCommId | Input | Identifier of the current sub-communicator, which is user-defined.<br>  - If the sub-communicator name **hcclCommName** is not configured in the **config** parameter, the system uses `{Global communicator name}_sub_{subCommId}` as the sub-communicator name. In this case, ensure that **subCommId** is unique within the global communicator.<br>  - If the sub-communicator name **hcclCommName** is configured in the **config** parameter, the name configured in **config** takes precedence, and this parameter is no longer verified. |
| subCommRankId | Input | Rank ID of the current rank in the sub-communicator.<br>Configure it as the subscript index of the current rank in the **rankIds** array. |
| config | Input | Communicator configuration items, including the buffer size, deterministic computation switch, communicator name, and communication operator expansion mode. Ensure that the configured parameters are within the valid value range. For details about the meanings and priorities of the parameters in **HcclCommConfig**, see the definition of [HcclCommConfig](./data_type_definition/HcclCommConfig.md).<br>Note: The **config** passed in must be initialized by calling [HcclCommConfigInit](HcclCommConfigInit.md) first. |
| subComm | Output | Returns the initialized sub-communicator to the caller as a pointer.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- Ranks belonging to the same sub-communicator must pass the same **rankNum**, **rankIds**, **subCommId**, and **config** when calling this API.
- Ranks that do not need to create a sub-communicator should pass **rankIds=nullptr** and **subCommId=0xFFFFFFFF**. In this case, the **subCommId** parameter is not verified.
- Only splitting a sub-communicator from a global communicator is supported. Further splitting a sub-communicator within a sub-communicator is not supported.
- Calling this API to split a sub-communicator from multiple processes or threads on a single device is not supported.
- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, a wild pointer, or a destroyed handle.

## Example

```c
// Initialize the global communicator.
HcclComm globalHcclComm;
HcclCommInitClusterInfo(rankTableFile, devId, &globalHcclComm);
// Communicator configuration.
HcclCommConfig config;
HcclCommConfigInit(&config);
config.hcclBufferSize = 50;
strcpy(config.hcclCommName, "comm_1");
// Initialize the sub-communicator.
HcclComm hcclComm;
uint32_t rankIds[4] = {0, 1, 2, 3};  // Rank list of the sub-communicator
// Set the rank ID of the current rank in the sub-communicator to 0.
HcclCreateSubCommConfig(&globalHcclComm, 4, rankIds, 1, 0, &config, &hcclComm);
```

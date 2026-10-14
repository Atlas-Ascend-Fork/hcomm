# HcclConfigGetInfo

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:40:51.035Z pushedAt=2026-10-12T06:12:35.578Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Obtains the HCCL configuration information of a specified communicator.

Queries the corresponding configuration information based on the configuration item type and writes it into the buffer provided by the caller. Currently, it supports querying the expansion mode of communication operators, the communication algorithm configuration string, and the number of UB multi-channels.

## Function Prototype

```c
HcclResult HcclConfigGetInfo(HcclComm comm, HcclConfigType cfgType, uint32_t infoLen, void *info);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator handle.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| cfgType | Input | Type of the configuration item to query. For the definition of HcclConfigType, see [HcclConfigType](./data_type_definition/HcclConfigType.md). |
| infoLen | Input | Size (in bytes) of the target configuration type. When querying **HCCL_CONFIG_TYPE_OP_EXPANSION_MODE**, it must be equal to the actual size of the configuration type to be queried. When querying **HCCL_CONFIG_TYPE_HCCL_ALGO**, it must be no less than **HCCL_COMM_ALGO_MAX_LENGTH** bytes. When querying **HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM**, it must be equal to **sizeof(uint32_t)**. |
| info | Output | Output buffer for the configuration information, which must be aligned to the target configuration type and writable. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, a wild pointer, or a destroyed handle.

## Example

```c
HcclConfigTypeOpExpansionMode mode;
uint32_t size = sizeof(HcclConfigTypeOpExpansionMode); // Must be equal to the target type size.
HcclResult ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, size, &mode);

// Query the communication algorithm string.
char algoInfo[HCCL_COMM_ALGO_MAX_LENGTH];
uint32_t algoSize = HCCL_COMM_ALGO_MAX_LENGTH; // Must be no less than HCCL_COMM_ALGO_MAX_LENGTH.
HcclResult ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_HCCL_ALGO, algoSize, algoInfo);

// Query the number of UB multi-channels.
uint32_t multiChannelNum = 0;
uint32_t numSize = sizeof(uint32_t); // Must be equal to the target type size.
HcclResult ret = HcclConfigGetInfo(comm, HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM, numSize, &multiChannelNum);
```

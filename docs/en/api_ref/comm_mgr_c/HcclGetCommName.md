# HcclGetCommName

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:41:28.760Z pushedAt=2026-10-12T06:12:35.580Z -->

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

Obtains the name of the communicator where the specified communication operation is performed.

## Function Prototype

```c
HcclResult HcclGetCommName(HcclComm comm, char* commName)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| **comm** | Input | Communicator where the collective communication operation is performed.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |
| **commName** | Output | Obtained communicator name.<br>Type: char*, with a maximum length of 128. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, a wild pointer, or a destroyed handle.

## Example

```c
// Initialize the communicator.
HcclComm comm;
// Query the communicator name.
char commName[128] = {0};
HcclResult ret = HcclGetCommName(comm, commName);
// Handle errors.
if (ret != HCCL_SUCCESS) {
}
```

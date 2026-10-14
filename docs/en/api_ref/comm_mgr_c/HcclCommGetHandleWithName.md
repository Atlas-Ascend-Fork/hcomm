# HcclCommGetHandleWithName

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:09:33.517Z pushedAt=2026-09-28T09:20:39.496Z -->

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
- Atlas inference products: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

Obtains a handle to the corresponding communicator based on the communicator name.

## Function Prototype

```c
HcclResult HcclCommGetHandleWithName(const char* commName, HcclComm* comm)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| commName | Input | Communicator name.<br>Type: const char*, with a maximum length of 128. |
| comm | Output | Handle of the obtained communicator.<br>For the definition of the HcclComm type, see [HcclComm](./data_type_definition/HcclComm.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
// Define the name of the communicator whose handle is to be obtained.
char commName[128] = "group_name_0";
HcclComm comm;
// Obtain the handle of the communicator corresponding to the communicator name.
HcclCommGetHandleWithName(commName, &comm);
```

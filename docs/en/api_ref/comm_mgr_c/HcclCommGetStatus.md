# HcclCommGetStatus

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:09:21.807Z pushedAt=2026-09-28T09:22:07.857Z -->

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

Obtains the communicator status during operator dispatch to determine whether an operator can be dispatched.

## Function Prototype

```c
HcclResult HcclCommGetStatus(const char *commId, HcclCommStatus *status)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| commId | Input | Communicator name.<br>Type: const char*, with a maximum length of 128. |
| status | Output | Communicator status. For the definition of the HcclCommStatus type, see [HcclCommStatus](./data_type_definition/HcclCommStatus.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

Used in custom communication operator scenarios.

## Example

```c
HcclComm comm;
char commName[128];
HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;

... // Create the communicator.

HCCLCHECK(HcclGetCommName(comm, commName));
HCCLCHECK(HcclCommGetStatus(commName, &commStatus));
```

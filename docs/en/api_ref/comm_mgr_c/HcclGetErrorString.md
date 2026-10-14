# HcclGetErrorString

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:22:43.114Z pushedAt=2026-09-29T01:14:11.667Z -->

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

Parses an error code of the **HcclResult** type.

## Function Prototype

```c
const char *HcclGetErrorString(HcclResult code)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| code | Input | Error code to be parsed, which is of the [HcclResult](./data_type_definition/HcclResult.md) type. |

## Return Value

A `const char *` pointer to the string description corresponding to the error code of the [HcclResult](./data_type_definition/HcclResult.md) type.

## Constraints

None

## Example

```c
// Initialize device resources.
aclInit(NULL);
uint32_t devId = 0;
aclrtSetDevice(devId);

// Create the communicator.
HcclComm hcclComm;
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);
HcclCommInitRootInfo(8, &rootInfo, 0, &hcclComm);

// Query the asynchronous error of the communicator and parse the error code.
HcclResult asyncError = HCCL_SUCCESS;
HcclGetCommAsyncError(hcclComm, &asyncError);
if (asyncError != HCCL_SUCCESS) {
    const char *errStr = HcclGetErrorString(asyncError);
    printf("comm async error: %s\n", errStr);
}

// Destroy the communicator.
HcclCommDestroy(hcclComm);
aclFinalize();
```

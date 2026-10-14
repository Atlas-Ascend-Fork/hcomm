# HcclCommRegCommStateCallback

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:12:58.643Z pushedAt=2026-09-28T10:36:26.515Z -->

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

Registers a callback function of the `HcclCommStateCallback` type defined by HCCL with HCOMM. The callback function is called at different phases of the communicator.

## Function Prototype

```c
HcclResult HcclCommRegCommStateCallback(const char *regName, HcclCommStateCallback cb, void *args)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| regName | Input | Name to be registered.<br>**const char*** type, with a maximum length of 160 bytes. |
| cb | Input | Type of the callback function to be registered with HCOMM. For details about the **HcclCommStateCallback** type, see [HcclCommStateCallback](./data_type_definition/HcclCommStateCallback.md). |
| args | Input | User context pointer passed to the callback function. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

The registered callback function is called before and after [HcclCommResume](./HcclCommResume.md) (resumption phase) and [HcclCommDestroy](./HcclCommDestroy.md) (destruction phase) are called. The specific phase is identified by the [HcclCommStatePhase](./data_type_definition/HcclCommStatePhase.md) enumeration value.

## Example

```c
// Callback function to be registered, which can be implemented as required.
HcclResult myCallback(HcclComm comm, HcclCommStatePhase state, void *args)
{
    (void)comm;
    (void)state;
    (void)args;
    return HCCL_SUCCESS;
}
```

```c
uint32_t rankSize = 8;
uint32_t deviceId = 0;

const char *regName = "my_callback";
void *userPtr = "userContext";

// Generate the rank identifier information of the root node.
HcclRootInfo rootInfo;
HCCLCHECK(HcclGetRootInfo(&rootInfo));
// Initialize the communicator.
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm));

// Register the callback function to be called during communicator resumption and destruction. regName is the name of the callback function to register, and userPtr is the context parameter pointer to be passed to the callback function.
HcclCommRegCommStateCallback(regName, myCallback, userPtr);
// Assume that the communicator has been suspended through the HcclCommSuspend API or the aclrtDeviceTaskAbort API provided by ACL. Resume the communicator.
HCCLCHECK(HcclCommResume(hcclComm));
// Destroy the communicator.
HCCLCHECK(HcclCommDestroy(hcclComm));
```

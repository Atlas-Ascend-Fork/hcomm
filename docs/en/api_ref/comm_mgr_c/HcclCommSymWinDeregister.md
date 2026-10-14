# HcclCommSymWinDeregister

<!-- md-trans-meta sourceCommit=b0b5276bf70c24e578edb2b23d612d0b28163287 translatedAt=2026-10-08T11:39:10.919Z pushedAt=2026-10-12T06:12:35.573Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
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

Deregisters a registered symmetric memory window and releases the symmetric memory window resources. This API does not release the memory allocated by the user. The user still needs to release the corresponding memory in the same way as it was allocated.

<!-- npu="950" id6 -->
- For Ascend 950PR&950DT products, this API supports the URMA scenario and the UB Memory scenario.
<!-- end id6 -->
<!-- npu="A3" id7 -->
- For Atlas A3 products, this API supports the HCCS link communication scenario.
<!-- end id7 -->

## Function Prototype

```c
HcclResult HcclCommSymWinDeregister(HcclCommSymWindow winHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| winHandle | Input | Handle of the registered symmetric window.<br>For the definition of the HcclCommSymWindow type, see [HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- This API must be used together with [HcclCommSymWinRegister](HcclCommSymWinRegister.md).
- The supported scope is the same as that of [HcclCommSymWinRegister](HcclCommSymWinRegister.md).
- Ensure that all ranks in the communicator call this API at the same time to release the symmetric window resources.
- This API does not support concurrent calls by multiple threads. The caller must ensure that calls to this API from different threads are executed serially.

## Example

See [Example](HcclCommSymWinRegister.md#example).

# HcommBatchModeEnd

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:17:10.074Z pushedAt=2026-10-12T06:12:35.663Z -->

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
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Submits and triggers the execution of all operations cached in batch mode. All data plane API calls made between **HcommBatchModeStart** and **HcommBatchModeEnd** are executed at this point.

## Function Prototype

```c
int32_t HcommBatchModeEnd(const char *batchTag)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| batchTag | Input | Batch task identifier, which must be consistent with the **batchTag** passed to **HcommBatchModeStart**. |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

- **HcommBatchModeStart** and **HcommBatchModeEnd** must be called in pairs and executed in the same thread.
- Concurrency note: This API must be called serially.

<!-- npu="950" id6 -->
- Only the Ascend 950PR&950DT products support batch mode and immediate execution mode (without calling the batch APIs). Other products must use batch mode.
<!-- end id6 -->

## Example

```c
char *tag = "";
// Start batch mode (temporary batch task).
HcommBatchModeStart(tag);

// Data plane APIs called in batch mode are not executed immediately.
// ...

// End batch mode and trigger execution.
HcommBatchModeEnd(tag);
```

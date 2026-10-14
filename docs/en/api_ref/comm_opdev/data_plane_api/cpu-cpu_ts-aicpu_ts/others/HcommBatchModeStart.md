# HcommBatchModeStart

<!-- md-trans-meta sourceCommit=01a4daa48785c34db54992320c7441d255cce5c2 translatedAt=2026-10-08T12:17:34.694Z pushedAt=2026-10-12T06:12:35.663Z -->

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

Enables batch mode. All data plane API calls (such as **HcommLocalCopy** and **HcommWrite**) between **HcommBatchModeStart** and **HcommBatchModeEnd** are cached and not executed immediately. All operations are submitted and executed together when **HcommBatchModeEnd** is called.

## Function Prototype

```c
int32_t HcommBatchModeStart(const char *batchTag)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| batchTag | Input | Batch task identifier (optional). If **NULL** or an empty string is passed, the task is a temporary batch task and is not cached after execution. If a non-empty string is passed, it is used to identify and manage subsequent batch tasks.<br>Note that in the AI CPU + TS communication engine scenario, task cache management based on a non-empty identifier is not yet fully supported. |

## Return Value

**int32_t**: This API returns **0** on success and a non-zero value on failure.

## Constraints

1. **HcommBatchModeStart** and **HcommBatchModeEnd** must be called in pairs and executed in the same thread.
2. Operations cached in batch mode are actually executed only after **HcommBatchModeEnd** is called.
3. Only the Ascend 950PR&950DT products support batch mode and immediate execution mode (without calling the batch APIs). Other products must use batch mode.
4. Concurrency note: This API must be called serially.

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

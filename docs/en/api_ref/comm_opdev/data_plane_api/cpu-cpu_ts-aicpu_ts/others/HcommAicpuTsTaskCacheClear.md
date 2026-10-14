# HcommAicpuTsTaskCacheClear

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:44:31.913Z pushedAt=2026-10-08T03:17:16.236Z -->

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
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Clears the cache entry corresponding to the specified **tag** in the AICPU Task Cache and releases the cache space occupied by the cache entry. This API is used to clear caches that are no longer needed in scenarios such as communicator destruction or fast fault recovery.

## Function Prototype

```c
HcommResult HcommAicpuTsTaskCacheClear(const char *tag)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| tag | Input | Cache identifier, specifying the tag corresponding to the cache entry to be cleared. |

## Return Value

**HcommResult**: This API returns **0** on success and other values on failure.

## Constraints

1. This API can be called only on the device side in AI CPU mode.
2. If the cache entry corresponding to the specified **tag** does not exist, the API still returns success.

## Example

```c
const char *tag = "op_tag_example";

// Clear the cache of the specified tag.
if (HcommAicpuTsTaskCacheClear(tag) != HCCL_SUCCESS) {
    return 1;
}
```

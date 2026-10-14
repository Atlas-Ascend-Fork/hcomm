# HcommAicpuTsTaskCacheEnd

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:44:35.446Z pushedAt=2026-10-08T10:32:13.942Z -->

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

Notifies the AICPU Task Cache to stop caching tasks in the cache miss scenario. This API is called after operator expansion is complete. It will submit a cache entry to update the internal address refresh information and token information as well as count the cache space consumption.

## Function Prototype

```c
HcommResult HcommAicpuTsTaskCacheEnd(const char *tag)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| tag | Input | Cache identifier, which must be consistent with the tag passed to **HcommAicpuTsTaskCacheStart**. |

## Return Value

**HcommResult**: This API returns **0** on success and other values on failure.

## Constraints

1. This API can be called only on the device side in AI CPU mode.
2. This API is called only in the cache miss scenario (that is, when **HcommAicpuTsTaskCacheLookup** returns **isHit** as **false**). Otherwise, an error is returned.
3. The **tag** must be consistent with the **tag** passed to **HcommAicpuTsTaskCacheStart**. Otherwise, an error is returned.
4. This API must be used together with **HcommAicpuTsTaskCacheStart**. After this API is called, the cache context is reset.

## Example

```c
const char *tag = "op_tag_example";
void *addrs[] = {inputAddr, outputAddr};
uint64_t sizes[] = {inputSize, outputSize};
uint64_t count = 2;

// Start caching on a cache miss.
HcommAicpuTsTaskCacheStart(tag, addrs, sizes, count);

// Perform operator expansion (dispatch SQE/WQE, etc.).
// ...

// End caching.
if (HcommAicpuTsTaskCacheEnd(tag) != HCCL_SUCCESS) {
    return 1;
}
```

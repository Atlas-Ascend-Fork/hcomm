# HcommAicpuTsTaskCacheExecute

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:45:39.263Z pushedAt=2026-10-08T03:31:08.262Z -->

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

Updates the cached task based on the newly passed memory address information and dispatches it for execution in the cache hit scenario. This API updates the address fields and related token information in the SQE/WQE based on the new memory base addresses, and then submits all tasks for execution in dispatch order.

## Function Prototype

```c
HcommResult HcommAicpuTsTaskCacheExecute(const char *tag, void **addrs, uint64_t *sizes, uint64_t count)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| tag | Input | Cache identifier, which must be consistent with the **tag** passed to **HcommAicpuTsTaskCacheLookup**. |
| addrs | Input | Array of new memory base address information, used to update the address fields in the cache entry. |
| sizes | Input | Array of new memory size information, which must be consistent with the memory sizes saved on a cache miss. |
| count | Input | Length of the memory information array, which must be consistent with the count saved on a cache miss. |

## Return Value

**HcommResult**: This API returns **0** on success and other values on failure.

## Constraints

1. This API can be called only on the device side in AI CPU mode.
2. This API is called only in the cache hit scenario (that is, when **HcommAicpuTsTaskCacheLookup** returns **isHit** as **true**). Otherwise, an error is returned.
3. The tag must be consistent with the tag passed to **HcommAicpuTsTaskCacheLookup**. Otherwise, an error is returned.
4. **count** and **sizes** must be consistent with those passed to **HcommAicpuTsTaskCacheStart** on a cache miss. Otherwise, an error is returned.
5. After this API is called, the cache context is reset.

## Example

```c
const char *tag = "op_tag_example";
bool isHit = false;
void *addrs[] = {inputAddr, outputAddr};
uint64_t sizes[] = {inputSize, outputSize};
uint64_t count = 2;

// Look up the cache.
HcommAicpuTsTaskCacheLookup(tag, &isHit);

if (isHit) {
    // Cache hit: update the addresses and dispatch.
    if (HcommAicpuTsTaskCacheExecute(tag, addrs, sizes, count) != HCCL_SUCCESS) {
        return 1;
    }
} else {
    // Cache miss: start caching.
    HcommAicpuTsTaskCacheStart(tag, addrs, sizes, count);
    // Expand the operator...
    HcommAicpuTsTaskCacheEnd(tag);
}
```

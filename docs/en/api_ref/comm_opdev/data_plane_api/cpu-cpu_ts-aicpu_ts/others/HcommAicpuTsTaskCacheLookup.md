# HcommAicpuTsTaskCacheLookup

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:46:10.335Z pushedAt=2026-10-08T03:33:55.479Z -->

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

Checks whether the operator expansion task corresponding to the specified tag is already cached in the AICPU Task Cache. This API is the entry API for using the AICPU Task Cache. Based on the lookup result, it determines whether to follow the cache hit path (calling **HcommAicpuTsTaskCacheExecute**) or the cache miss path (calling **HcommAicpuTsTaskCacheStart** and **HcommAicpuTsTaskCacheEnd**).

## Function Prototype

```c
HcommResult HcommAicpuTsTaskCacheLookup(const char *tag, bool *isHit)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| tag | Input | Cache identifier, used to uniquely identify a group of operator expansion tasks. |
| isHit | Output | Whether it is a cache hit. **true** indicates cached (cache hit), and **false** indicates not cached (cache miss). |

## Return Value

**HcommResult**: This API returns **0** on success and other values on failure.

## Constraints

1. This API can be called only on the device side in AI CPU mode.
2. Operator expansion operations with the same tag must be executed serially and cannot be concurrent. The reason is that the expansion process uses a "reserve first, backfill later" cache write mechanism. In concurrent scenarios, a subsequent thread may hit a cache entry that is not yet complete, causing an exception. For example, when the tag contains a communicator identifier (commId), operators with the same tag belong to the same communicator and are necessarily expanded in order.
3. This API must be used together with **HcommAicpuTsTaskCacheStart**/**HcommAicpuTsTaskCacheEnd** (cache miss scenario) or **HcommAicpuTsTaskCacheExecute** (cache hit scenario), and the tag passed to the subsequent APIs must be consistent with the tag of this API.

## Example

```c
const char *tag = "op_tag_example";
bool isHit = false;
void *addrs[] = {inputAddr, outputAddr};
uint64_t sizes[] = {inputSize, outputSize};
uint64_t count = 2;

// Look up the cache.
if (HcommAicpuTsTaskCacheLookup(tag, &isHit) != HCCL_SUCCESS) {
    return 1;
}

if (isHit) {
    // Cache hit: update and dispatch.
    HcommAicpuTsTaskCacheExecute(tag, addrs, sizes, count);
} else {
    // Cache miss: start caching.
    HcommAicpuTsTaskCacheStart(tag, addrs, sizes, count);
    // Execute operator expansion...
    HcommAicpuTsTaskCacheEnd(tag);
}
```

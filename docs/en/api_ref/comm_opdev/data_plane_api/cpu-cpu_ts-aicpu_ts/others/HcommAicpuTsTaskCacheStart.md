# HcommAicpuTsTaskCacheStart

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:46:23.392Z pushedAt=2026-10-08T03:36:44.850Z -->

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

Notifies the AICPU Task Cache to start caching tasks in the cache miss scenario. This API is called before operator expansion. It saves the passed-in memory base address and size information to a cache entry. The SQE/WQE issued during subsequent operator expansion are then captured and cached.

## Function Prototype

```c
HcommResult HcommAicpuTsTaskCacheStart(const char *tag, void **addrs, uint64_t *sizes, uint64_t count)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| tag | Input | Cache identifier, which must be consistent with the **tag** passed to **HcommAicpuTsTaskCacheLookup**. |
| addrs | Input | Array of memory base address information, pointing to the base addresses of the user input/output memory. |
| sizes | Input | Array of memory size information, corresponding one-to-one with **addrs**. |
| count | Input | Length of the memory information array, that is, the number of memory segments. |

## Return Value

**HcommResult**: This API returns **0** on success and other values on failure.

## Constraints

1. This API can be called only on the device side in AI CPU mode.
2. This API is called only in the cache miss scenario (that is, when **HcommAicpuTsTaskCacheLookup** returns **isHit** as **false**). Otherwise, an error is returned.
3. The **tag** must be consistent with the **tag** passed to **HcommAicpuTsTaskCacheLookup**. Otherwise, an error is returned.
4. This API must be used together with **HcommAicpuTsTaskCacheEnd**, with the operator expansion performed between the two calls.
5. If the AICPU Task Cache is full, no new cache entry is added. In this case, operator expansion still executes normally but is not cached.

## Example

```c
const char *tag = "op_tag_example";
void *addrs[] = {inputAddr, outputAddr};
uint64_t sizes[] = {inputSize, outputSize};
uint64_t count = 2;

// Start caching on a cache miss.
if (HcommAicpuTsTaskCacheStart(tag, addrs, sizes, count) != HCCL_SUCCESS) {
    return 1;
}

// Perform operator expansion (issue SQE/WQE, etc.).
// ...

// End caching.
HcommAicpuTsTaskCacheEnd(tag);
```

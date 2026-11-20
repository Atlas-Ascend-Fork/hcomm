# HcclRankGraphGetInstSizeListByLayer

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:42:00.110Z pushedAt=2026-09-30T02:47:53.223Z -->

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

Given a communicator and a topology layer number, queries the number of topology instances at this layer and the number of ranks contained in each instance.

<!-- npu="950" id6 -->
When a virtual grouping view has been built for the layer, the API returns the number of ranks in each virtual segment group, and `listSize` is the number of virtual groups. Each original topology instance is partitioned into contiguous segments of rank IDs in ascending order. Groups from different instances are not merged. The returned list is ordered by the first rank ID of each group in ascending order. Layers without a virtual grouping view retain the original topology instance semantics.

For example, suppose two topology instances at an intermediate layer contain ranks `{0,1,2,3,8,9,10,11}` and `{4,5,6,7,12,13,14,15}`, respectively. With a virtual grouping view, the API returns `instSizeList=[4,4,4,4]` and `listSize=4`, instead of `[8,8]` and `2` for the original topology instances.
<!-- end id6 -->

![Topology model](figures/topo_model.png)

Take the preceding topology model as an example:

- Layer 0 contains two topology instances. For ease of understanding, the topology instance IDs are defined as 0 and 1, and each instance contains three ranks.
- Layer 1 contains one topology instance, which contains six ranks.

## Function Prototype

```c
HcclResult HcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t **instSizeList, uint32_t *listSize)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer number. |
| instSizeList | Output | List of the number of ranks contained in each topology instance at this layer. |
| listSize | Output | Size of **instSizeList**, that is, the number of topology instances at this layer. |

<!-- npu="950" id7 -->
When a virtual grouping view has been built for the layer, `instSizeList` contains the rank counts of the contiguous virtual segment groups, and `listSize` is the number of virtual groups.
<!-- end id7 -->

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

<!-- npu="950" id8 -->
- On Ascend 950PR&950DT products, virtual grouping is enabled when the communicator rank graph is built through the RootInfoDetect path, and disabled when built through the rank table file path. Sub-communicators inherit this policy. Snapshot recovery preserves the policy when it is stored in the snapshot. Snapshots saved by versions without the grouping policy recover with virtual grouping disabled, returning the original topology instance size list.
- The current virtual grouping policy skips the lowest layer joined by the current rank and evaluates all other joined layers, including the highest layer. A virtual grouping view is built for a layer only if at least one topology instance in that layer contains non-contiguous rank IDs. For the lowest joined layer, or when the rank IDs within every instance of the layer are contiguous, the API returns the original topology instance size list. With only one joined layer, the API also returns the original topology instance size list.
- Virtual grouping views are built before rank graph initialization completes and cannot be rebuilt after initialization. The following constraints on caching, immutable results, and pointer validity also apply to virtual grouping results.
<!-- end id8 -->

- On Ascend 950PR&950DT products with layer 0 disabled, querying `netLayer=0` still returns the member counts of the communicator's groups and the group count if local group identity information is retained; a sub-communicator counts only its own members. This does not imply usable layer 0 topology instances. Without that identity information, the query returns `HCCL_E_PARA`.
- The returned memory is managed by the library. The caller must not release it.
- Returned results are read-only data cached by the `netLayer` parameter within each communicator. The result returned by each call remains unchanged. Subsequent calls within the same communicator, whether with the same or different parameters, do not invalidate previously returned results. Each communicator maintains an independent cache.
- **comm** must be a valid communicator handle obtained from a communicator creation API and must remain valid throughout the call. Null, dangling, or destroyed handles are not allowed.
- Returned pointers become invalid only when the communicator is rebuilt during snapshot recovery or destroyed. Snapshot recovery and destruction must not run concurrently with queries.
- Do not modify the returned memory. A deep copy is not required before using it across threads.
- Thread safety: On Ascend 950PR&950DT products, concurrent calls to this API from multiple threads within the same communicator are supported. Atlas A2 and Atlas A3 products do not support concurrent calls from multiple threads.
- Communicator destruction (**HcclCommDestroy**) must not run concurrently with this API. Before destruction, ensure that all query threads have finished all calls involving the communicator.

## Example

Take the topology model in [Description](#description) as an example:

```c
HcclComm comm;
uint32_t *instSizeList;
uint32_t listSize;
HcclRankGraphGetInstSizeListByLayer(comm, 0, &instSizeList, &listSize);
// instSizeList=[3,3], listSize=2
HcclRankGraphGetInstSizeListByLayer(comm, 1, &instSizeList, &listSize);
// instSizeList=[6], listSize=1
```

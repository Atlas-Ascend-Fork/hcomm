# HcclRankGraphGetRanksByTopoInst

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:43:54.831Z pushedAt=2026-09-30T02:59:55.722Z -->

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

Given a communicator and a topology layer ID, queries the rank information contained in the specified topology instance corresponding to the current rank.

## Function Prototype

```c
HcclResult HcclRankGraphGetRanksByTopoInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, uint32_t **ranks, uint32_t *rankNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| topoInstId | Input | Topology instance ID (existing in the topology file). |
| ranks | Output | List of ranks contained in the corresponding topology instance. |
| rankNum | Output | Number of ranks. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- When virtual grouping is enabled and the queried layer has virtual groups, only ranks shared by the specified topology instance and the current rank's virtual group are returned. An empty intersection returns HCCL_E_NOT_FOUND. Check the return code before using the output parameters.
- On Ascend 950PR&950DT products with layer selection configured, querying a disabled layer, including `netLayer=0`, returns `HCCL_E_PARA`. Retained local group members are not substituted for members of that layer's topology instance.
- The returned memory is managed by the library. The caller must not release it.
- Returned results are read-only data cached by the (`netLayer`, `topoInstId`) parameter pair within each communicator. The result returned by each call remains unchanged. Subsequent calls within the same communicator, whether with the same or different parameters, do not invalidate previously returned results. Each communicator maintains an independent cache.
- Returned pointers become invalid only when the communicator is rebuilt during snapshot recovery or destroyed. Snapshot recovery and destruction must not run concurrently with queries.

## Example

```c
 // 8-device communicator, the same 8p Mesh
HcclComm comm;
uint32_t netLayer = 0;
uint32_t topoInstId = 0;
uint32_t *ranks;
uint32_t rankNum;
HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInstId, &ranks, &rankNum);
 // ranks = [0,1,2,…,7],  rankNum=8
```

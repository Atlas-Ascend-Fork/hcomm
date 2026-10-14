# HcclRankGraphGetRanksByLayer

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:55:49.118Z pushedAt=2026-10-12T06:12:35.615Z -->

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

Given a communicator and a topology layer ID, returns the list of all rank IDs and the number of ranks in the topology instance where the current rank resides at this layer.

![Topology model](figures/topo_model.png)

Take the preceding topology model as an example:

- Layer 0 contains two topology instances. For ease of understanding, the topology instance IDs are defined as 0 and 1.
- Layer 1 contains one topology instance.

Assume that this API is called on rank 0. If layer 0 is specified, the API returns the rank ID list \[0,1,2\] and the rank quantity 3. If layer 1 is specified, the API returns the rank ID list \[0,1,2,3,4,5\] and the rank quantity 6.

## Function Prototype

```c
HcclResult HcclRankGraphGetRanksByLayer(HcclComm comm, uint32_t netLayer, uint32_t **ranks, uint32_t *rankNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| ranks | Output | Rank ID list. |
| rankNum | Output | Number of ranks. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- The returned memory is managed by the library. The caller must not release it.
- Copy the returned data in a timely manner. Repeated calls in the same communicator may invalidate the previous result.
- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

Take the topology model in [Description](#description) as an example.

For rank 0:

```c
HcclComm commTp;
uint32_t* ranks = nullptr;
uint32_t rankNum;
HcclRankGraphGetRanksByLayer(commTp, 0, &ranks, &rankNum);
// For layer 0 topology, ranks=[0,1,2], rankNum=3.
HcclRankGraphGetRanksByLayer(commTp, 1, &ranks, &rankNum);
// For layer 1 topology, ranks=[0,1,2,3,4,5], rankNum=6.
```

For rank 3:

```c
HcclComm commTp;
uint32_t* ranks = nullptr;
uint32_t rankNum;
HcclRankGraphGetRanksByLayer(commTp, 0, &ranks, &rankNum);
// For layer 0 topology, ranks=[3,4,5], rankNum=3.
HcclRankGraphGetRanksByLayer(commTp, 1, &ranks, &rankNum);
// For layer 1 topology, ranks=[0,1,2,3,4,5], rankNum=6
```

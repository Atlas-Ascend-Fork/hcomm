# HcclRankGraphGetRankSizeByLayer

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:45:09.492Z pushedAt=2026-09-30T03:02:06.920Z -->

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

Given a communicator and a topology layer ID, returns the number of ranks in the topology instance where the current rank resides at this layer.

![Topology model](figures/topo_model.png)

Take the preceding topology model as an example:

- Layer 0 contains two topology instances. For ease of understanding, the topology instance IDs are defined as 0 and 1.
- Layer 1 contains one topology instance.

Assume that this API is called on rank 0. If layer 0 is specified, the API returns 3 as the number of ranks. If layer 1 is specified, the API returns 6 as the number of ranks.

## Function Prototype

```c
HcclResult HcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| rankNum | Output | Number of ranks. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- On Ascend 950PR&950DT products with layer 0 disabled, querying `netLayer=0` still returns the current rank's group member count if local group identity information is retained; a sub-communicator counts only its own members. This does not enable layer 0 communication. Without that identity information, the query returns `HCCL_E_PARA`.

## Example

Take the topology model in [Description](#description) as an example:

```c
HcclComm comm;
uint32_t rankNum;
HcclRankGraphGetRankSizeByLayer(comm, 0, &rankNum);
// rankNum=3
HcclRankGraphGetRankSizeByLayer(comm, 1, &rankNum);
// rankNum=6
```

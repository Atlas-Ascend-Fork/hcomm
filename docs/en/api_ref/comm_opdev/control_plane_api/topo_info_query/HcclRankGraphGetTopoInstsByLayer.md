# HcclRankGraphGetTopoInstsByLayer

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:45:31.077Z pushedAt=2026-09-30T03:03:20.181Z -->

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

Given a communicator and a topology layer ID, queries the set of topology instances where the current rank resides.

## Function Prototype

```c
HcclResult HcclRankGraphGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t **topoInsts, uint32_t *topoInstNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| topoInsts | Output | Topology instance list. |
| topoInstNum | Output | Number of topology instances. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- On Ascend 950PR&950DT products with layer selection configured, querying a disabled layer, including `netLayer=0`, returns `HCCL_E_PARA`. Retained local groups are not returned as topology instances of that layer.
- The returned memory is managed by the library. The caller must not release it.
- Copy the returned data in a timely manner. Repeated calls in the same communicator may invalidate the previous result.

## Example

```c
HcclComm comm;
uint32_t netLayer = 0;
uint32_t *topoInsts = nullptr;
uint32_t topoInstNum = 0;
// Defaults to 0 when topoInstanceId is not configured in the topo file.
HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInstNum); 
// topoInsts = [0], topoInstNum = 1
```

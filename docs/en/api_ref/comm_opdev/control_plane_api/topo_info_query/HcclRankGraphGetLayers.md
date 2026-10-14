# HcclRankGraphGetLayers

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:55:20.895Z pushedAt=2026-10-12T06:12:35.614Z -->

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

Queries the list of topology layer IDs containing the current rank and the number of topology layers.

![Topology model](figures/topo_model.png)

Taking the preceding topology model as an example, it contains two topology layers: Layer 0 and Layer 1. After this API is called, the returned topology layer ID list is \[0,1\], and the number of topology layers is 2.

## Function Prototype

```c
HcclResult HcclRankGraphGetLayers(HcclComm comm, uint32_t **netLayers, uint32_t *netLayerNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator where the current rank resides.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayers | Output | Topology layer ID list. |
| netLayerNum | Output | Number of topology layer IDs. |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- The returned memory is managed by the library. The caller must not release it.
- Copy the returned data in a timely manner. Repeated calls in the same communicator may invalidate the previous result.
- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

Take the topology model in [Description](#description) as an example:

```c
HcclComm comm;
uint32_t *netLayers;
uint32_t layerNum;
HcclRankGraphGetLayers(comm, &netLayers, &layerNum);
// netLayers=[0,1], layerNum=2
```

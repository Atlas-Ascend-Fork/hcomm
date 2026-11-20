# HcclRankGraphGetLayers

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:42:35.130Z pushedAt=2026-10-08T09:27:51.766Z -->

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

- On Ascend 950PR&950DT products with layer selection configured, this API returns only the allowed layers actually built for the current rank. Disabled layers, including layer 0, are excluded, and layer IDs are not renumbered.
- The returned memory is managed by the library. The caller must not release it.
- Copy the returned data in a timely manner. Repeated calls in the same communicator may invalidate the previous result.

## Example

Take the topology model in [Description](#description) as an example:

```c
HcclComm comm;
uint32_t *netLayers;
uint32_t layerNum;
HcclRankGraphGetLayers(comm, &netLayers, &layerNum);
// netLayers=[0,1], layerNum=2
```

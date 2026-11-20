# HcclRankGraphGetTopoType

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:45:49.799Z pushedAt=2026-09-30T03:06:01.317Z -->

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

Given a communicator and a topology layer ID, queries the topology type corresponding to the current rank in the specified topology instance.

## Function Prototype

```c
HcclResult HcclRankGraphGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo *topoType)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| topoInstId | Input | Topology instance ID. |
| topoType | Output | Topology type.<br>For the definition of the CommTopo type, see [CommTopo](../../datatype_definition/CommTopo.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- On Ascend 950PR&950DT products with layer selection configured, querying a disabled layer, including `netLayer=0`, returns `HCCL_E_PARA`. Retained local grouping information does not make that layer's topology instances or their types available.

## Example

```c
HcclComm comm;
CommTopo topoType;
HcclRankGraphGetTopoType(comm, 0, 0, &topoType); // topoType=1 (1DMesh)
```

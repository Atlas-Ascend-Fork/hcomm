# HcclRankGraphGetTopoTypeByLayer

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:46:34.076Z pushedAt=2026-09-30T03:08:06.199Z -->

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

Given a communicator and a topology layer ID, returns the topology type of the layer where the current rank resides.

![Topology model](figures/topo_model.png)

Take the preceding topology model as an example:

- Layer 0 contains two topology instances. For ease of understanding, the topology instance IDs are defined as 0 and 1. The topology type of instance 0 is 1DMesh, and that of instance 1 is Clos.
- Layer 1 contains one topology instance, whose topology type is Clos.

## Function Prototype

```c
HcclResult HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, CommTopo *topoType)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| netLayer | Input | Topology layer ID. |
| topoType | Output | Topology type, including 1DMesh, Clos, and custom types.<br>For the definition of the CommTopo type, see [CommTopo](../../datatype_definition/CommTopo.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- On Ascend 950PR&950DT products with layer selection configured, querying a disabled layer, including `netLayer=0`, returns `HCCL_E_PARA`. Retained local grouping information does not make the layer's topology type available.

## Example

Take the topology model in [Description](#description) as an example.

For rank 0:

```c
HcclComm comm;
CommTopo topoType;
HcclRankGraphGetTopoTypeByLayer(comm, 0, &topoType);  
// The topoType of Layer0 is 1 (1DMesh).
HcclRankGraphGetTopoTypeByLayer(comm, 1, &topoType);  
// The topoType of Layer1 is 0 (Clos).
```

For rank 3:

```c
HcclComm comm;
CommTopo topoType;
HcclRankGraphGetTopoTypeByLayer(comm, 0, &topoType);  
// The topoType of Layer0 is 0 (Clos).
HcclRankGraphGetTopoTypeByLayer(comm, 1, &topoType);  
// topoType of Layer1 = 0 (Clos).
```

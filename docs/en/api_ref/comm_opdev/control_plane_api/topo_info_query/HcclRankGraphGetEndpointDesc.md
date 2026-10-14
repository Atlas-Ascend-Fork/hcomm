# HcclRankGraphGetEndpointDesc

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:54:16.988Z pushedAt=2026-10-12T06:12:35.611Z -->

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

Obtains the endpoint description list of a topology instance.

## Function Prototype

```c
HcclResult HcclRankGraphGetEndpointDesc(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t *descNum, EndpointDesc *endpointDesc)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| layer | Input | Topology layer number. |
| topoInstId | Input | Topology instance ID. |
| descNum | Input/Output | As an output, it indicates the number of communication device descriptions actually obtained.<br>As an input, it must be equal to the value of **num** output by [HcclRankGraphGetEndpointNum](HcclRankGraphGetEndpointNum.md). |
| endpointDesc | Output | Endpoint description list. The caller needs to allocate memory for it.<br>For the definition of the EndpointDesc type, see [EndpointDesc](../../datatype_definition/EndpointDesc.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
// Communicator handle.
HcclComm comm;

// Obtain the number of endpoints.
uint32_t layer = 0;
uint32_t topoInstId = 0;
uint32_t num = 0;
HcclRankGraphGetEndpointNum(comm, layer, topoInstId, &num);

// Obtain the endpoint description list.
uint32_t descNum = num;
EndpointDesc endpointDesc[descNum];
HcclRankGraphGetEndpointDesc(comm, layer, topoInstId, &descNum, endpointDesc);
```

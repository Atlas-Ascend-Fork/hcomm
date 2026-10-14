# HcclRankGraphGetEndpointInfo

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:54:27.910Z pushedAt=2026-10-12T06:12:35.612Z -->

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

Obtains the topology attribute information of a specified communication device.

## Function Prototype

```c
HcclResult HcclRankGraphGetEndpointInfo(HcclComm comm, uint32_t rankId, const EndpointDesc *endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void *info)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Communicator.<br>The HcclComm type is defined as follows:<br>typedef void *HcclComm; |
| rankId | Input | Rank ID of the endpoint to be queried. |
| endpointDesc | Input | Endpoint descriptor, which is the **endpointDesc** obtained through the [HcclRankGraphGetEndpointDesc](HcclRankGraphGetEndpointDesc.md) API. |
| endpointAttr | Input | Type of the endpoint attribute to be queried.<br>For the definition of the EndpointAttr type, see [EndpointAttr](../../datatype_definition/EndpointAttr.md). |
| infoLen | Input | Size of the provided info buffer (in bytes). |
| info | Output | Pointer to the output buffer that stores the attribute information. |

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

// Obtain the topology attribute information.
uint32_t rankId = 0; // Rank ID of the endpoint to be queried.
EndpointAttrBwCoeff bwCoeff = {0};
uint32_t size = sizeof(EndpointAttrBwCoeff); // Must be equal to the target type size.
HcclRankGraphGetEndpointInfo(comm, rankId, endpointDesc, ENDPOINT_ATTR_BW_COEFF, size, &bwCoeff);
```

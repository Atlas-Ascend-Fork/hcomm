# HcommEndpointCreate

<!-- md-trans-meta sourceCommit=4b9095af56b00ac348521b7520df0ebbf3e3ef78 translatedAt=2026-10-08T11:44:24.389Z pushedAt=2026-10-12T06:12:35.587Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->

## Description

Creates an endpoint for a communication device.

## Function Prototype

```c
HcommResult HcommEndpointCreate(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpoint | Input | Endpoint initialization configuration information.<br>Mandatory members: **protocol**, **commAddr**, and **loc**.<br>For details about the EndpointDesc type, see [EndpointDesc](../../datatype_definition/EndpointDesc.md). |
| endpointHandle | Output | Returned endpoint handle.<br>For details about the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

The supported protocols depend on the endpoint location (**loc**) and the chip model, as detailed below. If the protocol does not match the chip type, the API returns failure.

<!-- npu="950" id6 -->
For Ascend 950PR/Ascend 950DT:

- The supported protocols are RoCE, UB_CTP, UB_MEM, PCIe, UBoE, and UB_RTP.
<!-- end id6 -->

<!-- npu="A3" id7 -->
For Atlas A3 training products/Atlas A3 inference products:

- Only endpoints on the device side are supported.
- The protocols supported on the device side are RoCE and HCCS.

Creating an endpoint on the host side is not supported, and protocols such as UB_CTP, UB_MEM, PCIe, UBoE, and UB_RTP are not supported.
<!-- end id7 -->

<!-- npu="910b" id8 -->
For Atlas A2 training products/Atlas A2 inference products:

- Only endpoints on the device side are supported.
- The protocols supported on the device side are RoCE and HCCS.

Creating an endpoint on the host side is not supported, and protocols such as UB_CTP, UB_MEM, PCIe, UBoe, and UB_RTP are not supported.
<!-- end id8 -->

## Example

```c
struct in_addr ipAddr;
inet_pton(AF_INET, "192.168.1.100", &ipAddr);
const EndpointDesc endpointDesc = {
    .protocol = COMM_PROTOCOL_UB_RTP,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = ipAddr
    },
    .loc = {
        .locType = ENDPOINT_LOC_TYPE_DEVICE,
        .device = {
            .devPhyId = 0,
            .superDevId = 0,
            .serverIdx = 0,
            .superPodIdx = 0
        }
    },
    .raws = {0}
};
EndpointHandle endpointHandle = nullptr;
HcommResult result = HcommEndpointCreate(&endpointDesc, &endpointHandle);
```

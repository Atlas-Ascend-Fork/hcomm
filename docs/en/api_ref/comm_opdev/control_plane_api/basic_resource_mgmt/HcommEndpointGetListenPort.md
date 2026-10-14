# HcommEndpointGetListenPort

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:58:23.585Z pushedAt=2026-09-29T06:22:00.390Z -->

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

Obtains the listening port of the communication device endpoint.

## Function Prototype

```c
HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t *port)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| **endpointHandle** | Input | Endpoint handle.<br>For details about the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). |
| **port** | Output | Returned listening port number. |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

- **endpointHandle** must be a valid handle created by [HcommEndpointCreate](HcommEndpointCreate.md).
- The **port** parameter cannot be a null pointer.

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

uint32_t listenPort = 0;
result = HcommEndpointGetListenPort(endpointHandle, &listenPort);
```

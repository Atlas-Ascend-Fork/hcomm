# HcommMemExport

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:59:18.861Z pushedAt=2026-10-08T08:29:05.630Z -->

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

Exports the description of the specified memory for exchange after memory registration.

## Function Prototype

```c
HcommResult HcommMemExport(EndpointHandle endpointHandle, HcommMemHandle memHandle, void** memDesc, uint32_t* memDescLen)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpointHandle | Input | Endpoint handle.<br>For details about the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). |
| memHandle | Input | Registered memory handle. |
| memDesc | Output | Pointer to the returned description information. |
| memDescLen | Output | Length of the returned description information. |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

- Endpoints of the NIC plugin type do not support this API by default: a "not supported" warning is printed in the log when it is called.

- Whether an endpoint supports this API depends on its location, protocol, and chip model, as described below.

   <!-- npu="950" id6 -->
   - For Ascend 950PR&950DT products:
     - On the host side, endpoints using communication protocol RoCE or UB_CTP are supported.
     - On the device side, endpoints using communication protocol UB_CTP, UBoE, or UB_RTP are supported.
     - Endpoints using communication protocol UB_MEM or PCIe (which can be created only on the device side) do not support this API: no actual operation is performed when it is called, and the "not supported" message is printed in logs.
   <!-- end id6 -->

   <!-- npu="A3" id7 -->
   - For Atlas A3 products: only endpoints on the device side are supported, and endpoints whose communication protocol is RoCE or HCCS are supported.
   <!-- end id7 -->

   <!-- npu="910b" id8 -->
   - For Atlas A2 products: only endpoints on the device side are supported, and endpoints whose communication protocol is RoCE or HCCS are supported.
   <!-- end id8 -->

## Example

```c
const EndpointDesc endpointDesc = {
    .protocol = COMM_PROTOCOL_ROCE,
    .commAddr = {
        .type = COMM_ADDR_TYPE_IP_V4,
        .addr = {{192, 168, 1, 100}}
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
const char *memTag = "HcclBuffer";
CommMem mem = {
    .type = COMM_MEM_TYPE_DEVICE,
    .addr = reinterpret_cast<void*>(0x1111),
    .size = 100
};
HcommMemHandle memHandle;
result = HcommMemReg(endpointHandle, memTag, &mem, &memHandle);

uint32_t memDescLen = 0;
void* memDesc = nullptr;
result = HcommMemExport(endpointHandle, memHandle, &memDesc, &memDescLen);
```

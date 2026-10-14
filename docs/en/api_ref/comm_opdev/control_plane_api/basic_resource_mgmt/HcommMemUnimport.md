# HcommMemUnimport

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:02:50.687Z pushedAt=2026-09-29T06:53:46.096Z -->

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

Disables memory import.

If the same `memDesc` is imported multiple times by the same endpoint (with a reference count greater than 0), each call to `HcommMemUnimport` only decrements the reference count and returns 0. The memory import is actually closed only when the last reference is released.

## Function Prototype

```c
HcommResult HcommMemUnimport(EndpointHandle endpointHandle, const void *memDesc, uint32_t descLen)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| **endpointHandle** | Input | Endpoint handle.<br>For details about the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). |
| `memDesc` | Input | Description information pointer. |
| **descLen** | Input | Description information length. |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

- Endpoints of the NIC plugin type do not support this API by default: a "not supported" warning is printed in the log when the API is called.

- Whether an endpoint supports this API depends on its location, protocol, and chip model, as described below.

   <!-- npu="950" id6 -->
   - For the Ascend 950PR&950DT products:
     - On the host side, endpoints using communication protocol RoCE or UB_CTP are supported.
     - On the device side, endpoints using communication protocol UB_CTP, UBoE, or UB_RTP are supported.
     - Endpoints using communication protocol UB_MEM or PCIe (which can be created only on the DEVICE side) do not support this API: no actual operation is performed when it is called, and a "not supported" message is printed in the log.
   <!-- end id6 -->

   <!-- npu="A3" id7 -->
   - For the Atlas A3 products: only endpoints on the device side are supported, and endpoints using communication protocol RoCE or HCCS are supported.
   <!-- end id7 -->

   <!-- npu="910b" id8 -->
   - For the Atlas A2 products: only endpoints on the device side are supported, and endpoints using communication protocol RoCE or HCCS are supported.
   <!-- end id8 -->

## Example

```c
// Export-side operations.
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

// Import-side operations.
CommMem outMem;
result = HcommMemImport(endpointHandle, memDesc, memDescLen, &outMem);

result = HcommMemUnimport(endpointHandle, memDesc, memDescLen);
```

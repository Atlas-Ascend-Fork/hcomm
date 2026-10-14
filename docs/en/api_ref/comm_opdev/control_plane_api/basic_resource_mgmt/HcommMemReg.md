# HcommMemReg

<!-- md-trans-meta sourceCommit=1435fccbb795a1ecf7fa4675cb37d8bf3e83f779 translatedAt=2026-10-08T11:44:17.427Z pushedAt=2026-10-12T06:12:35.588Z -->

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

Registers memory to a specified endpoint.

## Function Prototype

```c
HcommResult HcommMemReg(EndpointHandle endpointHandle, const char *memTag, const CommMem *mem, HcommMemHandle *memHandle)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpointHandle | Input | Endpoint handle.<br>For details about the EndpointHandle type, see [EndpointHandle](../../datatype_definition/EndpointHandle.md). |
| memTag | Input | Memory string identifier, ending with '\0', with a maximum length of 256 characters (including the '\0' terminator, that is, up to 255 valid characters). If the length is exceeded, the API returns **HCCL_E_PARA**. |
| mem | Input | Memory description, including the physical location type of the memory, memory address, and memory region size in bytes.<br>For details about the CommMem type, see [CommMem](../../datatype_definition/CommMem.md). |
| memHandle | Output | Registered memory handle. |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

- Endpoints of the NIC plugin type do not support this API by default: when the API is called, a "not supported" warning is printed in the log.

- Whether an endpoint supports this API depends on its location, protocol, and chip model, as described below.

   <!-- npu="950" id6 -->
   - For Ascend 950PR&950DT products:
     - When the endpoint is on the host side, the supported communication protocols are RoCE and UB_CTP.
     - When the endpoint is on the device side, the supported communication protocols are UB_CTP, UB_MEM, PCIe, UBoE, and UB_RTP.
   <!-- end id6 -->

   <!-- npu="A3" id7 -->
   - For the Atlas A3 products: only endpoints on the device side are supported, and the supported communication protocols are RoCE and HCCS.
   <!-- end id7 -->

   <!-- npu="910b" id8 -->
   - For the Atlas A2 products: only endpoints on the device side are supported, and the supported communication protocols are RoCE and HCCS.
   <!-- end id8 -->

## Example

```c
struct in_addr ipAddr;
inet_pton(AF_INET, "192.168.1.100", &ipAddr);
const EndpointDesc endpointDesc = {
    .protocol = COMM_PROTOCOL_ROCE,
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
const char *memTag = "HcclBuffer";
CommMem mem = {
    .type = COMM_MEM_TYPE_DEVICE,
    .addr = reinterpret_cast<void*>(0x1111),
    .size = 100
};
HcommMemHandle memHandle;
result = HcommMemReg(endpointHandle, memTag, &mem, &memHandle);
```

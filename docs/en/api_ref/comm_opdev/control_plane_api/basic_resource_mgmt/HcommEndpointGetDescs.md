# HcommEndpointGetDescs

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:58:09.682Z pushedAt=2026-09-29T06:15:55.242Z -->

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

Queries the network-semantic `Endpoint` description on a specified NPU device, including the communication protocol, communication address, and location of the `Endpoint`.

## Function Prototype

```c
HcommResult HcommEndpointGetDescs(
    int32_t deviceLogicId, uint32_t *descNum, EndpointDesc *endpointDescs)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| `deviceLogicId` | Input | Logical ID of the NPU device. |
| `descNum` | Input/Output | On input, this parameter indicates the number of `EndpointDesc` elements that the `endpointDescs` array can actually hold. On success of the API call, it outputs the number of `Endpoint` elements actually returned. On failure of the API call, the parameter value remains the input array capacity. This parameter cannot be a null pointer. |
| `endpointDescs` | Output | This parameter returns the `Endpoint` description array, whose memory is allocated and released by the caller. On success of the API call, the first `descNum` elements are valid. On failure of the API call, the array content must not be used. This parameter cannot be a null pointer.<br>For the definition of the `EndpointDesc` type, see [`EndpointDesc`](../../datatype_definition/EndpointDesc.md). |

> [!CAUTION] Caution
>
> - The input value of `descNum` indicates the actual available capacity of the `endpointDescs` array.
> - It is recommended to allocate the array based on the number returned by [`HcommEndpointGetDescNum`](HcommEndpointGetDescNum.md) and pass that number as-is to this API.
> - If the device configuration changes between two calls, the actual number may still exceed the allocated capacity. In this case, query the number again, allocate the array, and retry.

On success, the key fields returned by the API are as follows.

| `protocol` | `commAddr.type` | Valid Address Field |
| --- | --- | --- |
| `COMM_PROTOCOL_UBC_CTP` | `COMM_ADDR_TYPE_EID` | `commAddr.eid` |
| `COMM_PROTOCOL_UBG` | `COMM_ADDR_TYPE_EID` | `commAddr.eid` |
| `COMM_PROTOCOL_UBOE` | `COMM_ADDR_TYPE_IP_V4` | `commAddr.addr` |

For all returned items, `loc.locType` is `ENDPOINT_LOC_TYPE_DEVICE`, and `loc.device.devPhyId` indicates the physical ID of the NPU device where the `Endpoint` resides.

## Return Value

`HcommResult`: The API returns `0` on success and other values on failure.

## Constraints

- Currently, only the Ascend 950PR&950DT products are supported.
- The query scope includes only network-semantic `Endpoint` objects. The returned communication protocols include `COMM_PROTOCOL_UBC_CTP`, `COMM_PROTOCOL_UBG`, and `COMM_PROTOCOL_UBOE`, and do not return memory-semantic `Endpoint` objects such as `COMM_PROTOCOL_UB_MEM` and `COMM_PROTOCOL_HCCS`.
- `deviceLogicId` must be the logical ID of a valid and in-place NPU device.
- `endpointDescs` must be allocated by the caller in advance. It is recommended to call [`HcommEndpointGetDescNum`](HcommEndpointGetDescNum.md) first to query the array capacity.
- The input value of `descNum` should be the actual available capacity of the `endpointDescs` array.
- When the capacity of the `endpointDescs` array is smaller than the actual number of `Endpoint` objects, the API returns a parameter error and `descNum` remains unchanged from its input value. The caller must call [`HcommEndpointGetDescNum`](HcommEndpointGetDescNum.md) again, reallocate the array, and retry.
- When the NPU device is not configured with an EID, the API returns a not-found error.

## Example

```c
#include <stdio.h>
#include <stdlib.h>

#include "hcomm_res.h"

int main(void)
{
    /* Step 1: Query the number of endpoints. */
    const int32_t deviceLogicId = 0;
    uint32_t descNum = 0;
    HcommResult result = HcommEndpointGetDescNum(deviceLogicId, &descNum);
    if (result != 0) {
        (void)fprintf(stderr, "HcommEndpointGetDescNum failed, result = %d\n", (int)result);
        return 1;
    }
    if (descNum == 0) {
        (void)printf("No endpoint description found.\n");
        return 0;
    }

    /* Step 2: Allocate the description array based on the queried number. */
    EndpointDesc *endpointDescs = (EndpointDesc *)calloc(descNum, sizeof(*endpointDescs));
    if (endpointDescs == NULL) {
        (void)fprintf(stderr, "Allocate endpoint description array failed.\n");
        return 1;
    }

    /* Step 3: Input the array capacity. On success, descNum is updated to the actual number returned. */
    result = HcommEndpointGetDescs(deviceLogicId, &descNum, endpointDescs);
    if (result != 0) {
        (void)fprintf(stderr, "HcommEndpointGetDescs failed, result = %d\n", (int)result);
        free(endpointDescs);
        return 1;
    }

    /* Step 4: Use the returned descriptions. */
    for (uint32_t i = 0; i < descNum; ++i) {
        const EndpointDesc *desc = &endpointDescs[i];
        (void)printf("endpoint[%u]: protocol=%d, addrType=%d, devPhyId=%u\n",
            i, (int)desc->protocol, (int)desc->commAddr.type, desc->loc.device.devPhyId);
    }

    free(endpointDescs);
    return 0;
}
```

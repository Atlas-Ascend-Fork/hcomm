# HcommEndpointGetDescNum

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:57:29.160Z pushedAt=2026-09-29T06:07:48.201Z -->

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

Queries the number of network-semantic `Endpoint` objects on a specified NPU device. The caller can allocate an `Endpoint` description array based on the returned number, and then call [`HcommEndpointGetDescs`](HcommEndpointGetDescs.md) to obtain the `Endpoint` descriptions.

## Function Prototype

```c
HcommResult HcommEndpointGetDescNum(int32_t deviceLogicId, uint32_t *descNum)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| `deviceLogicId` | Input | Logical ID of the NPU device. |
| `descNum` | Output | When the API is called successfully, the number of network-semantic `Endpoint` objects is returned. When this pointer is non-null but the API call fails, the value is `0`. This parameter cannot be a null pointer. |

## Return Value

`HcommResult`: The API returns `0` on success and other values on failure.

## Constraints

- Currently, this API is supported only on Ascend 950PR&950DT products.
- The query scope includes only network-semantic `Endpoint` objects. Supported communication protocols include `COMM_PROTOCOL_UBC_CTP`, `COMM_PROTOCOL_UBG`, and `COMM_PROTOCOL_UBOE`. Memory-semantic `Endpoint` objects such as `COMM_PROTOCOL_UB_MEM` and `COMM_PROTOCOL_HCCS` are not returned.
- `deviceLogicId` must be the logical ID of a valid and present NPU device.
- If the NPU device has no EID configured, the API returns a not-found error, and `descNum` is `0`.

## Example

```c
#include <stdio.h>

#include "hcomm_res.h"

int main(void)
{
    /* Input: logical ID of the NPU device to query. */
    const int32_t deviceLogicId = 0;

    /* Output: number of network-semantic endpoints. */
    uint32_t descNum = 0;
    HcommResult result = HcommEndpointGetDescNum(deviceLogicId, &descNum);
    if (result != 0) {
        (void)fprintf(stderr, "HcommEndpointGetDescNum failed, result = %d\n", (int)result);
        return 1;
    }

    (void)printf("Endpoint description count: %u\n", descNum);
    return 0;
}
```

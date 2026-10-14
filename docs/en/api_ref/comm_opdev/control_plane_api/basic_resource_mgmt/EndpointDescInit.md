# EndpointDescInit

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:51:00.890Z pushedAt=2026-09-29T03:29:07.992Z -->

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

Initializes the endpoint description list. This API sets the fields in the [EndpointDesc](../../datatype_definition/EndpointDesc.md) structure to reserved values, ensuring that the structure is in an explicitly invalid initial state.

## Function Prototype

```c
HcommResult EndpointDescInit(EndpointDesc *endpoint, uint32_t num)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| endpoint | Output | Endpoint description list. The list length is **num**. This API initializes the structure.<br>For the definition of the EndpointDesc type, see [EndpointDesc](../../datatype_definition/EndpointDesc.md). |
| num | Input | Number of endpoint descriptions. |

## Return Value

[HcommResult](../../datatype_definition/HcommResult.md): The API returns **0** on success and other values on failure.

## Constraints

None

## Example

Take initializing an endpoint description list with two entries as an example:

```c
uint32_t descNum = 2;
std::vector<EndpointDesc> endpointDesc(descNum);
HcommResult ret = EndpointDescInit(endpointDesc.data(), descNum);
if (ret != 0) {
    // Handle the error.
}
```

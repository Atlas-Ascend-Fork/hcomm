# HcommEndpointCheckFeature

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:55:47.105Z pushedAt=2026-09-29T05:59:34.160Z -->

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

Queries whether a specified network endpoint supports a certain feature, without actually creating the endpoint.

By passing in the endpoint description and the feature type, it queries whether the underlying hardware/network driver supports the feature, and returns the result through the value parameter. This API is mainly used for capability detection before creating an endpoint, so that the upper layer can select different communication strategies based on hardware capabilities.

Currently, only the NPU Direct RDMA Async (NDA) feature can be queried.

## Function Prototype

```c
HcommResult HcommEndpointCheckFeature(HcommEndpointFeatureType featureType, const EndpointDesc *endpointDesc, bool *value);
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| featureType | Input | Feature type to be queried.<br>For the definition of the HcommEndpointFeatureType type, see [HcommEndpointFeatureType](../../datatype_definition/HcommEndpointFeatureType.md). |
| endpointDesc | Input | Endpoint description, used to specify the endpoint attributes (such as protocol, address, and location) to be queried. The feature can be queried without actually creating the endpoint.<br>For the definition of the EndpointDesc type, see [EndpointDesc](../../datatype_definition/EndpointDesc.md).<br>This parameter cannot be a null pointer. |
| value | Output | Result of feature support.<br>**true** indicates that the feature is supported, and **false** indicates that it is not supported.<br>This parameter cannot be a null pointer. |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

- You do not need to create an endpoint before calling this API. You only need to fill in the **EndpointDesc** description information.
- When querying the **HCOMM_ENDPOINT_FEATURE_NDA** feature, the protocol in **EndpointDesc** must be **COMM_PROTOCOL_ROCE** and the endpoint location type must be **ENDPOINT_LOC_TYPE_HOST**. Otherwise, **false** is returned.
- This API is mainly used in the capability detection phase and should not be called frequently during communication.

## Example

```c
// Fill in the endpoint description.
EndpointDesc endpointDesc;
// Fill in EndpointDesc by referring to HcommEndpointCreate.
...

// Query whether the NDA feature is supported.
bool isNdaSupported = false;
HcommResult ret = HcommEndpointCheckFeature(HCOMM_ENDPOINT_FEATURE_NDA, &endpointDesc, &isNdaSupported);
if (ret != 0) {
    printf("Failed to check feature, ret = %d\n", ret);
    return ret;
}

if (isNdaSupported) {
    // Communicate using the NDA feature.
    // ...
} else {
    // Fall back to other communication methods.
    // ...
}
```

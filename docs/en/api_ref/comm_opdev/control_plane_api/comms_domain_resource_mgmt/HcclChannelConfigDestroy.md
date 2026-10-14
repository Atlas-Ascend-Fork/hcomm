# HcclChannelConfigDestroy

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:21:53.821Z pushedAt=2026-09-29T11:24:38.240Z -->

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

Destroys the channel configuration object created by [HcclChannelConfigCreate](HcclChannelConfigCreate.md) and releases the memory resources it occupies.

## Function Prototype

```c
HcclResult HcclChannelConfigDestroy(HcclChannelConfig config)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Input | Handle of the channel configuration object to be destroyed.<br>For the definition of the HcclChannelConfig type, see [HcclChannelConfig](../../datatype_definition/HcclChannelConfig.md).|

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

- After destruction, the handle must not be used again; otherwise, undefined behavior occurs.
- The configuration object can be destroyed immediately after [HcclChannelAcquireWithConfig](HcclChannelAcquireWithConfig.md) is called, without affecting the created channels.

## Example

```c
HcclChannelConfig config = nullptr;
HcclChannelConfigCreate(&config);
// ... Use config to create a channel ...
HcclChannelConfigDestroy(config);
config = nullptr;
```

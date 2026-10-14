# HcommChannelConfigDestroy

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:51:39.710Z pushedAt=2026-09-29T03:28:43.424Z -->

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

Destroys the channel configuration object created through [HcommChannelConfigCreate](HcommChannelConfigCreate.md) and releases the memory resources it occupies.

## Function Prototype

```c
HcommResult HcommChannelConfigDestroy(HcommChannelConfig config)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Input | Handle of the channel configuration object to be destroyed.<br>For the definition of the HcommChannelConfig type, see [HcommChannelConfig](../../datatype_definition/HcommChannelConfig.md). |

## Return Value

**HcommResult**: The API returns **0** on success and other values on failure.

## Constraints

- After destruction, the handle must not be used again. Otherwise, undefined behavior occurs.
- The configuration object can be destroyed immediately after the [HcommChannelCreateWithConfig](HcommChannelCreateWithConfig.md) call completes, without affecting the created channel.

## Example

```c
HcommChannelConfig config = nullptr;
HcommChannelConfigCreate(&config);
// Use config to create a channel.
HcommChannelConfigDestroy(config);
config = nullptr;
```

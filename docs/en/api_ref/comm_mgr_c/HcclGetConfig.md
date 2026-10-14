# HcclGetConfig

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:22:06.139Z pushedAt=2026-09-29T01:12:43.620Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

Obtains the configuration related to collective communication.

## Function Prototype

```c
HcclResult HcclGetConfig(HcclConfig config, HcclConfigValue *configValue)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Input | Collective communication configuration parameter.<br>Type: [HcclConfig](./data_type_definition/HcclConfig.md). In the current version, only **HCCL_DETERMINISTIC** is supported. |
| configValue | Output | Value of the parameter configured in **config**.<br>For details, see [HcclConfigValue](./data_type_definition/HcclConfigValue.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
// Query the deterministic computation switch.
HcclConfig config = HCCL_DETERMINISTIC;
union HcclConfigValue configValue;
HcclGetConfig(HCCL_DETERMINISTIC, &configValue);
```

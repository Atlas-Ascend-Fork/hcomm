# HcclGetCommConfigCapability

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:20:55.487Z pushedAt=2026-09-29T01:09:12.733Z -->

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
<!-- npu="310p" id4 -->
- Atlas inference products: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

Determines whether the current software version supports a specific communicator initialization configuration.

For the complete set of configurations supported during communicator initialization, see [HcclCommConfigCapability](./data_type_definition/HcclCommConfigCapability.md). The configurations include the shared data buffer size, deterministic computation switch, communicator name, and communication operator expansion mode.

The process of using the **HcclGetCommConfigCapability** API to determine whether the current software supports a specific configuration is as follows:

1. Call the **HcclGetCommConfigCapability** API to obtain a value that represents the communicator initialization configuration capability of the current software.
2. Compare the value with a configuration enum value in [HcclCommConfigCapability](./data_type_definition/HcclCommConfigCapability.md). If the value is greater than the enum value, the current software supports the configuration capability corresponding to that enum value in [HcclCommConfigCapability](./data_type_definition/HcclCommConfigCapability.md); if the value is less than or equal to the enum value, the capability is not supported.

   For example, to determine whether the current software supports configuring the communicator name, compare the return value of **HcclGetCommConfigCapability** with the enum value **HCCL_COMM_CONFIG_COMM_NAME**. If the return value is greater than **HCCL_COMM_CONFIG_COMM_NAME**, the current software supports configuring the communicator name; if the return value is less than or equal to **HCCL_COMM_CONFIG_COMM_NAME**, the current software does not support configuring the communicator name.

## Function Prototype

```c
uint32_t HcclGetCommConfigCapability()
```

## Parameters

None

## Return Value

**uint32_t**: value indicating the communicator initialization configuration capability.

For details about the meaning of this value, see [Description](#description).

## Constraints

None

## Example

```c
uint32_t configCapability = HcclGetCommConfigCapability();
bool isSupportCommName = configCapability > HCCL_COMM_CONFIG_COMM_NAME;  // Determine whether configuring the communicator name is supported.
```

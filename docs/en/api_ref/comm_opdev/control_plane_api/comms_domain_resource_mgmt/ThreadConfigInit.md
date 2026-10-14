# ThreadConfigInit

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T07:38:48.656Z pushedAt=2026-09-30T02:36:38.654Z -->

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

Initializes the thread configuration structure.

## Function Prototype

```c
HcommResult ThreadConfigInit(ThreadConfig *config, uint32_t num)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Output | Thread configuration structure array. The array length is **num**. The function initializes this structure.<br>For details about the ThreadConfig type, see [ThreadConfig](../../datatype_definition/ThreadConfig.md). |
| num | Input | Number of thread configurations. |

## Return Value

**HcommResult**: The return values of the API are described as follows:

| Return Value | Description |
| --- | --- |
| 0 | The thread configuration structure is initialized successfully. |
| 2 | The **config** passed in is a null pointer, corresponding to **HCCL_E_PTR**. |
| 4 | Internal error, corresponding to **HCCL_E_INTERNAL**. |

## Constraints

- The **ThreadConfig** structure must be initialized by calling this API. Otherwise, the [HcclThreadAcquireWithConfig](HcclThreadAcquireWithConfig.md) API returns a parameter error.

- After initialization, set the **notifyNumPerThread** field based on service requirements to specify the number of synchronization resources for each communication thread.

## Example

The following uses the initialization of five thread configuration structures as an example:

```c
uint32_t threadNum = 5;
ThreadConfig configs[5];
ThreadConfigInit(configs, threadNum);
for (uint32_t i = 0; i < threadNum; i++) {
    configs[i].notifyNumPerThread = 2;
}
```

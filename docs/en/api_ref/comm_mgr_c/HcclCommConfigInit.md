# HcclCommConfigInit

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:07:40.522Z pushedAt=2026-09-28T08:51:44.470Z -->

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

Initializes the communicator configuration.

## Function Prototype

```c
static inline void HcclCommConfigInit(HcclCommConfig *config)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| config | Output | Communicator configuration to be initialized.<br>For the definition of the HcclCommConfig type, see [HcclCommConfig](./data_type_definition/HcclCommConfig.md). |

## Return Value

None

## Constraints

None

## Example

```c
uint32_t rankSize = 8;
uint32_t deviceId = 0;
// Generate the rank identifier information of the root node.
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);

// Create and initialize the communicator configuration.
HcclCommConfig config;
HcclCommConfigInit(&config);
// Modify the communicator configuration as needed.
config.hcclBufferSize = 1024;  // Buffer size for shared data, in MB. The value must be greater than or equal to 1, and the default value is 200.
config.hcclDeterministic = 1;  // Enable deterministic computation for reduction-type communication operators. Default value: 0, which disables deterministic computation.
strncpy(config.hcclCommName, "comm_1", COMM_NAME_MAX_LENGTH - 1);
config.hcclCommName[COMM_NAME_MAX_LENGTH - 1] = '\0';
// Initialize the communicator.
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, deviceId, &config, &hcclComm));

// Destroy the communicator.
HcclCommDestroy(hcclComm);
```

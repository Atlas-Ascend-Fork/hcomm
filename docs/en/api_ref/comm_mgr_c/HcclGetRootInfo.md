# HcclGetRootInfo

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:23:58.038Z pushedAt=2026-09-29T01:24:32.537Z -->

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

Generates the rank identifier information (**HcclRootInfo**) of the root node. This API must be called before the HCCL initialization API [HcclCommInitRootInfo](HcclCommInitRootInfo.md) or [HcclCommInitRootInfoConfig](HcclCommInitRootInfoConfig.md), and only needs to be called on the root node.

- This API must be used together with the initialization API [HcclCommInitRootInfo](HcclCommInitRootInfo.md) or [HcclCommInitRootInfoConfig](HcclCommInitRootInfoConfig.md), and cannot be used alone.

- This API supports loop calls in a single thread. That is, developers can specify different devices and call this API in a for loop to obtain **rootInfo** of different devices in a single thread.

    Assume that an AI server has eight devices, which are divided into four communicators. The two devices in each communicator communicate with each other, as shown in the following figure.

    ![Communicator division example](figures/comm_domain_split.png)

    The process of obtaining **rootInfo** and performing collective communication initialization is shown in the following figure.

    ![Single-thread loop call example](figures/single_thread_loop_call_example.png)

    First, create four **rootInfo** entries in a single thread by switching devices, and store them in an array of length 4. After the **rootInfo** entries are obtained, start four threads and call **HcclCommInitRootInfo** or **HcclCommInitRootInfoConfig** (illustrated with **HcclCommInitRootInfo** in the preceding figure) to initialize the communicator based on different **rootInfo** entries.

- In a multi-server collective communication scenario, before calling **HcclGetRootInfo**, you can perform the following operations (optional):
  - Configure the environment variable [HCCL_IF_IP](https://gitcode.com/cann/hccl/blob/9.3.0-beta.1/docs/en/user_guide/hccl_env/HCCL_IF_IP.md) or [HCCL_SOCKET_IFNAME](https://gitcode.com/cann/hccl/blob/9.3.0-beta.1/docs/en/user_guide/hccl_env/HCCL_SOCKET_IFNAME.md) to specify the root NIC IP for HCCL initialization. (The environment variable **HCCL_IF_IP** takes precedence over **HCCL_SOCKET_IFNAME**. If neither is configured, the root NIC is selected in ascending lexicographic order of NIC names by default.)
  - Configure the environment variable [HCCL_WHITELIST_DISABLE](https://gitcode.com/cann/hccl/blob/9.3.0-beta.1/docs/en/user_guide/hccl_env/HCCL_WHITELIST_DISABLE.md) to enable trustlist verification, and use [HCCL_WHITELIST_FILE](https://gitcode.com/cann/hccl/blob/9.3.0-beta.1/docs/en/user_guide/hccl_env/HCCL_WHITELIST_FILE.md) to specify the communication trustlist configuration file. (If not configured, communication trustlist verification is disabled by default.)

## Function Prototype

```c
HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| rootInfo | Output | Identifier information of the current rank, mainly including the device IP and device ID. This information must be broadcast to all ranks in the cluster for HCCL initialization.<br>For the definition of the HcclRootInfo type, see [HcclRootInfo](./data_type_definition/HcclRootInfo.md). |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

None

## Example

```c
uint32_t rankSize = 8;
uint32_t deviceId = 0;
// Generate the rank identifier information of the root node.
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);
// Initialize the communicator.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);
// Destroy the communicator.
HcclCommDestroy(hcclComm);
```

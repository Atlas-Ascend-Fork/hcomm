# HcclTeamCreate

<!-- md-trans-meta sourceCommit=2a8d3e3f08bbf6912c727dc20117288527683955 translatedAt=2026-10-08T11:51:00.876Z pushedAt=2026-10-12T06:12:35.603Z -->

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

Creates a team for HCCL communication. Based on the `protocol` and `netLayer` specified in `desc`, it locates the pre-created **worldTeam** during communicator initialization, creates a team on top of it, and internally completes operations such as **syncMem** registration, channel link establishment, remote memory acquisition, and binding. After successful creation, it returns the team handle.

## Function Prototype

```c
HcclResult HcclTeamCreate(HcclComm comm, const HcclTeamCreateDesc* desc, HcommTeamHandle* team)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Initialized communicator handle, which cannot be **NULL**. For the definition of the **HcclComm** type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| desc | Input | Team creation descriptor, which must be initialized and populated with service fields through [HcclTeamCreateDescInit](HcclTeamCreateDescInit.md). For the definition of the HcclTeamCreateDesc type, see [HcclTeamCreateDesc](../../datatype_definition/HcclTeamCreateDesc.md). |
| team | Output | Created team handle, which cannot be **NULL**. For the definition of the HcommTeamHandle type, see [HcommTeamHandle](../../datatype_definition/HcommTeamHandle.md). |

## Return Value

[HcclResult](../../../comm_mgr_c/data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

1. The **worldTeam** is automatically created during communicator initialization, and users do not need to create it manually. **HcclTeamCreate** locates the corresponding pre-created **worldTeam** based on the `protocol` and `netLayer` in `desc`.

2. `desc` must be initialized through [HcclTeamCreateDescInit](HcclTeamCreateDescInit.md). `protocol` cannot be **COMM_PROTOCOL_RESERVED**, and currently only the URMA protocol is supported (corresponding to the enumeration values **COMM_PROTOCOL_UB_CTP**, **UBC_TP**, and **UB_RTP**).

3. If link establishment fails, **HcclTeamCreate** automatically rolls back the created team and its resources, and `team` is output as **NULL**.
4. This API does not support concurrent calls by multiple threads. The caller must ensure that calls to this API from different threads are executed serially.
5. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
uint32_t rankSize = 2;
uint32_t deviceId = 0;
// Generate the rank identifier information of the root node.
HcclRootInfo rootInfo;
HcclResult ret = HcclGetRootInfo(&rootInfo);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// Initialize the communicator.
HcclComm comm;
ret = HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &comm);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// Allocate device memory.
void* devPtr = nullptr;
size_t memSize = 1024;
aclError aclRet = aclrtMalloc(&devPtr, memSize, ACL_MEM_MALLOC_HUGE_FIRST);
if (aclRet != ACL_SUCCESS) {
    return HCCL_E_RUNTIME;
}
HcclCommSymWindow symWin;
// Register the symmetric memory window.
ret = HcclCommSymWinRegister(comm, devPtr, memSize, &symWin, 1);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// Initialize the descriptor.
HcclTeamCreateDesc desc;
ret = HcclTeamCreateDescInit(&desc);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// Fill in the service fields.
uint32_t rankIds[2] = {0, 1};
desc.rankIds = rankIds;
desc.rankNum = 2;
desc.selfRankId = 0;
desc.netLayer = 0;
desc.protocol = COMM_PROTOCOL_UB_CTP;
desc.requirement.barrierCount = 1;
desc.engine = COMM_ENGINE_AIV;
desc.notifyNum = 8;
desc.channelCnt = 1;
// Create the team.
HcommTeamHandle team = nullptr;
ret = HcclTeamCreate(comm, &desc, &team);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// Destroy the team.
ret = HcclTeamDestroy(team);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// Deregister the symmetric memory window.
ret = HcclCommSymWinDeregister(symWin);
if (ret != HCCL_SUCCESS) {
    return ret;
}
// Free the memory.
aclRet = aclrtFree(devPtr);
if (aclRet != ACL_SUCCESS) {
    return HCCL_E_RUNTIME;
}
// Destroy the communicator.
ret = HcclCommDestroy(comm);
if (ret != HCCL_SUCCESS) {
    return ret;
}
```

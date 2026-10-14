# HcclTeamGetLsaTeam

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:51:27.415Z pushedAt=2026-10-12T06:12:35.604Z -->

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

Obtains the UB Memory LSA WorldTeam handle pre-created during communicator initialization. This handle can be used to convert between LSA team member numbers and communicator rank IDs, and to use UB Memory symmetric memory together with [HcclSymWinGetPeerPointer](../../../comm_mgr_c/HcclSymWinGetPeerPointer.md).

## Function Prototype

```c
HcclResult HcclTeamGetLsaTeam(HcclComm comm, HcommTeamHandle *lsaTeam)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Initialized communicator handle, which cannot be **NULL**. For the definition of the HcclComm type, see [HcclComm](../../../comm_mgr_c/data_type_definition/HcclComm.md). |
| lsaTeam | Output | Pointer to the LSA WorldTeam handle, which cannot be **NULL**. The returned handle is managed by **comm**. For the definition of the HcommTeamHandle type, see [HcommTeamHandle](../../datatype_definition/HcommTeamHandle.md). |

## Return Value

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The LSA WorldTeam is obtained successfully. |
| HCCL_E_PTR | comm or lsaTeam is NULL. |
| HCCL_E_NOT_SUPPORT | comm is not a CommunicatorV2 communicator. |
| HCCL_E_NOT_FOUND | No LSA WorldTeam is pre-created in the communicator. |
| HCCL_E_INTERNAL | The LSA WorldTeam metadata is abnormal. |

## Constraints

1. Only CommunicatorV2 communicators are supported.
2. The communicator must successfully pre-create the UB Memory LSA WorldTeam during initialization; otherwise, **HCCL_E_NOT_FOUND** is returned.
3. The returned LSA WorldTeam is owned by the communicator, and the caller must not destroy it separately through **HcclTeamDestroy**. After the communicator is destroyed, the handle becomes invalid.
4. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
HcommTeamHandle lsaTeam = NULL;
HcclResult ret = HcclTeamGetLsaTeam(comm, &lsaTeam);
if (ret != HCCL_SUCCESS) {
    return ret;
}
```

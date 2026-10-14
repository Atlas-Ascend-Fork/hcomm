# HcclTeamMemberToRank

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:51:42.339Z pushedAt=2026-10-12T06:12:35.604Z -->

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

Converts the member ID **memberId** in a specified team to the corresponding communicator rank ID. For the UB Memory LSA WorldTeam, **memberId** can be used as the LSA member ID of [HcclSymWinGetPeerPointer](../../../comm_mgr_c/HcclSymWinGetPeerPointer.md).

## Function Prototype

```c
HcclResult HcclTeamMemberToRank(
    HcclComm comm, HcommTeamHandle team, uint32_t memberId, uint32_t *rankId)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Initialized communicator handle, which cannot be **NULL**. |
| team | Input | Team handle to be queried, which cannot be **NULL** and must belong to **comm**. |
| memberId | Input | Member ID within the team, with a value range of [0, number of team members). |
| rankId | Output | Pointer to the communicator rank ID, which cannot be **NULL**. |

## Return Value

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | Conversion succeeded. |
| HCCL_E_PTR | **comm**, **team**, or **rankId** is NULL. |
| HCCL_E_NOT_SUPPORT | **comm** is not a CommunicatorV2 communicator. |
| HCCL_E_NOT_FOUND | **team** is not registered in the team manager corresponding to **comm**. |
| HCCL_E_PARA | **team** does not belong to **comm**, or **memberId** is out of the team member range. |

## Constraints

1. Only CommunicatorV2 communicators are supported.
2. **team** must belong to **comm** and remain valid during the call.
3. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
uint32_t rankId = 0;
HcclResult ret = HcclTeamMemberToRank(comm, lsaTeam, memberId, &rankId);
if (ret != HCCL_SUCCESS) {
    return ret;
}
```

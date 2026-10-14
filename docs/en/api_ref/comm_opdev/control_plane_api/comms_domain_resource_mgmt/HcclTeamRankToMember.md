# HcclTeamRankToMember

<!-- md-trans-meta sourceCommit=306b11da159831ad0e3dbb73fec73f1ad849a0a2 translatedAt=2026-10-08T11:52:12.203Z pushedAt=2026-10-12T06:12:35.605Z -->

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

Converts the communicator Rank ID to the member number **memberId** in the specified Team. For the UB Memory LSA WorldTeam, the converted **memberId** can be used for data-plane LSA access.

## Function Prototype

```c
HcclResult HcclTeamRankToMember(
    HcclComm comm, HcommTeamHandle team, uint32_t rankId, uint32_t *memberId)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | Initialized communicator handle, which cannot be NULL. |
| team | Input | Team handle to be queried, which cannot be NULL and must belong to **comm**. |
| rankId | Input | Communicator Rank ID. |
| memberId | Output | Pointer to the Team member number, which cannot be NULL. |

## Return Value

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | Conversion succeeded. |
| HCCL_E_PTR | **comm**, **team**, or **memberId** is NULL. |
| HCCL_E_NOT_SUPPORT | **comm** is not a CommunicatorV2 communicator. |
| HCCL_E_NOT_FOUND | The team is not registered, or **rankId** does not belong to the team. |
| HCCL_E_PARA | The team does not belong to **comm**. |

## Constraints

1. Only CommunicatorV2 communicators are supported.
2. **team** must belong to **comm** and remain valid during the call.
3. **comm** must be a valid communicator handle (obtained through the communicator creation API and kept valid during the call). Do not pass a null pointer, wild pointer, or destroyed handle.

## Example

```c
uint32_t memberId = 0;
HcclResult ret = HcclTeamRankToMember(comm, lsaTeam, rankId, &memberId);
if (ret != HCCL_SUCCESS) {
    return ret;
}
```

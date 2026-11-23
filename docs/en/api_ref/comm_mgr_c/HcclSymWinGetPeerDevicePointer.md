# HcclSymWinGetPeerDevicePointer

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT series products: Supported in the UB Memory scenario, not supported in the URMA scenario
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 series products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 series products: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Obtains the peer device address pointer corresponding to a symmetric memory window on the host side based on the symmetric memory window resource handle, offset, and peer identifier.

<!-- npu="950" id8 -->
For the UB Memory scenario of Ascend 950PR&950DT series products, the peer identifier is the member ID of the LSA WorldTeam, and the API returns the device address of the LSA member in the symmetric VA space. This API is called on the host side. The URMA scenario does not support this API. Use [HcclSymWinGetRemoteAddr](HcclSymWinGetRemoteAddr.md) to obtain the remote address instead.
<!-- end id8 -->

<!-- npu="A3" id6 -->
For Atlas A3 series products, the symmetric memory window is not bound to the UB Memory layout. This API returns HCCL_E_NOT_SUPPORT and sets *ptr to nullptr. In this case, use [HcclSymWinGetPeerPointer](HcclSymWinGetPeerPointer.md) to obtain the address instead.
<!-- end id6 -->

## Function Prototype

```c
HcclResult HcclSymWinGetPeerDevicePointer(HcclCommSymWindow winHandle, size_t offset, uint32_t peerMemberId, void** ptr)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| winHandle | Input | Symmetric memory window resource handle.<br>For the definition of the HcclCommSymWindow type, see [HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md). |
| offset | Input | Offset obtained by calling [HcclCommSymWinGet](HcclCommSymWinGet.md). The value range is [0, window user size). |
| peerMemberId | Input | Peer identifier, which is the member ID of the LSA WorldTeam. The value range is [0, lsaTeamSize).<br>This parameter is not the rank ID in the communicator. It can be obtained by calling [HcclTeamRankToMember](../comm_opdev/control_plane_api/comms_domain_resource_mgmt/HcclTeamRankToMember.md). |
| ptr | Output | Pointer to the "address corresponding to the symmetric memory window". The API sets *ptr to nullptr on failure. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md):

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The peer address is obtained successfully. |
| HCCL_E_PTR | winHandle or ptr is a null pointer. |
| HCCL_E_NOT_SUPPORT | The window is not a UB Memory symmetric memory window (for example, in the URMA scenario). The caller needs to fall back. |
| HCCL_E_NOT_FOUND | winHandle is not registered or has been deregistered. |
| HCCL_E_PARA | peerMemberId is out of range, or offset exceeds the window range. |

## Constraints

<!-- npu="950" id9 -->
- For Ascend 950PR&950DT series products, only the UB Memory scenario is supported, and the input peerMemberId must be the member ID of the LSA WorldTeam.
- In the UB Memory scenario, the member ID is validated at runtime. If it is outside `[0, lsaTeamSize)`, `HCCL_E_PARA` is returned.
- In the UB Memory scenario, offset is validated at runtime. If it is outside `[0, window user size)`, `HCCL_E_PARA` is returned.
<!-- end id9 -->
- This API is called on the host side and returns the peer device address.
- The symbol of this API is provided by `libhcomm.so`. In Release builds, it is exported only through the host-side version script `src/legacy/ascend910/framework/hcomm.map`; the script is not applied in non-Release builds. The device-side libraries `libccl_kernel.so` and `libaicpu_custom.so` do not export this symbol. The caller should link against `libhcomm.so`, or load the API symbol from `libhcomm.so` through `dlsym`. Do not look up this symbol in device-side libraries.
- On failure, *ptr is always set to nullptr. The caller must not reuse the address obtained from a previous query.
- After the window corresponding to winHandle is deregistered ([HcclCommSymWinDeregister](HcclCommSymWinDeregister.md)), the address obtained by this API becomes invalid.

## Example

After the symmetric memory window is registered, obtain the peer device addresses one by one on the host side by LSA WorldTeam member ID. The following is the pseudocode description:

```c
// comm, addr, size, and peerRank are prepared by the caller: comm is the created communicator, addr and size are the local address and size registered as symmetric memory, and peerRank is the peer rank ID.
// Obtain the symmetric memory window registered by the current rank and its offset.
HcclCommSymWindow temp_win = nullptr;
size_t offset = 0;
HcclCommSymWinGet(comm, addr, size, &temp_win, &offset);

// Convert the rank ID in the communicator to the LSA WorldTeam member ID.
HcommTeamHandle lsaTeam = nullptr;
HcclTeamGetLsaTeam(comm, &lsaTeam);
uint32_t peerMemberId = 0;
HcclTeamRankToMember(comm, lsaTeam, peerRank, &peerMemberId);

// Obtain the peer address. peerPtr is nullptr on failure.
void *peerPtr = nullptr;
HcclResult ret = HcclSymWinGetPeerDevicePointer(temp_win, offset, peerMemberId, &peerPtr);
if (ret == HCCL_E_NOT_SUPPORT) {
    // The window is not a UB Memory symmetric window. Fall back to another address query method.
}

// The peerPtr obtained on the host side is the peer device address.
```

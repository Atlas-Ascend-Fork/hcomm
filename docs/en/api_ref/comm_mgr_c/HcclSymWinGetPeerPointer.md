# HcclSymWinGetPeerPointer

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:28:19.572Z pushedAt=2026-09-29T01:57:18.358Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT series products: Supported in the UB Memory scenario, not supported in the URMA scenario
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 series products: Supported
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

Obtains the address pointer corresponding to a symmetric memory window based on the symmetric memory window resource handle, offset, and peer identifier.

<!-- npu="950" id8 -->
For the UB Memory scenario of Ascend 950PR&950DT series products, the peer identifier is the member ID of the LSA WorldTeam, and the API returns the address pointer of the LSA member in the symmetric VA space. The URMA scenario does not support this API. Use [HcclSymWinGetRemoteAddr](HcclSymWinGetRemoteAddr.md) to obtain the remote address instead.
<!-- end id8 -->

<!-- npu="A3" id6 -->
For Atlas A3 series products, this API supports the HCCS link communication scenario and returns the corresponding valid address pointer. If the window is in URMA mode, this API returns an error and sets *ptr to nullptr. In this case, use [HcclSymWinGetRemoteAddr](HcclSymWinGetRemoteAddr.md) to obtain the address instead.
<!-- end id6 -->

## Function Prototype

```c
HcclResult HcclSymWinGetPeerPointer(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| winHandle | Input | Symmetric memory window resource handle.<br>For the definition of the HcclCommSymWindow type, see [HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md). |
| offset | Input | Offset obtained by calling [HcclCommSymWinGet](HcclCommSymWinGet.md). |
| peerRank | Input | Peer identifier. In the A3 scenario, it is the rank ID in the communicator, with a value range of [0, rankSize). In the UB Memory scenario of Ascend 950PR&950DT series products, it is the member ID of the LSA WorldTeam, with a value range of [0, lsaTeamSize). |
| ptr | Output | Pointer to the "address corresponding to the symmetric memory window". |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

<!-- npu="A3" id7 -->
- For Atlas A3 series products, only the HCCS link communication scenario is supported.
<!-- end id7 -->
<!-- npu="950" id9 -->
- For Ascend 950PR&950DT series products, only the UB Memory scenario is supported, and the input peerRank must be the member ID of the LSA WorldTeam.
- In the UB Memory scenario, the member ID is validated at runtime. If it is outside `[0, lsaTeamSize)`, `HCCL_E_PARA` is returned.
<!-- end id9 -->
- This API only supports the scenario where the communication operator expansion mode is AI CPU.
- This API can be called only on the device side.

## Example

After the symmetric memory window is registered on the host side, pass the window as a parameter to the AI CPU kernel. This function must be compiled to run on the device AI CPU. The following is the pseudocode description:

```c
AicpuKernelFunc(param):
// Obtain the symmetric window from param.
HcclCommSymWindow temp_win = param.win;
void *src_ptr;
void *dest_ptr;
int srcRankId = 0;
int destRankId = 1;
// In the A3 scenario, use win + offset + rank ID to obtain the corresponding address.
HcclSymWinGetPeerPointer(temp_win, 0, srcRankId, &src_ptr);
HcclSymWinGetPeerPointer(temp_win, 0, destRankId, &dest_ptr);
// The obtained address can be directly read and written using data-plane local copy. The caller must prepare the thread and size.
HcommLocalCopyOnThread(thread, dest_ptr, src_ptr, size);
```

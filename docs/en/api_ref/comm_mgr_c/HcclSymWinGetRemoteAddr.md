# HcclSymWinGetRemoteAddr

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T06:29:51.572Z pushedAt=2026-09-29T02:02:05.115Z -->

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
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Obtains the address pointer corresponding to the offset in the remote symmetric memory window of a specified rank ID based on the symmetric memory window resource handle and the offset.

<!-- npu="950" id6 -->
For Ascend 950PR&950DT products, this API supports the URMA scenario.
<!-- end id6 -->

## Function Prototype

```c
HcclResult HcclSymWinGetRemoteAddr(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| winHandle | Input | Symmetric memory window resource handle.<br>For the definition of the HcclCommSymWindow type, see [HcclCommSymWindow](./data_type_definition/HcclCommSymWindow.md). |
| offset | Input | Offset obtained by calling [HcclCommSymWinGet](HcclCommSymWinGet.md). |
| peerRank | Input | Rank ID, with a value range of [0, **rankSize**). |
| ptr | Output | Pointer to the corresponding address in the remote symmetric memory window. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns **HCCL_SUCCESS** on success, and other values on failure.

## Constraints

<!-- npu="950" id7 -->
- For Ascend 950PR&950DT products, only the URMA scenario is supported. In this scenario:
  - Before calling this API, ensure that the symmetric memory window has been registered and that the related URMA communication channel has completed link establishment and remote memory information update.

    When using collective communication APIs, channel creation and remote memory information update are completed internally by the collective communication framework. When using independent communication channel resource APIs, call this API only after the channel is successfully created.
  - The **winHandle** passed in must be a valid registered symmetric memory window handle. If the window handle obtained through [HcclCommSymWinGet](HcclCommSymWinGet.md) is not hit, the returned **winHandle** is empty and cannot be passed to this API. If the remote memory information of the corresponding **peerRank** has not been updated, this API returns an error.
  - If the window is not in URMA mode, this API returns an error and sets ***ptr** to **nullptr**. In this case, use [HcclSymWinGetPeerPointer](HcclSymWinGetPeerPointer.md) to obtain the address instead.
<!-- end id7 -->
- This API only supports the scenario where the communication operator expansion mode is AI CPU.
- This API can be called only on the device side.

## Example

After the symmetric memory window is registered on the host side, pass the window as a parameter to the AI CPU kernel. This function must be compiled to run on the device AI CPU. The following is the pseudocode description:

```c
AicpuKernelFunc(param):
// Obtain the symmetric window from param.
HcclCommSymWindow temp_win = param.win;
void *remote_ptr;
uint32_t peerRankId = 1;
// Use win + offset + peerRank to obtain the remote address corresponding to peerRank.
HcclSymWinGetRemoteAddr(temp_win, 0, peerRankId, &remote_ptr);
// The obtained address can be directly read and written through data-plane local copy. The caller must prepare the thread and size.
HcommLocalCopyOnThread(thread, remote_ptr, src_ptr, size);
```

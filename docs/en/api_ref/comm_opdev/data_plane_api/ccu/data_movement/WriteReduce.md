# WriteReduce

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T12:01:18.465Z pushedAt=2026-10-12T06:12:35.627Z -->

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

Initiates a cross-rank write and in-place reduction on the remote end within a CCU kernel. Through the established `ChannelHandle`, this API sends the local on-chip memory data to the remote end and merges it with the existing data in the remote on-chip memory using the specified operator (`remote = reduce(remote, local, opType)`). When the hardware completes the operation, bit `mask` of `event` is automatically set to **1**. This API is asynchronous.

Note:

- The parameter order follows the "destination first, source second" convention: `WriteReduce(ch, remote, local, ...)`, where `remote` (the destination, the remote accumulator of the reduction) is in the second position and `local` (the source, the local end) is in the third position.
- All `ChannelHandle` objects used in the same kernel must belong to the same die. This API does not perform die verification at the call site, so it does not fail due to die inconsistency (the call site may still return `CCU_E_PTR` because no kernel is being registered, or `CCU_E_NOT_FOUND` because the handle is invalid; for details, see the return value table). Die consistency is uniformly verified by `HcommCcuKernelRegister` (based on all channels used during this kernel). If they are inconsistent, `HcommCcuKernelRegister` returns `CCU_E_PARA` instead of this API's return value.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
CcuResult WriteReduce(ChannelHandle ch, RemoteAddr remote, LocalAddr local,
                      Variable len, HcclDataType dataType, HcclReduceOp opType,
                      Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| ch | Input | Cross-rank channel handle (`ChannelHandle`). The die bound to the channel must belong to the same die as all channels in this kernel (uniformly verified in `HcommCcuKernelRegister`; for details, see the caution above). |
| remote | Input | Target address of the remote on-chip memory (`RemoteAddr`). The remote memory must have a valid initial value written before the call (written by the remote kernel); after the hardware completes, the content at this address is updated to the reduction result. |
| local | Input | Source address of the local on-chip memory (`LocalAddr`). |
| len | Input | Number of bytes to operate, of type `Variable` (variable length at runtime). |
| dataType | Input | Data type. For values, see the `HcclDataType` enumeration. Only the following six types are supported: `HCCL_DATA_TYPE_UINT8`, `HCCL_DATA_TYPE_INT16`, `HCCL_DATA_TYPE_INT32`, `HCCL_DATA_TYPE_FP16`, `HCCL_DATA_TYPE_FP32`, and `HCCL_DATA_TYPE_BFP16`. Other values are rejected and an exception is thrown (with an error code). |
| opType | Input | Reduction operator. For values, see the `HcclReduceOp` enumeration. Only `HCCL_REDUCE_SUM` (sum), `HCCL_REDUCE_MAX` (maximum), and `HCCL_REDUCE_MIN` (minimum) are supported. `HCCL_REDUCE_PROD` is not supported; passing it is rejected and an exception is thrown (with an error code). When the SUM operation is used, the sum result of low-precision input data is first promoted to a higher precision and then adjusted back to the same precision as the input data. |
| event | Input | Completion event object. When the hardware reduction write completes, `event[mask]` is automatically set, and the downstream calls `EventWait(event, mask)` to wait. |
| mask | Input | 16-bit event mask. The default value is `1` (that is, bit0). |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation succeeds. |
| `CCU_E_PTR` | No kernel is currently in the registration phase (the API is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `remote`/`local`/`len`/`event` handle is not registered in the current kernel. |

> [!NOTE] Note
> This API does not return `CCU_E_PARA`; channel die inconsistency is returned by `HcommCcuKernelRegister` as `CCU_E_PARA`. If `dataType`/`opType` is outside the supported range (see the parameter description), an exception is thrown at runtime (with an error code).

## Constraints

- The peer `remote` memory must have a valid initial value written before the call (written by the remote kernel); otherwise, the reduction result is undefined.
- The parameter order is remote end (destination) first and local end (source) second: `WriteReduce(ch, remote, local, ...)`.
- All `ChannelHandle` objects in the same kernel must belong to the same die. Channels of different dies cannot be mixed in the same kernel.
- This API is asynchronous. You must call `EventWait(event, mask)` to wait for the reduction to complete, so that the remote data is guaranteed to be visible.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: Write the local FP16 data to the remote end and perform SUM reduction.
CcuResult MyKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter structure first.
    ChannelHandle ch = params->channelHandle;
    RemoteAddr remote;    // The remote end must have an initial value written in advance.
    LocalAddr src;
    Variable len;
    Event evt;

    WriteReduce(ch, remote, src, len, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```

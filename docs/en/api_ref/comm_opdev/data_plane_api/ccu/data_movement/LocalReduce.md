# LocalReduce

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T11:59:40.917Z pushedAt=2026-10-12T06:12:35.625Z -->

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

Initiates a local reduction operation (asynchronous) within a CCU kernel, merging the source data with the destination memory or multiple CcuBuffers using the specified operator. When the hardware completes the operation, bit `mask` of `event` is automatically set to 1. The following two data paths are supported:

| Overload  | Data Path                                      | Reduction Method                                                                                                                                                                                                                                                     |
| --- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Overload 1 | Local on-chip memory (`src`) → local on-chip memory (`dst`)                 | `dst = reduce(dst, src, opType)`, where the input and output types are the same.                                                                                                                                                                                                                |
| Overload 2 | N local CcuBuffers → `buffers[0]` (2 ≤ N ≤ 8) | `buffers[0] = reduce(buffers[0..N-1], opType)`, where the hardware reduces N CcuBuffers to `buffers[0]` in one pass. The input data type can be (`HCCL_DATA_TYPE_UINT8`/`HCCL_DATA_TYPE_INT16`/`HCCL_DATA_TYPE_INT32`/`HCCL_DATA_TYPE_FP16`/`HCCL_DATA_TYPE_BFP16`/`HCCL_DATA_TYPE_FP32`), and the output data type supports the same precision as the input, or precision expansion under `HCCL_REDUCE_SUM` (for details, see the `outputDataType` parameter description). |

> [!NOTE] Note
> This API is asynchronous. After calling it, you must wait for the reduction to complete by calling `EventWait(event, mask)`. Otherwise, the data in the destination memory is undefined. The reduction is an in-place operation. Before the call, `dst` (overload 1) or `buffers[0]` (overload 2) must already contain a valid initial value (such as 0 or negative infinity).

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
// Overload 1: local on-chip memory → local on-chip memory reduction.
CcuResult LocalReduce(LocalAddr dst, LocalAddr src, Variable len,
                      HcclDataType dataType, HcclReduceOp opType,
                      Event event, uint16_t mask = 1);
// Overload 2: reduce N local CcuBuffer instances to buffers[0] (2 ≤ count ≤ 8).
CcuResult LocalReduce(CcuBuffer* buffers, uint32_t count,
                      HcclDataType dataType, HcclDataType outputDataType,
                      HcclReduceOp opType,
                      Variable len, Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

### Overload 1 Parameters

| Parameter            | Input/Output | Description                                                                                                                                                                                            |
| -------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| dst            | Input/Output | Target on-chip memory address (`LocalAddr`). A valid initial value must be written before the call; after the hardware completes, it is updated to the reduction result.                                                                                                                                                 |
| src            | Input    | Source on-chip memory address (`LocalAddr`).                                                                                                                                                                          |
| len            | Input    | Number of bytes to operate on. The type is `Variable` (variable length at runtime).                                                                                                                                                                 |
| dataType       | Input    | Data type. For values, see the `HcclDataType` enumeration. Only the following six types are supported: `HCCL_DATA_TYPE_UINT8`, `HCCL_DATA_TYPE_INT16`, `HCCL_DATA_TYPE_INT32`, `HCCL_DATA_TYPE_FP16`, `HCCL_DATA_TYPE_FP32`, and `HCCL_DATA_TYPE_BFP16`. Other values are rejected and an exception is thrown (with an error code). |
| opType         | Input    | Reduction operator. For values, see the `HcclReduceOp` enumeration. Only `HCCL_REDUCE_SUM` (sum), `HCCL_REDUCE_MAX` (maximum), and `HCCL_REDUCE_MIN` (minimum) are supported. `HCCL_REDUCE_PROD` is not supported; passing it is rejected and an exception is thrown (with an error code). When the SUM operation is used, the sum result of low-precision input data is first promoted to a higher precision and then adjusted back to the same precision as the input data. |
| event          | Input    | Completion event object. When the hardware reduction completes, `event[mask]` is automatically set.                                                                                                                                                              |
| mask           | Input    | 16-bit event mask. The default value is `1` (that is, bit0).                                                                                                                                                                       |

### Overload 2 Parameters

| Parameter            | Input/Output | Description                                                                                                                                                                                                                                                                                                       |
| -------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| buffers        | Input/Output | Start address of the `CcuBuffer` array (`CcuBuffer*`), which cannot be `nullptr`. It is recommended to allocate it through `ccu::Array<CcuBuffer>` to ensure physical contiguity. `buffers[0]` is the output location of the reduction result and must be written with a valid initial value before the call. In the precision expansion scenario (low-precision input + SUM promoted-precision output, for example, when INT8→FP32 the output elements are 4× the input), the passed-in CcuBuffer array must cover both the input and the expanded output occupancy (for example, for 2-way INT8 input promoted to FP32 output, 4 CcuBuffers must be reserved instead of 2); insufficient reservation causes hardware read/write out-of-bounds and undefined behavior.                                                          |
| count          | Input    | Number of CcuBuffers. The value range is `[2, 8]` (exceeding the upper limit throws an exception and carries an error code). `count == 0` is directly rejected and returns `CCU_E_PARA`; `count == 1` is not rejected but the hardware behavior is undefined. For the single-CcuBuffer scenario, use overload 1. `count` must equal the actual length of the `buffers` array.                                                                                                                                                         |
| dataType       | Input    | Input data type. For values, see the `HcclDataType` enumeration. Only the following six types are supported: `HCCL_DATA_TYPE_UINT8`, `HCCL_DATA_TYPE_INT16`, `HCCL_DATA_TYPE_INT32`, `HCCL_DATA_TYPE_FP16`, `HCCL_DATA_TYPE_FP32`, and `HCCL_DATA_TYPE_BFP16`. Other values are rejected and an exception is thrown (with an error code). |
| outputDataType | Input    | Output data type. For values, see the `HcclDataType` enumeration. Two combinations are supported: ① same precision - the value is the same as `dataType`; ② promoted precision - supported only when `opType == HCCL_REDUCE_SUM`, which reduces low-precision inputs and then upgrades them to high-precision output (for example, `HCCL_DATA_TYPE_INT8` → `HCCL_DATA_TYPE_FP32`; see "Call Example - Scenario 2"). Other `dataType`/`outputDataType` combinations return `CCU_E_NOT_SUPPORT` (see the return value table). In the promoted-precision scenario, `buffers` must reserve CcuBuffers according to the precision expansion ratio. For details, see the `buffers` parameter description. |
| opType         | Input    | Reduction operator. For values, see the `HcclReduceOp` enumeration. Only `HCCL_REDUCE_SUM`, `HCCL_REDUCE_MAX`, and `HCCL_REDUCE_MIN` are supported; `HCCL_REDUCE_PROD` is not supported and is rejected with an exception (carrying an error code) when passed.                                                                                                                                                                              |
| len            | Input    | Number of bytes of each CcuBuffer participating in the reduction. The type is `Variable` and cannot exceed the size of a single slice (4096 bytes).                                                                                                                                                                                                                                                         |
| event          | Input    | Completion event object.                                                                                                                                                                                                                                                                                                  |
| mask           | Input    | 16-bit event mask. The default value is `1` (that is, bit0).                                                                                                                                                                                                                                                                                  |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value                 | Description                                                                    |
| ------------------- | --------------------------------------------------------------------- |
| `CCU_SUCCESS`       | Operation succeeded.                                                                 |
| `CCU_E_PARA`        | Parameter error, including `buffers` being `nullptr` or `count` being 0.                                 |
| `CCU_E_NOT_SUPPORT` | Overload 2 only: the `dataType`/`outputDataType` combination does not satisfy the promoted-precision/same-precision constraint (see the `outputDataType` description). |

> [!NOTE] Note
> If the values of `dataType`/`opType` are outside the supported range (for details, see the parameter description), an exception (carrying an error code) will be thrown during runtime rather than being reported through the return value.

## Constraints

- The reduction is an in-place operation. Before the call, `dst` (overload 1) or `buffers[0]` (overload 2) must already contain a valid initial value; otherwise, the reduction result is undefined.
- For overload 2, `buffers` must point to a physically contiguous `CcuBuffer` array. You are advised to allocate it using `ccu::Array<CcuBuffer>`.
- For overload 2, `count` is in the range `[2, 8]`. When `count > 8`, an exception is thrown (carrying an error code). When `count == 0`, `CCU_E_PARA` is returned directly. When `count == 1`, it is not rejected but the hardware behavior is undefined (use overload 1 instead).
- For overload 2, in the precision expansion scenario (low-precision input + SUM with promoted-precision output), `buffers` must reserve additional CcuBuffer according to the expansion ratio to accommodate the expanded output; otherwise, the hardware reads/writes out of bounds and the behavior is undefined.
- When `CcuBuffer` is involved (overload 2), `len` must not exceed the size of a single `CcuBuffer` (4096 bytes). This upper limit must be guaranteed by the caller; exceeding it causes undefined hardware behavior at runtime.
- This API is asynchronous. You must wait for the reduction to complete by calling `EventWait(event, mask)` before reading the result.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario 1: Local on-chip memory -> local on-chip memory reduction (FP16 SUM).
CcuResult MyKernel(CcuKernelArg arg) {
    LocalAddr dst, src;
    Variable len;
    Event evt;

    // dst must be pre-written with an initial value (for example, 0.0).
    LocalReduce(dst, src, len, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}

// Scenario 2: Four CcuBuffers are reduced to buffers[0], with INT8 input upcast to FP32 output (a valid "input != output" combination;
//        INT8 -> FP32 = 4x expansion, so the buffers array reserves CcuBuffers by the expansion ratio to accommodate the expanded output).
CcuResult MyKernel2(CcuKernelArg arg) {
    Array<CcuBuffer> bufs(4);    // Four physically contiguous CcuBuffers.
    Variable len;
    Event evt;

    // bufs[0] must be pre-written with an initial value.
    LocalReduce(bufs.data(), 4,
                HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_FP32,
                HCCL_REDUCE_SUM, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}

// Scenario 3: conventional reduction with the same input and output type (FP16 SUM).
CcuResult MyKernel3(CcuKernelArg arg) {
    Array<CcuBuffer> bufs(4);
    Variable len;
    Event evt;

    LocalReduce(bufs.data(), 4,
                HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP16,
                HCCL_REDUCE_SUM, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```

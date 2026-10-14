# Write

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T12:01:25.701Z pushedAt=2026-10-12T06:12:35.629Z -->

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

Initiates a cross-rank (asynchronous) write operation within a CCU kernel. It writes local data to the remote on-chip memory through the established `ChannelHandle`. When the hardware completes the operation, bit `mask` of `event` is automatically set to **1**. The following two source types are supported:

| Overload | Source | Target |
| --- | --- | --- |
| Overload 1 | Local on-chip memory (`LocalAddr`) | Remote on-chip memory (`RemoteAddr`) |
| Overload 2 | Local CcuBuffer (`CcuBuffer`) | Remote on-chip memory (`RemoteAddr`) |

Note:

- The parameter order follows the "destination first, source last" convention: `Write(ch, remote, local, ...)`, that is, `remote` (destination, remote end) is in the second position and `local` (source, local end) is in the third position. This is the reverse of the order in `Read`, where `local` comes first. Do not confuse the two. The C++ type system prevents parameter order errors at compilation time through the different types of `RemoteAddr` and `LocalAddr`.
- All `ChannelHandle` objects used in the same kernel must belong to the same die. This API does not perform die verification at the call site, so it does not fail due to die inconsistency (the call site may still return `CCU_E_PTR` because no kernel is being registered, or `CCU_E_NOT_FOUND` because the handle is invalid; for details, see the return value table). Die consistency is uniformly verified by `HcommCcuKernelRegister` (based on all channels used during this kernel). If they are inconsistent, `HcommCcuKernelRegister` returns `CCU_E_PARA` instead of this API's return value.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
// Overload 1: local on-chip memory → remote on-chip memory.
CcuResult Write(ChannelHandle ch, RemoteAddr remote, LocalAddr local,
                Variable len, Event event, uint16_t mask = 1);
// Overload 2: local CcuBuffer → remote on-chip memory.
CcuResult Write(ChannelHandle ch, RemoteAddr remote, CcuBuffer local,
                Variable len, Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| ch | Input | Cross-rank channel handle (`ChannelHandle`). The die bound to the channel must belong to the same die as all channels in this kernel (uniformly verified in `HcommCcuKernelRegister`; for details, see the caution above). |
| remote | Input | Target on-chip memory address on the remote end (`RemoteAddr`, a composite object of the remote on-chip memory address and token). |
| local | Input | Local source address. For overload 1, it is `LocalAddr` (a composite object of the local on-chip memory address and token); for overload 2, it is `CcuBuffer` (a local CcuBuffer slice object, with a maximum of 4096 bytes per slice). |
| len | Input | Number of bytes to write, of the `Variable` type (variable length at runtime). For overload 2, it cannot exceed 4096 bytes. |
| event | Input | Completion event object. When the hardware write completes, `event[mask]` is automatically set, and the downstream calls `EventWait(event, mask)` to wait. |
| mask | Input | 16-bit event mask. The default value is `1` (that is, bit0). |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation succeeds. |
| `CCU_E_PTR` | No kernel is currently in the registration state (the API is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `remote`/`local`/`len`/`event` handle is not registered in the current kernel. |

> [!NOTE] Note
> This API does not return `CCU_E_PARA`; channel die inconsistency is returned as `CCU_E_PARA` by `HcommCcuKernelRegister`. For details, see [Description](#description).

## Constraints

- The parameter order is remote end (destination) first and local end (source) second: `Write(ch, remote, local, ...)`.
- All `ChannelHandle` objects in the same kernel must belong to the same die. Channels of different dies cannot be mixed in the same kernel.
- In overload 2, `len` cannot exceed the size of a single `CcuBuffer` slice (4096 bytes). The caller must ensure this upper limit; exceeding it leads to undefined hardware behavior at runtime.
- This API is asynchronous. You must call `EventWait(event, mask)` to wait for the write to complete before the remote data is guaranteed to be visible.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario 1: local on-chip memory -> remote on-chip memory
CcuResult MyKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter struct first.
    ChannelHandle ch = params->channelHandle;
    RemoteAddr remote;
    LocalAddr src;
    Variable len;
    Event evt;

    Write(ch, remote, src, len, evt);    // remote comes first, local comes last.
    EventWait(evt);
    return CCU_SUCCESS;
}

// Scenario 2: local CcuBuffer -> remote on-chip memory
CcuResult MyKernel2(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter struct first.
    ChannelHandle ch = params->channelHandle;
    RemoteAddr remote;
    CcuBuffer buf;
    Variable len;
    Event evt;

    Write(ch, remote, buf, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```

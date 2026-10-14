# Read

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T11:59:31.648Z pushedAt=2026-10-12T06:12:35.624Z -->

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

Initiates a cross-rank read operation (asynchronous) within a CCU kernel. It reads data from the remote on-chip memory to the local end through the established `ChannelHandle`, and automatically sets bit `mask` of `event` to 1 when the hardware completes the operation. The following two target types are supported:

| Overload | Target | Source |
| --- | --- | --- |
| Overload 1 | Local on-chip memory (`LocalAddr`) | Remote on-chip memory (`RemoteAddr`) |
| Overload 2 | Local CcuBuffer (`CcuBuffer`) | Remote on-chip memory (`RemoteAddr`) |

Note:

- The parameter order follows the "destination first, source last" convention: `Read(ch, local, remote, ...)`, that is, `local` (destination) is in the second position and `remote` (source) is in the third position. This is the reverse of the order in `Write`, where `remote` comes first. Do not confuse them. The C++ type system prevents parameter order errors at compilation time through the distinct types of `LocalAddr` and `RemoteAddr`.
- All `ChannelHandle` objects used in the same kernel must belong to the same die. This API does not perform die verification at the call site, so it does not fail due to die inconsistency (the call site may still return `CCU_E_PTR` because no kernel is being registered, or `CCU_E_NOT_FOUND` because the handle is invalid; for details, see the return value table). Die consistency is uniformly verified by `HcommCcuKernelRegister` (based on all channels used during this kernel). If they are inconsistent, `HcommCcuKernelRegister` returns `CCU_E_PARA` instead of this API's return value.

## Function Prototype

```cpp
namespace AscendC {
namespace ccu {
// Overload 1: Remote on-chip memory → Local on-chip memory.
CcuResult Read(ChannelHandle ch, LocalAddr local, RemoteAddr remote,
               Variable len, Event event, uint16_t mask = 1);
// Overload 2: Remote on-chip memory → Local CcuBuffer.
CcuResult Read(ChannelHandle ch, CcuBuffer local, RemoteAddr remote,
               Variable len, Event event, uint16_t mask = 1);
} // namespace ccu
} // namespace AscendC
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| ch | Input | Cross-rank channel handle (`ChannelHandle`). The die bound to the channel must be the same die as all channels in the current kernel (uniformly verified in `HcommCcuKernelRegister`; for details, see [Description](#description)). |
| local | Input/Output | Local destination address. For overload 1, it is `LocalAddr` (a composite object of the local on-chip memory address and token); for overload 2, it is `CcuBuffer` (a local CcuBuffer slice object, with a maximum of 4096 bytes per slice). |
| remote | Input | Remote source address (`RemoteAddr`, a composite object of the remote on-chip memory address and token). |
| len | Input | Number of bytes to read, of the `Variable` type (variable length at runtime). For overload 2, it cannot exceed 4096 bytes. |
| event | Input | Completion event object. When the hardware completes the read, `event[mask]` is automatically set, and the downstream calls `EventWait(event, mask)` to wait. |
| mask | Input | 16-bit event mask. The default value is `1` (that is, bit0). |

## Return Value

[CcuResult](../../../datatype_definition/CcuResult.md): The API returns `CCU_SUCCESS` on success, and other values on failure.

| Return Value | Description |
| --- | --- |
| `CCU_SUCCESS` | The operation succeeds. |
| `CCU_E_PTR` | No kernel is currently in the registration state (the API is called outside the kernel registration phase). |
| `CCU_E_NOT_FOUND` | The passed `local`/`remote`/`len`/`event` handle is not registered in the current kernel. |

> [!NOTE] Note
> This API does not return `CCU_E_PARA`; channel die inconsistency is returned as `CCU_E_PARA` by `HcommCcuKernelRegister`. For details, see [Description](#description).

## Constraints

- The parameter order is local end (destination) first and remote end (source) last: `Read(ch, local, remote, ...)`.
- All `ChannelHandle` objects in the same kernel must belong to the same die. Channels of different dies cannot be mixed in the same kernel.
- In overload 2, `len` cannot exceed the size of a single `CcuBuffer` slice (4096 bytes). The caller must ensure this upper limit; exceeding it leads to undefined hardware behavior at runtime.
- This API is an asynchronous operation. You must wait for the read to complete by calling `EventWait(event, mask)` before accessing the target memory.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario 1: Read from remote on-chip memory to local on-chip memory.
CcuResult MyKernel(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter structure first.
    ChannelHandle ch = params->channelHandle;
    LocalAddr dst;
    RemoteAddr remote;
    Variable len;
    Event evt;

    Read(ch, dst, remote, len, evt);    // local comes first, remote comes last.
    EventWait(evt);
    return CCU_SUCCESS;
}

// Scenario 2: Read from remote on-chip memory to local CcuBuffer.
CcuResult MyKernel2(CcuKernelArg arg) {
    auto *params = static_cast<MyKernelArg *>(arg);  // CcuKernelArg is void*. Cast it to the user input parameter structure first.
    ChannelHandle ch = params->channelHandle;
    CcuBuffer buf;
    RemoteAddr remote;
    Variable len;
    Event evt;

    Read(ch, buf, remote, len, evt);
    EventWait(evt);
    return CCU_SUCCESS;
}
```

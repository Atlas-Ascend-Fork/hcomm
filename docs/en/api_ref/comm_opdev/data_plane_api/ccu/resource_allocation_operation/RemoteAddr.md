# RemoteAddr

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T12:07:11.031Z pushedAt=2026-10-12T06:12:35.636Z -->

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

`ccu::RemoteAddr` is a C++ wrapper class for the remote on-chip memory address within a CCU kernel. It is a composite object of "address (`Address`) + token (`Variable`)", and its structure mirrors that of [LocalAddr](LocalAddr.md).

- Allocation on construction: The default constructor allocates one Address (for `addr`) and one Variable (for `token`) at once.
- No release on destruction: The destructor does not release hardware resources. The virtual handle becomes invalid after translation is complete, and the physical resources are managed and reclaimed uniformly over the lifecycle of the CCU instance.

`RemoteAddr` is dedicated to cross-rank read/write operations (`Read`/`Write`/`ReadReduce`/`WriteReduce`), carrying the physical address and security token of the target memory on the remote rank. `addr` and `token` must come from the remote rank and must not be mixed with the `LocalAddr` fields of the local rank.

## Class Declaration

```cpp
namespace AscendC {
namespace ccu {
class RemoteAddr final {
public:
    RemoteAddr();                        // Allocation on construction (allocate both Address and Variable).
    Address addr;                        // Remote on-chip memory address field.
    Variable token;                      // Remote security token field.
    CcuRemoteAddrHandle handle{0};      // Composite handle.
};
} // namespace ccu
} // namespace AscendC
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `RemoteAddr ra;` | Allocates one Address and one Variable virtual handle at once, and writes back the three handles `ra.handle`/`ra.addr.handle`/`ra.token.handle`. |

The C++ constructor only allocates virtual handles: it always succeeds when called within the kernel registration phase; if it is not called within the kernel registration phase, an exception is thrown during construction (carrying error code `CCU_E_PTR`). When the physical resources of Address/Variable are insufficient, `CCU_E_UNAVAIL` is returned during the `HcommCcuKernelRegister` phase, rather than being thrown during construction.

> [!CAUTION] Caution
> The `RemoteAddr` class has copy/move constructors, which only copy the three fields `handle`, `addr.handle`, and `token.handle` and do not allocate new Address/Variable. After `RemoteAddr r2 = r1;`, `r1` and `r2` point to the same group of registers. `operator=(const RemoteAddr&)` executes a device-side register assignment instruction (with the same semantics as `LocalAddr`). The two are asymmetric, so pay special attention when using them.

## Field Description

| Field | Type | Description |
| --- | --- | --- |
| `addr` | `Address` | Target address of the remote on-chip memory. It must be filled with the VA and token of the remote rank (the result of the remote `HcommCcuGetMemToken`). For operator semantics, see [Address](Address.md). |
| `token` | `Variable` | Remote security token value. It must be paired with `addr`, come from the remote rank, and must not be mixed with the local token. For operator semantics, see [Variable](Variable.md). |

## Constraints

> [!CAUTION] Caution
> `token` is security information. It must not be printed in host or device logs, and must not be passed across ranks in plaintext. The `addr` and `token` fields must be the values of the remote rank's target memory, and their source is completely different from that of the local `LocalAddr` field.

- **RemoteAddr** can only be constructed during the kernel registration phase.
- The destructor does not release hardware resources. Do not save or compare the `handle` value outside the kernel; the handle becomes invalid once translation is complete.
- Do not call `Address()`/`Variable()` separately on the embedded `addr`/`token` fields to construct new objects — `RemoteAddr()` has already allocated all subfields at once.
- `RemoteAddr` must be used together with `ChannelHandle` to perform cross-rank RDMA operations over an established channel. Using it without a channel results in undefined behavior.

## Example

```cpp
using namespace AscendC::ccu;

// The address and token of the remote rank must be obtained from the remote end through inter-process communication (such as a message passing framework),
// and then passed into the kernel through kernelArg or taskArgs+LoadArg.

CcuResult MyKernel(CcuKernelArg arg) {
    auto* params = static_cast<MyKernelArg*>(arg);

    RemoteAddr remote;
    // Assign from kernelArg in the registration phase (fixed as an immediate value).
    remote.addr = params->remoteAddr;
    remote.token = params->remoteToken;

    // Use with the cross-rank Read/Write APIs.
    ChannelHandle ch = params->channelHandle;
    LocalAddr local;
    local.addr = params->localAddr;
    local.token = params->localToken;
    Variable len;
    Event evt;
    len = 1024;

    Read(ch, local, remote, len, evt);   // Read from the remote on-chip memory to the local on-chip memory.
    EventWait(evt);

    return CCU_SUCCESS;
}
```

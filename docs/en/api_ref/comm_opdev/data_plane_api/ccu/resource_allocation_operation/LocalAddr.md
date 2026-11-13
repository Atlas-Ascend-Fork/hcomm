# LocalAddr

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:10:48.213Z pushedAt=2026-09-30T08:10:06.463Z -->

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

`ccu::LocalAddr` is a C++ wrapper class for the local HBM address within a CCU kernel. It is a composite object of "address (`Address`) + token (`Variable`)".

- Allocation on construction: The default constructor allocates one GSA (for `addr`) and one XN (for `token`) at a time.
- No release on destruction: The destructor does not release hardware resources; the virtual handle becomes invalid after translation completes, and the physical resource is managed and reclaimed uniformly over the lifecycle of the CCU instance.

CCU hardware does not accept process virtual addresses. To access HBM, you must use the token converted by `HcommCcuGetMemToken` (called on the host side). The `addr` field of `LocalAddr` stores the physical address (or the tokenized VA), and the `token` field stores the matching security token value.

## Class Declaration

```cpp
namespace hcomm {
namespace ccu {
class LocalAddr final {
public:
    LocalAddr();                         // Allocation on construction (allocates GSA + XN simultaneously).
    Address addr;                        // Local HBM address field (GSA register).
    Variable token;                      // Security token field (XN register).
    CcuLocalAddrHandle handle{0};       // Composite handle.
};
} // namespace ccu
} // namespace hcomm
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `LocalAddr la;` | Allocates one GSA and one XN virtual handle at a time, and fills in the three handles `la.handle`/`la.addr.handle`/`la.token.handle`. |

The C++ constructor only allocates virtual handles: when called during the kernel registration phase, it always succeeds; if it is not called during the kernel registration phase, an exception is thrown at construction time (carrying the error code `CCU_E_PTR`). When the GSA/XN physical resources are insufficient, `CCU_E_UNAVAIL` is returned during the `HcommCcuKernelRegister` phase, rather than being thrown at construction time.

> [!CAUTION] Caution
> The `LocalAddr` class has copy/move constructors, which only copy the three fields `handle` / `addr.handle` / `token.handle` and do not allocate new GSA/XN resources. After `LocalAddr l2 = l1;`, `l1` and `l2` point to the same group of registers. However, `operator=(const LocalAddr& other)` (assignment, not construction) is not a handle copy: it executes `this->addr = other.addr; this->token = other.token;`, that is, it delivers two device-side register assignment instructions (see `Address::operator=` / `Variable::operator=`), performing a value transfer between two independent groups of registers. The two semantics are asymmetric, so pay special attention when using them.

## Field Description

| Field | Type | Description |
| --- | --- | --- |
| `addr` | `Address` | Local HBM address. Assign a value through `la.addr = imm` (immediate value) or `la.addr = var` (Variable). For operator semantics, see [Address](Address.md). |
| `token` | `Variable` | Security token value. After the host obtains it by calling `HcommCcuGetMemToken`, pass it into the kernel through `kernelArg` or `taskArgs`+`LoadArg`, and then assign it to `la.token`. For operator semantics, see [Variable](Variable.md). |

## Constraints

> [!CAUTION] Caution
> `token` is security information. It must not be printed in host or device logs, and must not be passed across ranks in plaintext.

- **LocalAddr** can be constructed only during the kernel registration phase.
- The destructor does not release hardware resources. Do not save or compare the `handle` value outside the kernel; the handle becomes invalid once translation is complete.
- Do not separately call `Address()`/`Variable()` on the embedded `addr`/`token` fields to construct new objects — `LocalAddr()` has already completed the allocation of all subfields at once.
- The `addr` field must be assigned a physical address accessible to the CCU or a tokenized VA. Passing a process virtual address that has not been tokenized directly will trigger a driver error.
- The `token` usually comes from the `HcommCcuGetMemToken` call on the host side, and must be paired with `addr`. It must not be mixed with the remote token.

## Example

```cpp
using namespace hcomm::ccu;

// Host side (before kernel registration):
// uint64_t tokenInfo = 0;
// HcommCcuGetMemToken(srcVa, size, &tokenInfo);
// kernelArg.srcAddr = srcVa;
// kernelArg.srcToken = tokenInfo;

CcuResult MyKernel(CcuKernelArg arg) {
    auto* params = static_cast<MyKernelArg*>(arg);

    LocalAddr src;
    // Assign from the kernelArg of the registration phase (fixed as an immediate value).
    src.addr = params->srcAddr;
    src.token = params->srcToken;

    // Or inject at runtime through taskArgs (more flexible).
    Variable addrVar, tokenVar;
    LoadArg(addrVar, 0);    // taskArgs[0] = address.
    LoadArg(tokenVar, 1);   // taskArgs[1] = token
    LocalAddr src2;
    src2.addr = addrVar;    // Runtime dynamic address.
    src2.token = tokenVar;

    return CCU_SUCCESS;
}
```

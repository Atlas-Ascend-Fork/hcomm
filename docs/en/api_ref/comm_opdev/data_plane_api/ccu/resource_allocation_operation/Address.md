# Address

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:07:34.684Z pushedAt=2026-09-30T07:15:14.373Z -->

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

`ccu::Address` is a C++ wrapper class for the address register (GSA) within a CCU kernel.

- Allocation on construction: The default constructor automatically requests one GSA virtual handle.
- No release on destruction: The destructor does not release hardware resources. The virtual handle becomes invalid after translation is complete, and the physical resources are managed and reclaimed uniformly over the lifecycle of the CCU instance.
- Operators are device operations: The assignment and arithmetic operators on `Address` describe operations executed on the device (hardware), rather than immediate computation on the host; at runtime, they operate on the corresponding GSA register.

Unlike `Variable` (a scalar value), `Address` specifically carries an address value (the physical address of HBM). A typical usage is to add a base address and an offset to obtain the target address for use by data movement APIs.

> [!NOTE] Note
> `Address` carries a device-side address value and cannot be directly dereferenced on the host side. An address cannot be assigned to a variable (no reverse assignment API is provided), but a variable can be assigned to an address (`addr = var;`).

## Class Declaration

```cpp
namespace hcomm {
namespace ccu {
class Address final {
public:
    Address();                                                   // Construct and allocate.
    void operator=(uint64_t immediate) const;                   // Assign an immediate address.
    void operator=(const Variable& var) const;                  // Assign the variable value to address.
    void operator=(const Address& other) const;                 // Assign between Address objects.
    void operator+=(const Variable& var) const;                 // Add Variable to Address in place.
    void operator+=(const Address& other) const;               // Add Address to Address in place.
    /* Internal expression object. */ operator+(const Address& that) const;    // Address + Address.
    /* Internal expression object. */ operator+(const Variable& var) const;    // Address + Variable.
    CcuAddressHandle handle{0};                                 // Virtual handle.
};
// Variable + Address (commutative, global function).
/*Internal expression object.*/ operator+(const Variable& var, const Address& addr);
} // namespace ccu
} // namespace hcomm
```

## Constructor Description

| Construction Form | Description |
| --- | --- |
| `Address a;` | Allocates one GSA virtual handle. It can only be called during the kernel registration phase (inside the kernel function body executed by `HcommCcuKernelRegister`). |

> [!CAUTION] Caution
> The `Address` class has copy/move constructors, which only copy the `handle` field and do not allocate a new GSA. After `Address a2 = a1;`, `a1` and `a2` point to the same address register. If an independent `Address` is required, you must explicitly use the default constructor `Address a2;`. `operator=(const Address&)` delivers a device-side register assignment instruction, which moves values between two independent registers.

If construction fails, an exception is thrown (carrying the [CcuResult](../../../datatype_definition/CcuResult.md) error code).

## Operator Description

### Assignment Operators

| Expression Syntax | Hardware Semantics |
| --- | --- |
| `addr = imm;` (`imm` is `uint64_t`) | `GSA_addr ← imm`. The address immediate is determined during the registration phase and is immutable at runtime. |
| `addr = var;` (`var` is `Variable`) | `GSA_addr ← XN_var`. Writes the runtime value of the variable into the address register, suitable for runtime dynamic addresses. |
| `dst = src;` (`src` is `Address`) | `GSA_dst ← GSA_src`. Register assignment between addresses. |

### Arithmetic Operator

| Expression Syntax | Hardware Semantics |
| --- | --- |
| `r = a + b;` (`a`/`b` are both `Address`) | `GSA_r ← GSA_a + GSA_b`. |
| `r = addr + var;` / `r = var + addr;` | `GSA_r ← GSA_addr + XN_var`. The two forms have the same semantics (commutative law). |
| `addr += var;` (`var` is `Variable`) | `GSA_addr ← GSA_addr + XN_var` (in-place operation, saving one instruction compared with `addr = addr + var`). |
| `addr += other;` (`other` is `Address`) | `GSA_addr ← GSA_addr + GSA_other`. |

> [!NOTE] Note
> Both `r = addr + var` and `r = var + addr` are valid (the latter uses the global `operator+(Variable, Address)` friend function). The two forms have the same semantics (commutative law), and the parameter order is handled uniformly inside the operator overloading.

## Constraints

- An `Address` can only be constructed during the kernel registration phase.
- The destructor does not release hardware resources. Do not save or compare the `handle` value outside the kernel; the handle becomes invalid once translation is complete.
- `Address` cannot be assigned to `Variable`, that is, the `Variable = Address` operation is not provided.
- Currently, only addition is supported. Subtraction, multiplication, and division are not supported.
- An immediate cannot directly participate in arithmetic (`addr + 0x100` is invalid). You must first assign the offset to a variable before using it in an operation.
- The C++ constructor only allocates a virtual handle and always succeeds without throwing an exception. When `GSA` physical resources are insufficient, `HcommCcuKernelRegister` returns `CCU_E_UNAVAIL` during the registration phase instead of throwing an exception at construction time.

## Example

```cpp
using namespace hcomm::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    Address base, dst;
    Variable offset, stride;

    // Assign an immediate address (fixed during the registration phase).
    base = 0x80000000ULL;

    // Assign a variable value to address (runtime dynamic address).
    // offset is injected at launch time through LoadArg.
    LoadArg(offset, 0);
    dst = offset;         // GSA_dst ← XN_offset (determined at runtime).

    // Address + Variable offset (two equivalent ways).
    stride = 4096;
    dst = base + stride;  // r = addr + var
    dst = stride + base;  // r = var + addr (equivalent).

    // In-place offset of address.
    base += stride;        // base += 4096

    // Address + Address
    Address result;
    result = base + dst;

    return CCU_SUCCESS;
}
```

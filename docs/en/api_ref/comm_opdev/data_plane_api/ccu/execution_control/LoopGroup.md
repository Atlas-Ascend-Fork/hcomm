# LoopGroup

<!-- md-trans-meta sourceCommit=a0b71989b4715e3833039dcf5ff93eee06a80eba translatedAt=2026-10-08T12:02:08.377Z pushedAt=2026-10-12T06:12:35.630Z -->

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

Organizes multiple `ccu::Loop` objects into a group to share the same **LoopEngine** resource pool, preventing resource exhaustion caused by multiple independent `Loop` objects each exclusively occupying the LoopEngine. This API is a hardware LoopGroup class in the CCU kernel.

When constructing `ccu::LoopGroup`, each `Loop` in the passed `loops` list is automatically added to the group.

## Class Definition

```cpp
namespace AscendC {
namespace ccu {

class LoopGroup {
public:
    // Construction method 1: config-based (group parameters are known at registration time).
    LoopGroup(const CcuLoopGroupConfig &loopGroupCfg, uint32_t maxLoopNum,
              const std::vector<Loop> &loops);

    // Construction method 2: var-based (group parameters are determined by variables at runtime).
    LoopGroup(Variable &parallelCfg, Variable &offsetCfg, uint32_t maxLoopNum,
              const std::vector<Loop> &loops);
};

} // namespace ccu
} // namespace AscendC
```

## Parameters

### Construction Method 1: config-based

| Parameter | Input/Output | Description |
| --- | --- | --- |
| loopGroupCfg | Input | LoopGroup configuration, of the `CcuLoopGroupConfig` type. See the following table for the field descriptions. |
| maxLoopNum | Input | Maximum number of loops that this group can hold. The framework reserves the LoopEngine resource pool capacity accordingly. |
| loops | Input | List of `ccu::Loop` objects to be added to this group. Each loop in the list is automatically registered to the group in the `LoopGroup` constructor. |

### Construction Method 2: var-based

| Parameter | Input/Output | Description |
| --- | --- | --- |
| parallelCfg | Input | Parallel configuration variable that determines parallel parameters at runtime. This parameter contains 64 bits, where [47:41] indicates the number of loop instructions contained in the loop group, [54:48] indicates the loop offset at which the loop instructions contained in the loop group need to complete automatic loop unrolling, and [61:55] indicates the number of times the loop needs to be unrolled. Example: parallelCfg[47:41]=4 indicates that the program contains 4 loop instructions, parallelCfg[54:48]=1 indicates that unrolling starts from the loop numbered 1, and parallelCfg[61:55]=3 indicates that loop1, loop2, and loop3 are each copied and unrolled 3 times, while loop0 is not copied. After unrolling, the total number of loops is 4 + (4-1) × 3 = 13. |
| offsetCfg | Input | Offset configuration variable, which determines the offset parameters at runtime. This parameter contains 64 bits, where [9:0] indicates the event resource offset used after the loop is unrolled, [20:10] indicates the CcuBuffer resource offset used after the loop is unrolled, and [52:21] indicates the address accumulation offset used by each data transfer instruction after the loop is unrolled. |
| maxLoopNum | Input | Same as construction method 1. |
| loops | Input | Same as construction method 1. |

### CcuLoopGroupConfig

Fields in the parameter structure used by the config-based construction method are as follows:

| Field | Type | Description |
| --- | --- | --- |
| `cloneNum` | `uint32_t` | Number of parallel clones, specifying the number of instances executed concurrently within the group. |
| `cloneLoopOffset` | `uint32_t` | Loop offset between clone instances. |
| `addrOffset` | `uint32_t` | Byte offset of `Address` between clone instances. |
| `ccuBufferOffset` | `uint32_t` | Slice offset of `CcuBuffer` between clone instances. |
| `eventOffset` | `uint32_t` | Bit offset of `Event` slots between clone instances. |

## Exceptions

When `ccu::LoopGroup` fails to be constructed, an exception is thrown (carrying a [CcuResult](../../../datatype_definition/CcuResult.md) error code). Common causes:

| Cause | Error Code |
| --- | --- |
| `maxLoopNum` is 0, or the runtime configuration variable for var-based construction is empty. | `CCU_E_PARA` |
| The number of loops actually added exceeds `maxLoopNum` (insufficient LoopEngine pool capacity). | `CCU_E_PARA` |
| Insufficient physical resources. | `CCU_E_UNAVAIL`, etc. |

## Constraints

- When `ccu::LoopGroup` is constructed, it immediately traverses the `loops` list and registers each loop to the group. After registration, the group members cannot be modified.
- The `ccu::Loop` objects in the `loops` list must have been constructed (that is, the body has been recorded) before the `ccu::LoopGroup` is constructed.
- The same `ccu::Loop` object should not be added to multiple `ccu::LoopGroup` objects.
- The body constraints of each loop in a group are the same as those of an independent `ccu::Loop` (see the constraints in [Loop](Loop.md)).
- `maxLoopNum` must be greater than 0. If it is 0, construction fails directly (`CCU_E_PARA`).
- `maxLoopNum` should be greater than or equal to the actual size of the `loops` list (that is, the number of loops that will actually be added to the group, including those reused by unrolling). If it is too small, adding loops later will fail due to insufficient LoopEngine pool capacity (`CCU_E_PARA`).
- When constructing `ccu::LoopGroup` in a var-based manner, ensure that the value of `parallelCfg` meets the runtime expectations. Bits [47:41] indicate the number of `ccu::Loop` objects contained in `loops` at construction time, and this value must equal the size of `loops`. Bits [54:48] indicate the loop ID to be automatically unrolled, and this ID must be less than or equal to the number of `ccu::Loop` objects. Bits [61:55] indicate the unrolling count. After unrolling all loops that need to be unrolled according to this value, the total number of loops obtained must be less than or equal to the value of `maxLoopNum`.

## Example

```cpp
using namespace AscendC::ccu;

// Scenario: Two loops share the LoopEngine resource pool.
CcuResult MyKernel(CcuKernelArg arg) {
    Variable r1, r2, numA, numB;
    numA = 10; numB = 20;

    Func body1([&] { r1 = numA + numB; });
    Func body2([&] { r2 = numA + numA; });

    CcuLoopConfig cfg1;
    cfg1.addrOffset = 0;
    cfg1.iterNum = 2;
    Loop l1(cfg1, body1);

    CcuLoopConfig cfg2;
    cfg2.addrOffset = 0;
    cfg2.iterNum = 3;
    Loop l2(cfg2, body2);

    // Organize l1 and l2 into a LoopGroup to share the LoopEngine resource pool.
    CcuLoopGroupConfig grpCfg;
    grpCfg.cloneNum = 0;
    grpCfg.cloneLoopOffset = 0;
    grpCfg.addrOffset = 0;
    grpCfg.ccuBufferOffset = 0;
    grpCfg.eventOffset = 0;
    LoopGroup g(grpCfg, /*maxLoopNum=*/2, {l1, l2});

    return CCU_SUCCESS;
}
```

When constructing a loop or loop group directly based on a variable, ensure that the value of the variable meets the instruction requirements.

```cpp
CcuResult CcuLoopAddDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    auto* args = static_cast<CcuLoopAddKernelArg*>(arg);

    Variable numA{}, numB{}, r1{}, r2{};
    numA = args->numA;
    numB = args->numB;

    // ========== LoopGroup variable-based ==========
    Variable varLoopParam1{}, varLoopParam2{}, varParallel{}, varOffset{};

    // loopParam: ctxId[52:45] | addrOffset[44:13] | iterNum[12:0]
    varLoopParam1 = 0x0000000002000003ULL; // addrOffset=4096, iterNum=3
    varLoopParam2 = 0x0000000002000004ULL; // addrOffset=4096, iterNum=4

    // repeatNum=2, repeatLoopIndex=1, totalLoopNum=2
    varParallel = 0x0101040000000000ULL;

    // addrOffset=4096, ccuBufferOffset=1, eventOffset=1
    varOffset = 0x0000000200000401ULL;

    Func body1([&]() {
        r1 = numA + numB;
    });
    Func body2([&]() {
        r2 = numA + numB;
    });

    Loop loop1(varLoopParam1, body1);
    Loop loop2(varLoopParam2, body2);

    LoopGroup group(varParallel, varOffset, /*maxLoopNum=*/4, {loop1, loop2});

    return CcuResult::CCU_SUCCESS;
}
```

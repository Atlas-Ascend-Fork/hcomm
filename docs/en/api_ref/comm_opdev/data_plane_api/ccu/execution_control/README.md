# Introduction

<!-- md-trans-meta sourceCommit=c5d21533f56c7d2f111d18f5605167320701dbbf translatedAt=2026-09-28T08:04:22.752Z pushedAt=2026-09-30T06:50:59.726Z -->

This section provides the APIs for expressing dynamic control flow within a CCU kernel, which are divided into three subclasses:

| Subclass | Applicable Scenario | API |
| --- | --- | --- |
| Software branch/loop | Flexible control flow, nested structures, and any CCU API can be used in the body. | [CCU_IF](CCU_IF.md), [CCU_ELSE](CCU_ELSE.md), [CCU_WHILE](CCU_WHILE.md), [CCU_DO](CCU_DO.md) |
| Hardware loop | A large number of iterations with the same structure, where the body contains local data movement operations, pursuing minimal instruction overhead. | [Loop](Loop.md), [LoopGroup](LoopGroup.md) |
| FuncBlock | The same logic is called multiple times within a kernel, saving SRAM. | [Func](Func.md), [CallFunc](CallFunc.md) |

Suggestions for selecting among the three subclasses:

- Complex control flow logic, nesting, and multiple CCU operations in the body → use the software branch/loop macros.
- A large number of iterations with the same structure and only local data movement operations in the body → use the hardware loop (`ccu::Loop`/`ccu::LoopGroup`).
- The same logic is called twice or more within a kernel → use FuncBlock (`ccu::Func`+`ccu::CallFunc`).

The three subclasses can be combined: nesting a hardware loop inside a software loop (`CCU_WHILE`/`CCU_IF`) is allowed. Do not nest in the reverse direction—software control flow (`CCU_IF`/`CCU_WHILE`/`CCU_DO`) should not be used inside a hardware loop body. The framework does not enforce validation of reverse nesting, but the behavior is undefined, so do not use it; only `CallFunc` inside a hardware loop body throws an exception carrying `CCU_E_INTERNAL` (after being uniformly caught at the kernel registration entry, the registration API returns `CCU_E_INTERNAL`).

## API List

- [CCU_IF](CCU_IF.md)
- [CCU_ELSE](CCU_ELSE.md)
- [CCU_WHILE](CCU_WHILE.md)
- [CCU_DO](CCU_DO.md)
- [Loop](Loop.md)
- [LoopGroup](LoopGroup.md)
- [Func](Func.md)
- [CallFunc](CallFunc.md)

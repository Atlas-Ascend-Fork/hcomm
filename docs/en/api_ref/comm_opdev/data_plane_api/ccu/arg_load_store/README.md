# Introduction

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T11:57:33.131Z pushedAt=2026-10-12T06:12:35.621Z -->

This section provides the APIs for passing scalar values between variables and external data sources in the CCU kernel, used to declare the source or destination of runtime data during the registration phase.

A variable is only a scalar placeholder during the registration phase and carries no value itself. The APIs in this section are responsible for exchanging data with the outside before/after kernel execution. External sources fall into the following two categories:

| Source/Destination | Address Determination Time | API |
| --- | --- | --- |
| `taskArgs[]` (injected by the host at each launch) | At runtime launch | [LoadArg](LoadArg.md) |
| On-chip memory immediate address (constant fixed at registration) | Registration phase | [Load](Load.md) (overload 1/2), [Store](Store.md) (overload 1/2) |
| On-chip memory indirect address (determined by Variable at runtime) | Runtime | [Load](Load.md) (overload 3/4), [Store](Store.md) (overload 3/4) |

Both `Load` and `Store` automatically select the immediate addressing or indirect addressing path based on the type of the first parameter (`uint64_t` or `Variable`). Each path internally supports two granularities: a single variable and a batch of `Array<Variable>`.

## API List

- [LoadArg](LoadArg.md)
- [Load](Load.md)
- [Store](Store.md)

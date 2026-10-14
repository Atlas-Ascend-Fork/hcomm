# Introduction

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T11:59:46.227Z pushedAt=2026-10-12T06:12:35.626Z -->

This section provides data movement APIs for moving bytes within a CCU kernel among the local on-chip memory, the local CcuBuffer, and the cross-rank remote on-chip memory, with an optional reduction operation performed during the movement.

All data movement APIs are asynchronous. When the hardware completes a movement, it automatically sets bit `mask` of `event` to 1, and the downstream waits for the completion signal through `EventWait`. Based on the data path, the APIs are divided into the following two categories:

| Type | Applicable Scenario | API |
| --- | --- | --- |
| Local operation | Copy and reduction between local on-chip memory ↔ local on-chip memory, or between local on-chip memory ↔ local CcuBuffer | [LocalCopy](LocalCopy.md), [LocalReduce](LocalReduce.md) |
| Cross-rank operation | Read and write data between local and remote on-chip memory through a channel, with optional reduction | [Read](Read.md), [ReadReduce](ReadReduce.md), [Write](Write.md), [WriteReduce](WriteReduce.md) |

Cross-rank operations require that all `ChannelHandle` belong to the same die, which is uniformly validated by the framework.

## API List

- [LocalCopy](LocalCopy.md)
- [LocalReduce](LocalReduce.md)
- [Read](Read.md)
- [ReadReduce](ReadReduce.md)
- [Write](Write.md)
- [WriteReduce](WriteReduce.md)

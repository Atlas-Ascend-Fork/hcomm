# CcuInsHandle

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T08:49:56.473Z pushedAt=2026-10-08T03:48:24.372Z -->

## Description

CCU instance handle, obtained from the HCCL communicator and used to identify a CCU instance. Subsequent kernel registration, translation, launch, and instance destruction operations are all performed through this handle.

## Prototype

```c
typedef uint64_t CcuInsHandle;
```

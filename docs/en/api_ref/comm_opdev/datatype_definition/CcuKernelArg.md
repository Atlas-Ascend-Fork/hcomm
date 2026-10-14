# CcuKernelArg

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T08:50:02.819Z pushedAt=2026-10-08T03:48:50.541Z -->

## Description

Parameter type of the **CCU kernel** function, used to pass user-defined parameters during **CCU kernel** registration. This type is a **void*** pointer, and the caller must manage the lifecycle of the parameter memory.

## Prototype

```c
typedef void *CcuKernelArg;
```

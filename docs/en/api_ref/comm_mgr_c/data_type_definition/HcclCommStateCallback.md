# HcclCommStateCallback

<!-- md-trans-meta sourceCommit=db421b11428a468be15280053798a93a1ac4f0bf translatedAt=2026-09-28T06:34:03.035Z pushedAt=2026-09-28T08:18:55.267Z -->

## Description

Defines the callback function type to be called at different phases of a communicator.

## Prototype

```c
typedef HcclResult (*HcclCommStateCallback)(HcclComm comm, HcclCommStatePhase state, void *args)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| comm | Input | HCCL communicator. |
| state | Input | Different phases of the communicator. For the definition of the HcclCommStatePhase type, see [HcclCommStatePhase](./HcclCommStatePhase.md). |
| args | Input | User context pointer passed to the callback function. |

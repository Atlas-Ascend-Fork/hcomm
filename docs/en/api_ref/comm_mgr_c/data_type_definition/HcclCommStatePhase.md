# HcclCommStatePhase

<!-- md-trans-meta sourceCommit=db421b11428a468be15280053798a93a1ac4f0bf translatedAt=2026-09-28T06:34:26.936Z pushedAt=2026-09-28T08:19:38.331Z -->

## Description

Different phases of a communicator.

## Prototype

```c
typedef enum {
    HCCL_COMM_STATE_PHASE_INVALID = -1,
    HCCL_COMM_STATE_PHASE_DESTROY_PRE = 0,   /* Before the communicator is destroyed by calling HcclCommDestroy. */
    HCCL_COMM_STATE_PHASE_DESTROY_POST = 1,  /* After the communicator is destroyed by calling HcclCommDestroy. */
    HCCL_COMM_STATE_PHASE_RESUME_PRE = 2,    /* Before the communicator resources are restored by calling HcclCommResume for step fast recovery. */
    HCCL_COMM_STATE_PHASE_RESUME_POST = 3    /* After the communicator resources are restored by calling HcclCommResume for step fast recovery. */
} HcclCommStatePhase;
```

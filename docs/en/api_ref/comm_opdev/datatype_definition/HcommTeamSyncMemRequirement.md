# HcommTeamSyncMemRequirement

<!-- md-trans-meta sourceCommit=d7ce44061ba2d7e75a2bf1c6bb466db58ee19c61 translatedAt=2026-09-28T09:01:59.619Z pushedAt=2026-10-08T06:25:44.121Z -->

## Description

**syncMem** requirement descriptor used to declare the required signal, counter, and barrier synchronization memory when creating a team.

## Prototype

```c
typedef struct {
    uint32_t signalCount;
    uint32_t counterCount;
    uint32_t barrierCount;
    uint32_t reserved[5];
} HcommTeamSyncMemRequirement;
```

## Field Description

| Field | Description |
| --- | --- |
| signalCount | Configuration is not supported yet; only **0** can be passed. |
| counterCount | Configuration is not supported yet; only **0** can be passed. |
| barrierCount | Number of barriers required, which must be greater than or equal to **1**. |
| reserved[5] | Reserved field. |

## Description

This structure is embedded in the **requirement** field of [HcclTeamCreateDesc](HcclTeamCreateDesc.md). When a team is created, the HCOMM layer calculates and outputs the number of **syncMem** bytes to be locally allocated based on this structure, using the formula: `(signalCount + counterCount + barrierCount) * sizeof(uint64_t) * memberNum`.

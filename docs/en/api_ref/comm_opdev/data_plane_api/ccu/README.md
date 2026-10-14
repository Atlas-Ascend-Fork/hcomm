# CCU API Introduction

<!-- md-trans-meta sourceCommit=bd478496298fb72920a1295ee6332b75edc75465 translatedAt=2026-09-28T07:48:13.082Z pushedAt=2026-09-30T09:25:02.707Z -->

This section provides the data-plane APIs within the Collective Communication Unit (CCU) kernel, which are used to describe communication and synchronization logic inside the operator kernel function body.

CCU adopts a two-phase model of "host registration + device execution": users call the APIs described in this section inside the kernel function body to describe the logic. During the registration phase, the framework first records these calls, then uniformly translates them into CCU device instructions when registration ends, and finally the CCU hardware executes them.

The APIs are classified by function as follows:

- [Resource Creation and Operation](./resource_allocation_operation/README.md)
- [Parameter Load/Store](./arg_load_store/README.md)
- [Data Movement](./data_movement/README.md)
- [Synchronization](./synchronization/README.md)
- [Flow Control](./execution_control/README.md)

# Concurrency Model

<!-- md-trans-meta sourceCommit=53fdf106ef330fac6b558d13103b73026d3055b6 translatedAt=2026-09-24T07:09:50.783Z pushedAt=2026-09-28T05:59:29.464Z -->

A communication operator consists of different communication tasks. If certain communication steps have no resource conflicts, they can be executed concurrently. HCCL provides a concurrency model in the operator programming model. The concurrency model requires defining concurrency units and the synchronization behavior between them, as shown in the following figure.

![](figures/concurrency_model.png)

- Concurrency unit: Provides a concurrency unit abstracted as a thread. Communication tasks are bound to threads, and operations across different threads can be executed concurrently.
- Synchronization between concurrency units:
  - A thread can contain multiple Notify instances. The number of Notify instances can be specified when creating a thread.
  - Within a communication entity, a thread can send a synchronization signal to another thread, and a thread can wait for a synchronization signal from another thread. For details about the APIs, see [Local Operations](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/README.md).

Under different communication engines, a thread may correspond to different concurrency entities and synchronization methods:

- AI CPU+TS and Host CPU+TS: The concurrency entity is a stream. Synchronization between streams is implemented using the NPU's Notify registers. A thread is an encapsulation abstraction of a stream and Notify registers.
- AIV: The concurrency entity is a Vector Core. Synchronization between AIV cores is implemented using device memory. A thread is an encapsulation abstraction of an AIV core and device memory.

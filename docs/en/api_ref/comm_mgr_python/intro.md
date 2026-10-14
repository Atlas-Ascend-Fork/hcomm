# Introduction

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-09-28T06:47:49.589Z pushedAt=2026-09-29T03:03:52.138Z -->

HCCL Python APIs are used to implement framework adaptation in graph mode. Currently, it is only used for distributed optimization of TensorFlow networks on the NPU.

## Related Concepts

| Concept | Description |
| --- | --- |
| Group | Refers to the process group that participates in collective communication, including:<br>  - **hccl_world_group**: The default global group, which contains all ranks participating in collective communication and is created through the **rank table** file.<br>  - Custom group: A subset of the process groups contained in **hccl_world_group**. You can use the **create_group** API to define the ranks in the rank table as different groups to execute collective communication algorithms in parallel. |
| Rank | Each communication entity in a group is called a rank. Each rank is assigned a unique identifier ranging from 0 to *n*-1 (where *n* is the number of NPUs). |
| Rank size | - **rank size**: The number of ranks in the entire group.<br>  - **local rank size**: The number of ranks of the processes in a group within the server where they reside. |
| Rank ID | - **rank id**: The rank identifier of a process in a group. Value range: 0 to (rank size - 1). For a custom group, ranks are renumbered starting from 0 within the group. For **hccl_world_group**, the rank ID is the same as the world rank ID.<br>  - **world rank id**: The rank identifier of a process in **hccl_world_group**. Value range: 0 to (rank size - 1).<br>  - **local rank id**: The rank number of the processes in a group within the server where they reside. Value range: 0 to (local rank size - 1). |

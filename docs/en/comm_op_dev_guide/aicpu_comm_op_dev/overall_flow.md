# Overall Process

<!-- md-trans-meta sourceCommit=53fdf106ef330fac6b558d13103b73026d3055b6 translatedAt=2026-09-24T07:04:54.351Z pushedAt=2026-09-28T05:59:29.455Z -->

The development and execution of HCCL AI CPU communication operators are divided into two phases: host-side preparation and AI CPU-side execution. The host side is responsible for parsing the topology, selecting algorithms, applying for resources, and dispatching AI CPU kernels. After the kernel starts, the AI CPU side orchestrates tasks based on the resource context. Therefore, ***kernel dispatch comes first, and task orchestration comes second***.

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontFamily": "Microsoft YaHei, Arial, sans-serif",
    "primaryColor": "#EAF2FF",
    "primaryBorderColor": "#4F7CAC",
    "primaryTextColor": "#1F2937",
    "actorBorder": "#4F7CAC",
    "actorBkg": "#F8FBFF",
    "activationBkgColor": "#DCEBFA",
    "activationBorderColor": "#4F7CAC",
    "sequenceNumberColor": "#6B7280",
    "noteBkgColor": "#FFF7D6",
    "noteBorderColor": "#D6A700"
  }
}}%%
sequenceDiagram
    title HCCL AI CPU communication operator execution flow

    participant HCCL_H as communication operator API (host)
    participant HCOMM_H as HCOMM(Host API)
    participant RTS as RTS
    participant HCCL_K as AICPU Kernel
    participant HCOMM_D as HCOMM(Device API)

    activate HCCL_H

    Note over HCCL_H: Prepare operator execution information on the host side.
    HCCL_H->>HCCL_H: Construct operator parameters (such as input/output addresses, data volume, and data type).
    HCCL_H->>+HCOMM_H: Query rank, topology level, and available links.
    HCOMM_H-->>-HCCL_H: Return communicator and topology information.

    opt The operator has multiple implementations.
        HCCL_H->>HCCL_H: Select an algorithm based on the operator type, execution engine, topology, and data volume.
    end

    Note over HCCL_H,HCOMM_H: Reuse resources at the granularity of operator and algorithm tags.
    HCCL_H->>+HCOMM_H: HcclEngineCtxGet(tag, engine)
    alt Resources already exist.
        HCOMM_H-->>HCCL_H: Return the device context.
    else First execution
        HCOMM_H-->>HCCL_H: The context does not exist.
        HCCL_H->>HCOMM_H: Create a context.
        HCCL_H->>HCOMM_H: Allocate and export host/AI CPU control threads.
        HCCL_H->>HCOMM_H: Allocate algorithm threads, Notify resources, channels, and communication memory.
        HCCL_H->>HCOMM_H: Serialize the resource context and copy it to the device.
        HCOMM_H-->>HCCL_H: Return the device context.
    end
    deactivate HCOMM_H

    Note over HCCL_H,HCCL_K: The host dispatches the kernel first, and task orchestration is executed after the kernel starts.
    HCCL_H->>+HCOMM_H: The host thread notifies the AI CPU control thread.
    HCOMM_H-->>-HCCL_H: HcclResult
    HCCL_H->>+RTS: aclrtLaunchKernelWithConfig dispatches the AI CPU kernel.
    RTS-->>HCCL_H: aclResult
    RTS->>HCCL_K: Start the kernel.
    deactivate RTS
    HCCL_H->>+HCOMM_H: The host thread waits for the AI CPU completion notification.

    activate HCCL_K
    HCCL_K->>HCCL_K: Deserialize the resource context.
    HCCL_K->>+HCOMM_D: HcommBatchModeStart(tag)
    HCOMM_D-->>-HCCL_K: HcclResult
    HCCL_K->>+HCOMM_D: The AI CPU control thread waits for the host startup notification.
    HCOMM_D-->>-HCCL_K: HcclResult

    Note over HCCL_K,HCOMM_D: Execute algorithm task orchestration within the kernel.
    HCCL_K->>HCCL_K: ExecOp(param, resCtx)
    HCCL_K->>+HCOMM_D: Orchestrate thread synchronization, channel synchronization, and data transfer tasks.
    HCOMM_D-->>-HCCL_K: HcclResult

    HCCL_K->>+HCOMM_D: The AI CPU control thread notifies the host thread.
    HCOMM_D-->>-HCCL_K: HcclResult
    HCCL_K->>+HCOMM_D: HcommBatchModeEnd(tag)
    HCOMM_D-->>-HCCL_K: HcclResult
    deactivate HCCL_K

    HCOMM_H-->>-HCCL_H: The AI CPU execution is completed.
    deactivate HCCL_H
```

The main responsibilities of each phase are as follows:

1. ***Defining the operator API***: Specify the input and output, data size, data type, communicator, execution stream, and other information.
2. ***Querying topology information***: Obtain the number of ranks, topology levels, intra-level connections, and available links to provide a basis for algorithm selection and resource computation.
3. ***Selecting algorithms***: Select a registered algorithm implementation based on conditions such as the operator type, execution engine, topology, data size, and data type. This step can be omitted for custom operators that have only one fixed implementation.
4. ***Creating resources***: Compute and allocate threads, Notify resources, channels, communication memory, and resource context, and copy the context required for AI CPU execution to the device.
5. ***Dispatching kernels***: The host establishes a startup synchronization relationship with the AI CPU control thread, and then dispatches the AI CPU kernel to the execution stream.
6. ***Orchestrating tasks***: After the AI CPU kernel starts and obtains the resource context, it calls the algorithm execution logic to orchestrate operations such as thread synchronization, channel synchronization, and data transfer onto the corresponding threads.
7. ***Completing synchronization***: After the AI CPU side completes orchestration, it notifies the host. The host side waits for this notification to ensure the execution order between the communication task and the business stream.

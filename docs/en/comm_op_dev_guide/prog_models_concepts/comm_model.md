# Communication Model

<!-- md-trans-meta sourceCommit=643c963e5b9fdd06457003472086752ffe3515a7 translatedAt=2026-09-24T07:09:44.462Z pushedAt=2026-09-28T05:59:29.464Z -->

**Figure 1**  HCCL communication model  
![](figures/hccl_communication_model.png "HCCL-communication-model")

The preceding figure describes the HCCL communication model, where all elements are software concepts. The following explains each concept:

- Communication memory: A block of memory on a communication member can be registered with a communicator, indicating that this memory can be accessed by other communication members within the communicator.
- Endpoint: Represents the port used for communication with other communication objects (such as the NetDevice of a NIC).
  - An endpoint contains attributes such as address and protocol. An endpoint may contain multiple physical ports (for example, in bonding scenarios).
  - Each communication object can contain multiple endpoints.

- Channel: A communication channel established between a specific endpoint of one communication object and a specific endpoint of another communication object.
  - Multiple channels can be established between a pair of endpoints.
  - When a channel is created, the local end and the remote end need to synchronously call the channel creation API.
  - When a channel is created, the memory information registered by the local end is exchanged with the memory information registered by the remote end. The control plane also provides an API for querying the remote memory information (address and size) based on the channel.

Communication models are further classified into network semantics communication models and memory semantics communication models:

- Network semantics communication model: The communication operator developer accesses the remote communication object memory based on the channel, or performs synchronization with the remote communication object.
- Memory semantics communication model: The communication memory of the remote object can be mapped to the local memory address space, and the communication operator developer can directly use local memory copy operations to access the remote communication memory.

## Network Semantics Communication Model

In the network semantics communication model, users use channels to read and write remote communication object memory or synchronize with remote communication objects. For details, see the API description in [Communication Operations](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/README.md).

**Figure 2**  Network semantics communication model  
![](figures/semantic_communication.png "Network-semantics-communication-model")

The key object of the network semantics communication model is the channel. Developers can create multiple channels between a pair of endpoints. The following opens the channel model and describes its internal elements in detail.

The following figure shows the channel model under the RoCE protocol.

**Figure 3** Channel model in RoCE scenarios  
![](figures/roce_channel_model.png "channel-model-in-RoCE-scenarios")

- A channel serves as the entry point for communication between the local and remote objects. The channel of the local communication object has a one-to-one association with the channel of the remote communication object.
- A channel is associated with one or more Queue Pair (QP) instances. The QP instances associated with the local channel correspond to those associated with the remote channel. When a channel is established, the corresponding QPs are linked.
- A channel contains multiple Notify instances, which are used for synchronization operations between communication objects. Notify is an abstract concept for synchronization operations and may be implemented by different entities under different communication engines.
  - The number of Notify instances contained in a channel can be specified when the channel is created.
  - The local end can send a synchronization signal to a specific Notify (specified by a sequence number) in the channel corresponding to the remote object through a channel.
  - The local end can wait for a synchronization signal from the remote communication object based on a specific Notify of a channel, and subsequent operations can be performed only after the synchronization signal is received.

## Memory Semantics Communication Model

In the memory semantics communication model, the communication memory of a remote object can be mapped to the local process address space. Communication operator developers can use local operation APIs to implement data movement or synchronization between nodes. For details, see [Local Operations](../../api_ref/comm_opdev/data_plane_api/cpu-cpu_ts-aicpu_ts/README.md).

The following figure shows the memory semantics communication model of HCCL.

**Figure 4** Memory semantics communication model  
![](figures/memory_semantic_model.png "memory-semantics-communication-model")

- Endpoint: indicates the network logical port used for communication with other communication objects.
  - Under the memory semantics communication model, the endpoint used for channel creation is used for control plane link establishment. Data plane communication does not necessarily use this endpoint. The network port used for data plane communication is determined by the memory mapping mechanism.
  - An endpoint contains attributes such as address and protocol. One endpoint may contain multiple physical ports (for example, in bonding scenarios).

- Channel: A communication channel established between the local communication object and the remote communication object. In the memory semantics scenario, the establishment of a channel indicates that the memory mapping functionality between two communication objects is enabled, and it is not used for communication between the communication objects.
  - When a channel is created, the memory information registered on the local end is exchanged with the memory information registered on the remote end, and memory mapping is performed.
  - The control plane provides the capability to query the address and size of remote memory after it is mapped to the local end, based on the channel.

## CCU Communication Model

The CCU communication model is similar to the network semantics communication model, except that it supports writing local CcuBuffer data to the communication peer's memory, or reading the communication peer's memory to the local CcuBuffer.

For the CcuBuffer concept, see [Resource Abstraction](./CCU_models_concepts.md#resource-abstraction).

**Figure 5**  CCU communication model  
![](figures/CCU_communication_model.png "CCU-communication-model")

- A channel can contain multiple Notify instances (corresponding to CCU synchronization registers), which are used for synchronization operations between communication objects:
  - The local end can send a synchronization signal to a specific Notify (specified by sequence number) of the peer end through the channel.
  - The local end can wait for a synchronization signal from the peer end based on a specific Notify of the channel, and returns only after receiving the synchronization signal.

- A channel can contain multiple Variable instances (corresponding to CCU general-purpose registers) for synchronizing data in Variables with the peer end.

  **Figure 6** Channel model
  ![](figures/ccu_channel.png)

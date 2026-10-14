# Programming Models and Concepts

<!-- md-trans-meta sourceCommit=997c68163541f267c1a61bc5a8753ea8c7310bab translatedAt=2026-09-24T07:08:42.285Z pushedAt=2026-09-28T05:59:29.462Z -->

This chapter describes the core programming models and key concepts in HCCL communication operator development, helping developers understand the communication engine, communication model, concurrency model, topology model, and CCU programming model.

- [Communication Engine](comm_engine.md): Introduces the applicable scenarios and task execution processes of the four communication engines supported by HCCL: AI CPU+TS, Host CPU+TS, AIV, and CCU.
- [Communication Model](comm_model.md): Describes the core concepts of the HCCL communication model, such as communication memory, endpoint, and channel, as well as the three communication models: network semantics, memory semantics, and CCU.
- [Concurrency Model](concurrency_model.md): Introduces the programming model that uses threads as the concurrency units and implements concurrent execution and synchronization of communication tasks through the Notify mechanism.
- [Topology Model](topology_model.md): Introduces how HCCL models the connection relationships between ranks in a communicator, including Node, Endpoint, Edge, Link, and hierarchical topology concept.
- [CCU Programming Models and Concepts](CCU_models_concepts.md): Introduces the architecture, resource abstraction, data movement and compute capabilities, concurrency model, synchronization mechanism, and flow control of the CCU collective communication acceleration unit.

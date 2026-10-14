# Topology Model

<!-- md-trans-meta sourceCommit=4b3acc1183ff175f340b0421dbe1faf9a723a585 translatedAt=2026-08-11T07:20:01.735Z pushedAt=2026-08-20T11:39:14.579Z -->

## Background

When implementing communication operators, topology query APIs must be provided on the control plane for the following reasons:

- The operator control plane implementation needs to create channels for the data plane. Whether different ranks are interconnected and through which endpoints they are interconnected is essential information for creating channels. Therefore, topology query control plane APIs must be provided.
- Different clusters may have different connection relationships, and the performance of operator implementation is strongly correlated with topology connections. To enable operators to adapt to different topology forms while maintaining good performance, it is necessary to perceive topology connection relationships.

Therefore, HCCL models the connection relationships between different ranks within a communicator as a topology, and the resulting topology graph is called a rank graph. It also provides control plane APIs for querying connection relationships. For details, see [Querying Topology Information](../../api_ref/comm_opdev/control_plane_api/topo_info_query/README.md).

## Topology Model Introduction

HCCL uses a traditional graph representation with nodes and edges to model the topology. Since large-scale AI clusters are typically built hierarchically (for example, a server contains multiple interconnected NPUs and multiple servers form a rack or SuperPoD), HCCL adds a topology hierarchy abstraction on top of the graph representation.

The following figure shows an example of a topology model, which is used to introduce the concepts in the topology model.

![](figures/topo_model.png "topology model")

- Node: A node in the graph, which includes two types:
    1. Communication entity: an entity identified by a rank ID in the communicator.
    2. Fabric: an abstraction of network switching/routing.
        1. A Fabric can only connect to communication entities.
        2. A Fabric can be a single switch or a network facility composed of multiple switches.
        3. A network facility abstracted as a Fabric node must meet the following condition: all communication entities connected to it can communicate with each other through it.

- Endpoint: A logical port of a node. A node can contain one or more endpoints.
- Edge: An edge in a graph, representing the connection relationship between different nodes. The two ends of an edge are the endpoints of two nodes.
- Link: Information indicating that a link can be established between two communication entities, including the endpoints at both ends.
- Topology hierarchy: Actual network topologies are hierarchical. As shown in the following figure, the topology is divided into two layers: layer 0 and layer 1, with two layer-0 topologies inside. Each layer of the network topology has its own topology type, such as Fullmesh and Clos.

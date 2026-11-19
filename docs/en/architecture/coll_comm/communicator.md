# Communicator

The communicator module manages and coordinates collective communication operations. In HCOMM, each `HcclComm` corresponds to a communication domain context. Internally, `CollComm` holds the RankGraph, local rank information, communication engines, channels, memory, and other resources.

> [!NOTE] Note
> The static communication domain scale-in and scale-out APIs are experimental features. They may be changed, renamed, or removed without notice in later versions, and backward compatibility is not guaranteed.

## Responsibilities

- Communication context management
- Communication domain creation, destruction, and configuration management
- Collective operation creation and scheduling
- Topology information maintenance
- Static communication domain scale-in and scale-out

## Communication Domain Lifecycle

A communication domain is typically used as follows:

1. Create the initial communication domain by using rootInfo or a rankTable.
2. Create communication resources on the initial communication domain and perform collective communication.
3. When the membership needs to change, perform a static scale-in or scale-out based on the current communication domain to obtain a new communication domain handle.
4. Stop using the old communication domain, and destroy it after confirming that no operations are in flight.
5. Destroy the current communication domain when the service ends.

## Core Interfaces

### Scale-in and Scale-out Interfaces

This section lists only the communication domain scale-in and scale-out interfaces. For basic interfaces such as communication domain creation, destruction, and rank query, see [Communicator Creation and Management APIs (C Language)](../../api_ref/comm_mgr_c/README.md).

| Category | Interface | Description |
| --- | --- | --- |
| Scale-in | [HcclCommShrink](../../api_ref/comm_mgr_c/HcclCommShrink.md) | Excludes the specified ranks and creates a new, smaller communication domain. |
| Scale-out | [HcclCommGrow](../../api_ref/comm_mgr_c/HcclCommGrow.md), [HcclCommGetRootInfo](../../api_ref/comm_mgr_c/HcclCommGetRootInfo.md) | Obtains the rootInfo for scale-out and adds new ranks to create a larger communication domain. |

## Static Scale-in and Scale-out

### Scale-in

During scale-in, all surviving ranks call [HcclCommShrink](../../api_ref/comm_mgr_c/HcclCommShrink.md) to exclude the specified ranks from the current communication domain. The rank IDs in the new communication domain are renumbered contiguously in ascending order of the original rank IDs.

Scale-in does not add new ranks. It usually creates a smaller communication domain based on the existing topology and communication domain information. If the excluded ranks include the root, the new communication domain has no valid root after scale-in. Before a subsequent scale-out, call [HcclCommGetRootInfo](../../api_ref/comm_mgr_c/HcclCommGetRootInfo.md) to trigger root reselection.

### Scale-out

During scale-out, all surviving ranks and new ranks call [HcclCommGrow](../../api_ref/comm_mgr_c/HcclCommGrow.md) together. Surviving ranks retain their original rank IDs, and new ranks are assigned contiguous IDs starting from the current rank count.

Before scale-out, one surviving rank calls [HcclCommGetRootInfo](../../api_ref/comm_mgr_c/HcclCommGetRootInfo.md) to obtain `rootInfo` and distributes it to all participants through the service control plane. Scale-out requires topology information collection and new communication link establishment. Therefore, all participants must complete the call in the same scale-out round.

### Derived Communication Domain

A communication domain produced by scale-in or scale-out is a derived communication domain of the current communication domain:

- The derived communication domain has a new communication domain handle and independent communication resources.
- The regular resources of the parent communication domain are not modified in place.
- Only the tail communication domain in the derivation chain can continue to perform scale-in or scale-out.
- The persistent topology detection resources are held by the tail derived communication domain for subsequent scale-in or scale-out operations.

### Resource Ownership

A new communication domain created by scale-in or scale-out has independent resources. Resources of the old communication domain are not automatically migrated:

- Symmetric memory windows must be registered again on the new communication domain, and remote addresses must be obtained again as required.
- CCU instances, CCU channels, and other CCU resources must be queried or bound again based on the new communication domain.
- CCL buffers, channels, threads, engine contexts, and other resources must be obtained or created again on the new communication domain.
- The old communication domain must be explicitly destroyed by the service by calling [HcclCommDestroy](../../api_ref/comm_mgr_c/HcclCommDestroy.md).

## Data Flow

## Code Path

- Communication domain implementation: `src/coll_communicator_mgr/communicator/coll_comm.cc`
- C interface adaptation: `src/coll_communicator_mgr/api_c_adpt/coll_comm_c_adpt.cc`
- Topology detection and derivation: `src/coll_communicator_mgr/rank_info_detect/`
- Local resources and communication engines: `src/coll_communicator_mgr/resource_mgr/`

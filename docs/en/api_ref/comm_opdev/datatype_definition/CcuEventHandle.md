# CcuEventHandle

<!-- md-trans-meta sourceCommit=ea28a2a84bd47924bd51751e37d9926552e1ad78 translatedAt=2026-09-28T08:49:40.311Z pushedAt=2026-10-08T03:48:01.799Z -->

## Description

Handle type related to CCU Event (`uint64_t`). The same type has two usages in different scenarios, which cannot be mixed:

| Purpose | Source | Usage |
| --- | --- | --- |
| Reserved handle | [HcommCcuEventAlloc](../control_plane_api/ccu_resource_mgmt/HcommCcuEventAlloc.md) on the host side | Passed to [HcommCcuEventGetAddr](../control_plane_api/ccu_resource_mgmt/HcommCcuEventGetAddr.md), or passed to the reserved handle construction form of [Event](../data_plane_api/ccu/resource_allocation_operation/Event.md) / [Array\<Event\>](../data_plane_api/ccu/resource_allocation_operation/Array.md) in the kernel |
| Virtual handle in the kernel | Resource creation APIs such as the default construction of `Event` during the kernel registration phase | Used for data plane synchronization APIs (such as `EventRecord` and `EventWait`) of the same kernel. It becomes invalid after kernel instruction translation is complete. |

The reserved handle cannot be used as `Event::handle`, nor can `Event::handle` be passed to `HcommCcuEventGetAddr`.

## Prototype

```c
typedef uint64_t CcuEventHandle;
```

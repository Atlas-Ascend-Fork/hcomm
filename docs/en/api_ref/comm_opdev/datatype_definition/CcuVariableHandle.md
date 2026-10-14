# CcuVariableHandle

<!-- md-trans-meta sourceCommit=ea28a2a84bd47924bd51751e37d9926552e1ad78 translatedAt=2026-09-28T08:51:16.407Z pushedAt=2026-10-08T03:54:44.742Z -->

## Description

Handle type related to CCU Variable (`uint64_t`). The same type has two usages in different scenarios, which must not be mixed:

| Purpose | Source | Usage |
| --- | --- | --- |
| Reserved handle | [HcommCcuVariableAlloc](../control_plane_api/ccu_resource_mgmt/HcommCcuVariableAlloc.md) on the host side | Passed to [HcommCcuVariableGetAddr](../control_plane_api/ccu_resource_mgmt/HcommCcuVariableGetAddr.md), or passed to the reserved handle constructor form of [Variable](../data_plane_api/ccu/resource_allocation_operation/Variable.md) / [Array\<Variable\>](../data_plane_api/ccu/resource_allocation_operation/Array.md) in the kernel |
| Virtual handle in kernel | Resource creation APIs such as the default constructor of `Variable` during the kernel registration phase | Used for the data plane APIs of the same kernel. It becomes invalid after kernel instruction translation is complete. |

The reserved handle must not be used as `Variable::handle`, and `Variable::handle` must not be passed to `HcommCcuVariableGetAddr`.

## Prototype

```c
typedef uint64_t CcuVariableHandle;
```

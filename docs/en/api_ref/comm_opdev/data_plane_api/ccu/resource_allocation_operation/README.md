# Introduction

<!-- md-trans-meta sourceCommit=9a28f2fdc3bd8008a5fa6d178f4a7efe8b2d3eb9 translatedAt=2026-10-08T12:05:49.824Z pushedAt=2026-10-12T06:12:35.633Z -->

This section provides the APIs for allocating and binding virtual resource handles within the CCU kernel, as well as the APIs for assignment and arithmetic operations on `Variable`/`Address`.

CCU resource allocation follows a "virtual-first, physical-later" two-stage model: during the kernel registration stage (inside the kernel function body), calling the default constructors such as `Variable()`/`Address()`/`Event()` only produces virtual handles (consuming no hardware resources and always succeeding), while the actual physical resource allocation is completed during the `HcommCcuKernelRegister` stage (after the kernel function finishes executing). All default-constructed C++ wrapper classes follow the semantics of "virtual allocation on construction, no release on destruction"; physical resources are managed uniformly over the CCU instance lifecycle and are not released by C++ destructors.

The APIs are classified by resource type as follows:

| Resource Type | Unit Allocation | Batch Allocation | Channel Reference | Binding Host-Side Reserved Resources |
| --- | --- | --- | --- | --- |
| Scalar register | [Variable](Variable.md) | [Array\<Variable\>](Array.md) | [GetResByChannel](GetResByChannel.md) | [`Variable(varHandle, index)`](Variable.md), [`Array<Variable>(acqHandle, count)`](Array.md) |
| Address register | [Address](Address.md) | — | — | — |
| Completion event unit | [Event](Event.md) | [Array\<Event\>](Array.md) | — | [`Event(acqHandle, index)`](Event.md), [`Array<Event>(acqHandle, count)`](Array.md) |
| On-chip CcuBuffer (4 KB) | [CcuBuffer](CcuBuffer.md) | [Array\<CcuBuffer\>](Array.md) | — | — |
| Local on-chip memory composite address | [LocalAddr](LocalAddr.md) | — | — | — |
| Remote on-chip memory composite address | [RemoteAddr](RemoteAddr.md) | — | — | — |

In the table, **Channel Reference** and **Binding Host-Side Reserved Resources** do not follow the virtual-first, physical-later model: the former binds the slot reserved when the channel link is established, while the latter binds the resources reserved on the host side by [HcommCcuVariableAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuVariableAlloc.md) and [HcommCcuEventAlloc](../../../control_plane_api/ccu_resource_mgmt/HcommCcuEventAlloc.md). Neither allocates new resources; they throw an exception when the parameters are invalid, unlike the default constructors which always succeed.

In addition to resource allocation, [Variable](Variable.md) and [Address](Address.md) also provide assignment and arithmetic operators; these operators describe operations executed on the device side, operating on the corresponding registers during hardware execution, rather than being computed immediately on the host side.

## API List

- [Variable](Variable.md)
- [Address](Address.md)
- [Event](Event.md)
- [CcuBuffer](CcuBuffer.md)
- [LocalAddr](LocalAddr.md)
- [RemoteAddr](RemoteAddr.md)
- [Array](Array.md)
- [GetResByChannel](GetResByChannel.md)

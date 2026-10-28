# base_comm Module Overview

## Functional Description

base_comm is the basic communication layer (L3, bottom layer) of the hcomm repository: resource
management + communication primitive execution. Per the layering rules in the repo-root AGENTS.md,
base_comm must not reversely depend on coll_communicator_mgr / coll_comm_ops.

Core capabilities:

- Communication resource management: endpoint, channel, socket, registered memory, CCU resources
  (resources/);
- Communication primitive execution: host-side C adaptation interfaces and AICPU-side primitives
  (primitives/);
- Configuration and infrastructure: config management (config_mgr/), DFX exception reporting (dfx/),
  dynamic library symbol loading and other common components (common/);
- Resource tree management: process-level and device-level resources are consolidated into the
  resource tree rooted at BaseCommMgr, fixing probabilistic coredumps caused by unordered static
  destruction at process exit — see "Resource Tree Management"; read that section before changing
  the tree shape, adding/removing nodes, or reordering member declarations.

## Directory Description

```text
base_comm/
├── common/                          Common components (TpMgr / EidInfoMgr / HccpTlvHdcMgr / orion
│                                      adaptation / dynamic library symbol loading)
├── config_mgr/                      Configuration management (ConfigMgr / CommConfiger adaptation)
├── dfx/                             DFX (exception handling and reporting)
├── primitives/                      Communication primitives (api_c_adpt C adaptation / aicpu / dfx)
├── resources/                       Communication resources (ccu / comm_engine_res / endpoint_pairs /
│                                      endpoints / flush / hccp / reged_mems / southbound_adpt)
├── base_comm_mgr.{h,cc}             Resource tree root BaseCommMgr
├── base_comm_node_accessors.{h,cc}  Narrow tree accessors (only entry for call sites)
├── base_comm_res_per_dev.{h,cc}     Device subtree (one per devPhyId)
├── handle_registry_procs.{h,cc}     Channel / Thread handle registry node
├── legacy950_res_mgr_proc.{h,cc}    Legacy singleton anchor node
└── README.md / README_en.md         This document
```

(File relationship table and component interaction diagram to be added)

## Process Description

- Creation: process-level nodes are built in one pass during the first
  `BaseCommMgr::GetInstance()` (R6); device subtrees are lazily created per devPhyId on
  first access and returned as references through narrow accessors (R7/R9).
- Destruction: an explicit `DeInit(ExitProfile)` runs the 9-step action chain (R4,
  observably verified by the exit_path ST); destructors free memory only, and
  `~BaseCommMgr` replays the DeInit chain as a fallback.

## Data Description

- Layering: process level = one per process or a key without a device dimension; device
  level = one instance per devPhyId (see "Shape of the Tree" for the rule).
- Device-subtree storage: `BaseCommResPerDevMgr` keeps an active table, a retired area,
  and a reserve fallback (R9); steady-state reads are lock-free, first-build takes the lock.

## Interface Description

The unified entry points exposed to upper layers are the narrow accessors
(`base_comm_node_accessors.h`; usage rules in R7), grouped by key type:

- Process level, keyless: `GetConfigMgr` / `GetCommConfiger` / `GetDlUrmaFunction` /
  `GetAicpuCommEngineBinProc` / `GetEndpointHandleMgr` / `GetSharedJettyMgr` /
  `GetProcRegedMemMgrCache` / `GetEndpointProcMgr` / `GetServerSocketManager` /
  `GetChannelHandleRegistryProc` / `GetThreadHandleRegistryProc` / `GetDpuNotifyManager` /
  `GetProcRegedMemMgrCacheHolder` (shared_ptr non-owning alias, see R7);
- Device level, phyId-keyed: `GetEidInfoMgr` / `GetEndpointCtxMgr` / `GetSocketMgr` /
  `GetGlobalNetDevMgr` / `GetTpMgr` / `GetHccpTlvHdcMgr`;
- Device level, logicId-keyed: `GetCcuKernelMgr` / `GetCcuInstanceMgr` /
  `GetCcuVarEventResMgr` / `GetSocketProcess`.

See R9 for out-of-range/failure fallback semantics; the mixed key types of device-level
accessors are historical and must not be interchanged.

## Usage Restrictions

- No re-entry via narrow accessors during construction (deadlock red line, R2);
- Business code must use narrow accessors only — no direct tree-access chains and no new
  file-scope globals (R1/R7);
- Under `EXIT_PROFILE_RESET`, all device-touching actions are skipped; device-side returns
  of leftover endpoint/channel tables fall back to the destruction chain and are not
  guaranteed (known limitations in R5);
- Out-of-range fallbacks are baseline-compatible behavior with three coexisting semantics
  (R9); new accessors must register their semantics first;
- Before adopting a singleton into the tree or rerouting call sites, reverse-audit whether
  its destruction chain touches the tree (R10).

## Resource Tree Management

All resources in this directory are managed by a single **process-level resource tree**.
Before changing the tree structure, adding/removing nodes, or reordering member
declarations, read this section in full. Changes that violate these rules must not be merged.

Design principles: single root (R1) / no re-entry during construction (R2) / ownership
is lifetime (R4) / layered granularity (tree shape) / declaration order = destruction
contract (R3) / minimal intrusion / binary boundary (R8) / evolution-ready (placeholder
shells reserve slots; phase 2 fills content without reshaping the tree).

### Shape of the Tree

```text
BaseCommMgr (process-level root, the only newly-added singleton in base_comm;
pre-existing function-static factories are listed in R1 exceptions, anchored per R8)
├── [Infrastructure] isDeInited_ / lastDeinitWasReset_                   ← declared first = dies last
├── Legacy950ResMgrProc    legacy950 singleton anchor node (libhccl_v2.so, anchored only, not in tree)
├── ConfigMgrProc          configuration (ConfigMgr value member; CommConfiger forwards)
├── DlsymProc              dynamic library symbol table (DlUrmaFunction)
├── DfxProc                exception reporting / profiling
├── EndpointProcMgr        network base: process-level netdev/connection tables + ServerSocketManager
├── NotifyMgrProc          notify management (DpuNotifyManager; process-level: process-wide ID space)
├── CommEngineBinMgrProc   process-level bin family (Aicpu entity + Aiv/Dpu placeholders)
├── BaseCommResPerDevMgr   per-device subtree store (active table + retired area + fallback instance)
│     └── BaseCommResPerDev (one tree per physical device, lazily built on first access)
│           ├── NetDevProc          per-device netdev instance
│           ├── ServerSocketMgrProc SocketMgr
│           ├── SocketProcess       socket cleanup (destructor returns sockets via GetSocketMgr)
│           ├── CommEngineMgrProc   engine family: ThreadMgr / MemMgr / HdcProc / CcuInstanceProc
│           └── EndpointCtxMgr      EndpointCtx dedup cache
├── HandleRegistryMgrProc  Channel / Thread handle registries (residual channel destruction touches
│                            device subtrees; dies first)
└── EndpointMgr            endpoint family (handle table + SharedJetty + RegedMem cache + EidInfo) ← dies first
```

Level classification: **process-level** = one instance per process, or its key has no device
dimension; **device-level** = one instance per devPhyId. Process-level resources with no
device dimension (e.g. bin handles) must not be placed in a device subtree.

### Core Rules

#### R1 Single root; no new global variables

No new Meyers singletons / function-local static objects / file-scope global variables
may be added under base_comm. Resources that need global semantics become value members
of a node at the appropriate level of the tree, exposed through a narrow accessor (R7).

Exceptions (require review and registration in this README):

- Legacy singletons that cannot be moved across binaries (i.e. not compiled into
  hcomm_base_obj: `Hccl::` in libhccl_v2.so / `hccl::CommConfiger`) — anchored only,
  not in tree (R8);
- ACL callback registration flags (`g_deviceRefreshCallbackRegistered`, etc.) — their
  lifecycle is governed by ACL, not by the tree;
- Reserve fallback static instances (R9);
- Base_comm-owned pre-existing singletons (open CCU family: `CcuPfeCfgMgr`,
  `hcomm::CcuComponent`, `hcomm::CcuResBatchAllocator`, `hcomm::CcuResSpecifications`;
  keyed by userDevId, which does not match the per-device subtree's phyId key / unused
  in current mixed mode) — anchored only for now (R8), tree adoption to be evaluated
  in phase 2;
- AICPU-only components (`ExceptionHandle` / `ExceptionCallbackMgr`, compiled only with
  ccl_kernel) — the host process has no such objects; not in tree, not anchored;
- Legacy platform-base adaptation (`hrtTryGetDeviceIndexByPhyId` /
  `hrtTryGetDevicePhyIdByIndex`, src/legacy/ascend910) — quiet variants of the
  phyId<->logicId probing conversions required by anchoring (R8) and by the narrow
  accessors' expected fallback paths; thin wrappers of the pre-existing static
  `GetDeviceIndexByPhyId(...)` / `GetDevicePhyIdByIndex(...)` with no new
  implementation logic.
  Conversion failure on VM / missing-card setups is an expected path, and the
  non-quiet variant logs ERROR inside legacy before returning, which an outer
  wrapper cannot suppress — so the interface has to be added as a quiet variant;
  symbols follow the existing .map export mechanism. Registered exception to the
  repo-root AGENTS.md rule "no new interfaces under src/legacy/"; to be reclaimed
  in phase 2 once legacy provides a formal quiet variant;
- Host-side pre-existing statics (all compiled into hcomm_base_obj; their destructors
  never go through narrow accessors and never touch the tree, so no R10 UB path —
  kept as-is, not anchored):
  - `FlushManager` (resources/flush/flush_manager.cc) — first touched by the upper
    layer at coll_comm.cc; its destructor DestroyAll tears down QP/CQ/MR (pre-existing
    master behavior), and its dependency `RdmaHandleManager` is anchored (R8);
    phase 2 will evaluate promoting it to a process-level node under the tree root;
  - `Hccl::DfxDlProfFunction` (dfx/dfx_dlprof_function.cc) — destructor only dlclose;
  - `EnvPlfDebugConfig` (common/debug/plf_debug_config.cc) — env parsing plus an
    atomic value; trivial destructor;
  - NicPlugin registry cluster (`LoadOnce` / `LoadedPlugins` / `ProtocolPlugins` in
    primitives/api_c_adpt/nic_plugin/nic_plugin_manager.cc) — destruction unloads
    plugins (dlclose chain);
  - `AicpuTsRoceEndpoint::NetDevMapMutex/GetNetDevMap` (resources/endpoints/
    aicpu_ts_roce_endpoint.cc) — host-side netdev reference table;
  - `AicpuTsRoceServerSocketContext::ListenSocketMapMutex/GetServerSocketMap`
    (resources/endpoints/server_socket_context/aicpu_ts_roce_server_socket_context.cc)
    — host-side listen-socket table;
- Tree-teardown flag (function-local static atomic in base_comm_mgr.cc, queried via
  `hcomm::IsResourceTreeTeardownFinished()`) — trivially destructible and safe to read
  after the tree dies; lets anchored singletons (which outlive the tree) skip
  tree-touching releases in their destructor chains (R10).

#### R2 No re-entry via narrow accessors during construction (deadlock red line)

First-access lazy build of a device subtree runs entirely under `BaseCommResPerDevMgr`'s internal lock (a
non-recursive mutex) inside `GetBaseCommResPerDev()`, so node constructors and
`AnchorDeviceSingletons()` execute with the lock held. Calling any device-level narrow
accessor (`hcomm::GetSocketMgr`, etc.) in this window re-locks the same mutex on the same
thread — a deadlock that only fires on the first-access build path and is very hard to
diagnose.

- Anchoring targets legacy singletons only: call their `GetInstance()` directly, never
  through narrow accessors;
- Tree value members are built with the constructor — they need no "anchoring" and must
  not be anchored; reference sibling members directly during construction;
- No constructor path may call narrow accessors or anything that re-enters
  `BaseCommMgr::GetInstance()` / `GetBaseCommResPerDev()` (half-constructed recursion /
  re-entrant-lock deadlock).

#### R3 Member declaration order = destruction contract (declared first, dies last)

C++ members are destroyed in reverse declaration order. The member declaration order
inside every tree node is the destruction-order contract:

- Before declaring a member, answer: "who calls back into / returns resources to me
  during destruction?" The depended-upon member must be declared first (dies last).
- Infrastructure (mutexes, DeInit flags) is declared before all its users (dies last) —
  destruction-time callback chains take those locks.
- **Changing any declaration order = changing the destruction contract; it requires a
  separate review** and a recorded rationale in this README.

Process-level contract (dies first → dies last):
`EndpointMgr → HandleRegistryMgrProc → device subtrees → CommEngineBinMgrProc → NotifyMgrProc → EndpointProcMgr → Dfx/Dlsym/Config → Legacy950ResMgrProc → locks and flags`

Device subtree contract (dies first → dies last):
`EndpointCtxMgr → CommEngineMgrProc (stop Thread → Mem → HDC → CCU) → SocketProcess → ServerSocketMgrProc → NetDevProc`

EndpointMgr internal contract (dies first → dies last):
`EndpointHandleMgr → ProcRegedMemMgrCache` — endpoints are exclusively owned by the handle
table, and their destruction dereferences the cache through the non-owning alias cacheAlias_
(the R7 alias rule); no ordering constraints among the other EndpointMgr members.

Registered (declaration-order changes, closing a review HIGH):

- `handleRegistryMgrProc_` moved after `perDevResMgr_` — under RESET / exit-fallback, residual channels
  die with the handle table and their destructors return socket/notify via narrow accessors, so the table
  must die before device subtrees (otherwise it touches destroyed objects);
- `mtx_`/`deinited_` in `BaseCommResPerDevMgr` moved before `reserve_` — the slow path of falling back to
  reserve_ during retired-area destruction must take the lock and read the gate, so both die last.
- `channelMapMtx_` in `ChannelHandleRegistryProc` moved before the three tables — the tables' destruction
  (channel destructors lock via the accessor) must find the mutex alive; in the old order the mutex died
  first, so locking during table destruction hit a destroyed `std::mutex` (UB).
- `engineThreadMapMtx_` in `ThreadHandleRegistryProc` moved before the two maps — defensive: when map
  destruction releases `shared_ptr<Thread>` and a `~Thread` were to lock via the accessor, the mutex must
  not die first (current derived destructors only run DeInitImpl and do not call back).
- `reserve_` stays declared before `active_`/`published_`/`retired_` (a review suggestion to move it
  later was rejected — do not re-raise): the retired-area destruction fallback must reach a live
  reserve_; reserve_ is never Init'd and its subtree uses default destructors, so its destruction never
  re-enters FindOrCreate — the review's assumed re-entry path does not exist; moving it later would
  instead make the retired-area fallback hit a destroyed object (a real UAF).

#### R4 Two lifecycle paths: DeInit runs actions, destructors free memory

- `DeInit(ExitProfile)`: the explicit action path (stop threads, clear tables, return
  device resources), executed in contract order, **idempotent** (the root uses an atomic
  exchange so only the first call takes effect; node-level DeInit is self-idempotent).
- Destructors free memory only and contain no action logic; the action fallback is
  `~BaseCommMgr` running the DeInit chain first.
- After DeInit completes, device subtrees move from the active table to the retired area:
  the tree must remain fully resolvable while the action chain runs (destruction-chain
  callbacks go through the accessors).
- DeInit vs concurrent first-touch: DeInit raises the rejection gate (`MarkDeinited`) at
  its start; the slow path re-checks it under the lock and refuses creation (falling back
  to reserve); the action chain iterates over a locked snapshot (`SnapshotActive`),
  mutually exclusive with concurrent insertion.
- Rationale for the DeInit action-chain order (one-to-one with the step[N/9] log markers
  in the implementation; must be preserved when adding/moving actions):
  1. step[1/9] Stop threads first — live threads would call back into dying objects (currently a
     placeholder: ThreadMgrProc has no entities and StopThreads is a no-op; threads
     actually stop when the handle-registry node destructs, which declaration order
     already guarantees to happen before the device subtrees die; explicit stopping
     is a phase-2 work item);
  2. step[2/9] Clear endpoints — their destruction chain calls back into the device-side
     EndpointCtxMgr and returns NIC contexts and registered-memory references, so it must
     precede engine-family teardown;
  3. step[3/9] Clear leftover channels — channel destruction returns sockets/notify via the
     accessors, so it must precede engine-family teardown;
  4. step[4/9] Per-device SocketProcess table cleanup and EndpointCtxMgr leftover force-release —
     returning sockets / force-releasing ctx touches SocketMgr/CCU, so it must
     precede engine-family teardown; endpoint destruction already called Release;
  5. step[5/9] Engine-family teardown — the CCU managers must be alive when ctx callbacks
     return CCU resources;
  6. step[6/9] Bin unload — the engine family no longer references the bin after teardown
     (currently a placeholder: no unload logic; bin handles are reclaimed by the OS
     at process exit);
  7. step[7/9] Net infrastructure — EndpointProcMgr must be alive while NIC contexts are
     returned and listen sockets closed;
  8. step[8/9] Infrastructure and device-state callbacks — pure host actions, identical in
     both profiles (channel-table cleanup already happened at step 3); the refresh callback
     is unregistered by name and the flags are reset (resetting is exit semantics and allows
     re-registration; the reset callback is paired with ACL by name, dies with the process,
     and is not unregistered);
  9. step[9/9] Moving the table into the retired area happens last — the active table must
     stay resolvable while the action chain runs (destruction-chain callbacks go through
     accessors).

#### R5 Dual-profile semantics (exit_profile.h)

- `EXIT_PROFILE_GRACEFUL` (default): full action chain, including device-touching
  actions; the process-exit fallback always uses it.
- `EXIT_PROFILE_RESET`: the device may already be unavailable; skip all device-touching
  actions, host-side cleanup only. Triggered only by explicit calls; the action order is
  the same as graceful (skipped actions are skipped explicitly, never reordered).
- When adding an "action", you must state which profile it belongs to: device-touching
  actions must be skipped under RESET.
- Naming note: the `OnDeviceResetPre` device hot-reset callback is per-device local
  cleanup; it does not DeInit the tree and is not connected to this profile.
- Known limitation: RESET skips the device-side release of the endpoint table and the
  channel table; if handles remain when the process later exits, member destruction still
  touches the device (`~BaseCommMgr` emits a WARNING to expose it). Under RESET the engine
  family (HDC/CCU instances) teardown is likewise skipped (the node logs a WARNING), but
  destruction is not suppressed: `~CcuInstanceMgr` still clears the instance table and runs
  the `~CcuInstance` device-release chain — the RESET "skip device-touching actions" promise
  is pierced at destruction. RESET currently has no production caller (only the exit_path ST
  invokes it explicitly), so this is registered as a known limitation; before wiring a real
  caller, destructor suppression must land first (a flag that makes the destructor
  deliberately leak the device-side resources, with host memory reclaimed at process exit).
  Process-level network resources are likewise left out of the RESET action chain:
  `EndpointProcMgr`'s netDevCtx/serverSocket tables and each device's `NetDevProc`
  `GlobalNetDevMgr` are deferred to destruction, and `~GlobalNetDevMgr` calls
  `HcclNetDeInit` (device-touching) — the RESET "skip device-touching actions"
  promise is pierced on this path as well.

#### R6 Build children at construction; no unlocked lazy init in getters

- A node builds all its children **in its constructor** (including the reserve fallback
  subtree); getters simply return the member reference. Unlocked `if (nullptr) reset(new)`
  lazy init is a first-access data race and must not be added — no exceptions.
- No re-entry via narrow accessors during construction (deadlock red line, see R2).

#### R7 All call sites go through narrow accessors

Tree access chains are long and exceed the column limit. Business code must include
`base_comm_node_accessors.h` and use the short free functions `hcomm::GetXxxMgr(...)`
(returning references); **never write tree access chains directly**. When adding a node,
add its narrow accessor in the same change.
This header is host-side only (ccl_kernel/AICPU has no resource tree). Accessors return
references to instances held by tree nodes — object identity and lifetime follow the
tree; no new globals, no injection.
Legacy `shared_ptr` interfaces are served by empty-deleter non-owning aliases (e.g.
`GetProcRegedMemMgrCacheHolder()`): the alias borrows the address and its refcount does
not keep the object alive — lifetime is guaranteed by the tree's declaration order, and
every alias must die before the object (holders are declared ahead of it). Such aliases
are phase-1 adapters and may be removed in phase 2.
Node classes currently expose internal containers and mutexes through public mutable
getters (e.g. `ChannelMap()` / `ChannelMapMtx()`). This is a transitional form for the
migration: the baseline file-scope statics were visible to a single TU, and after
adoption the lock discipline ("one mutex guards three tables") can no longer be
enforced at compile time and relies on caller discipline. Encapsulation convergence
(named operations with in-class locking, removing the mutable getters) is a phase-2
work item; existing compound operations (multi-table updates under one lock
acquisition) must keep their single-lock atomic granularity during convergence.
File-local aliases (e.g. `ThreadRegistry()` in thread.cc) are allowed only as line-width
shorthands; their bodies must call the narrow accessor directly and must never write
tree access chains.

#### R8 Legacy singletons are anchored, not adopted

Singletons inside libhccl_v2.so cannot be moved across binaries; they are anchored, not
adopted, and anchoring only controls lifetime (no business action). The two anchor levels
differ in guarantee strength:

- Process level: first-touched inside the `Legacy950ResMgrProc` constructor, i.e. during
  tree-root construction — locked to die after the whole tree;
- Device level: a single dependency list (`AnchorDeviceSingletons`; trivial constructors,
  no side effects). Lazy device-subtree first-touch happens after root construction, which
  alone cannot lock the order (reverse static destruction would kill them first); so the
  `Legacy950ResMgrProc` constructor calls the list once with the backup slot
  (MAX_MODULE_DEVICE_NUM) to preheat — the static arrays register during root
  construction and are locked to die after the whole tree. Device-subtree construction
  calls it again with its own slot (effective for UT/standalone construction; a no-op on
  the normal path).

When adding a legacy dependency, add one first-touch line at the corresponding anchor
point (device-level singletons go into `AnchorDeviceSingletons` only).

#### R9 Out-of-range ids fall back to reserve instances

Device-level accessors keep the "out-of-range / conversion failure falls back to a
reserve instance" semantics; key types match the original GetInstance: the CCU trio
(CcuKernelMgr/CcuInstanceMgr/CcuVarEventResMgr) takes deviceLogicId, the rest take
devPhyId:

- phyId-keyed accessors: 0~65 are all valid devices (65 is the spare-card slot, routed
  to the real instance of that device subtree); out-of-range fallbacks mirror the
  baseline accessor by accessor — three idioms coexist (do not "unify" behavior):
  - `GetSocketMgr`: the baseline array has no backup slot; out-of-range (incl. 65) falls
    back to device 0 + WARNING (socket_mgr.cc);
  - `GetEidInfoMgr`/`GetTpMgr`/`GetHccpTlvHdcMgr`: >=65 clamps to 65 + WARNING (same as
    baseline; even the spare card 65 warns, eid_info_mgr.cc);
  - `GetGlobalNetDevMgr`/`GetEndpointCtxMgr`: >65 clamps to 65 + WARNING,
    65 is silently valid (the baseline GetDeviceResMgr warned on every explicit 65;
    this PR deliberately makes it silent — 65 is a legitimate spare slot and warning
    on every legitimate access is noise; the behavior change is registered here);
- logicId-keyed accessors (CCU three-piece): an out-of-range logicId falls back to the
  process-level reserve instance (class-static); the reserve
  is unique per process, so multiple invalid ids share it without per-device isolation
  (exception fallback; the normal path never reaches it). The id member is set by the owner
  at Init time (CcuDrvHandle::Init / CcuInstance::Init); accessors no longer write it back —
  concurrent writes to the shared reserve were a data race (review fix); reserve is never
  Init'd and keeps the default id -1.
   "Never Init" is a convention,
  not enforced: if an invalid id triggers Init (e.g. CcuDrvHandle::Init via conversion
  failure), the reserve's destructor performs hrtFree/Deinit device actions — the same risk
  as the baseline slot-65 backup (equally shared, equally Init-able), i.e. pre-existing
  behavior; the conversion-failure fallback is new exposure added by this PR and lapses
  once phase 2 refuses Init on reserve;
- logicId-keyed accessor (SocketProcess): out-of-range / conversion failure falls back
  to the device-0 subtree (same semantics as the baseline GetInstance slot 0; no reserve);
- GlobalNetDevMgr: on conversion or Init failure it returns the uninitialized instance
  of that phyId's subtree (failures stay isolated per device; no shared fallback).
- Destructor-time fallback semantics: after DeInit (retired-area destruction / RESET leftovers), falling
  back to reserve_ or the device-0 subtree makes cleanup a silent no-op — device-side returns were already
  skipped under RESET; destruction guarantees no crash, not resource return (same root as the R5 known
  limitation; this clause lapses once phase-2 destructor suppression lands).

Fallback paths must log at WARNING/ERROR level.

#### R10 Reverse-audit destructor tree-touching dependencies before adopting a singleton

Before adopting a singleton into the tree, or rerouting its call sites to narrow
accessors, **reverse-audit**: scan the repo for destructors that touch the tree via
narrow accessors (`~Xxx()` calling `GetXxxMgr(...)` / `GetBaseCommResPerDev(...)`).
A static singleton left outside the tree has no contractual destruction order versus
the tree root — the order depends on first-touch timing. If its destructor re-enters
`BaseCommMgr::GetInstance()` through a narrow accessor, both orderings are UB:
constructed before the root means it dies after the root and hits a destroyed mutex
and map; dying before the root after an explicit DeInit means it falls back to the
empty reserve instance and cleanup silently no-ops.
Pick one of three remedies:
1. Adopt it as a tree value member so declaration order (R3) guarantees it dies
   before the nodes it touches (preferred);
2. Cache the dependency pointers at Init, CcuInstance style, and let the destructor
   use only the cache — never a global accessor;
3. Keep it as an anchored singleton (R8) and record its status and plan in the R1
   exception list.
Never "half-adopt": rerouting call sites without fixing the destructor chain is
forbidden — SocketProcess was the live example of this hole (now adopted as a
BaseCommResPerDev value member, remedy 1).
For anchored singletons (which outlive the tree), remedy 2's cache pattern does not
apply — the cached target dies with the tree first, so the cached pointer dangles
too. Tree-touching points on their destructor chains must instead be guarded by
`hcomm::IsResourceTreeTeardownFinished()`: once the flag is set, skip the release
and only clear local tables (first instance: CcuComponent::ReleaseAllTpInfos).

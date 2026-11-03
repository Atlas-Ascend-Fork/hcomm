# light_performance Plugin Design Document (Including Simulator Design)

> Document location: `docs/light_performance/simulator_design.md`
> Requirements source: `docs/light_performance/HCCLSIM/Requirements Description`
> Reference implementation: `src/plugin/checker/src/framework/task_graph_generator_v3/` (graph construction principle),
> `docs/light_performance/HCCLSIM/hccl_sim/` (simulator prototype, **authoritative baseline for the simulation module**)
> Alignment outstanding items and to-be-matched list: Chapter 10
> db task / CCU instruction -> JSON task type mapping master table: Chapter 11

---

## 1. Requirements Background and Document Description

### 1.1 End-to-End Requirements

Integrate light_performance (lightweight performance simulation tool) as an independent plugin into the current project, completing the following flow end-to-end:

```
hccl-vm flow execution --generates--> task sequence / instruction db files
        |
        v
light_performance plugin reads db files --generates--> task graph for performance simulation
        |
        v
Performance simulation module performs performance simulation based on the task graph
        |
        v
Outputs performance profiling json file
```

### 1.2 Sub-Requirement Decomposition and Chapter Mapping

| Sub-Requirement | Content | Status | Document Chapter |
|--------|------|------|---------|
| Sub-Req 1 | Plugin engineering: command line, compile and install, complete plugin architecture | Command line and install design detailed (@estimate one-shot execution channel) | Chapter 3 |
| Sub-Req 2 | Integration with hccl-vm: db file reading, task graph generation | **Detailed design in this iteration** | Chapters 4 and 5 |
| Sub-Req 3 | CCU mode task graph generation (SQE queue + microcode instruction sequence -> task graph) | **Detailed design in this iteration** | Chapter 6 |
| Sub-Req 4 | Performance simulation module (task graph parsing, performance simulation) | **Aligned and incorporated based on HCCLSIM baseline in this iteration** | Chapter 7 |
| Sub-Req 5 | Performance profiling json output (performance metrics, task graph) | Conceptual design | Chapter 8 |

> Chapter 10 records **field gaps, type mismatches, and other conflict points** identified during this alignment (between the db graph construction path and the HCCLSIM simulation model) for future matching reference. Chapter 11 section 11.4 is the mapping between v3-collected profInfo (`HcclTaskProfInfo` three fields) and JSON display types, plus the target state for switching.

### 1.3 Delivery Scope of This Iteration

This iteration focuses only on **db file reading and parsing, and the graph construction module**:

- **Input**: db files under hccl_vm_install/data directory (format and data reference `hccl_vm_install/data/hccl_vm_data.db`).
- **Output**: `DependencyGraph` and `SimConfig` task graph required by the performance module (hccl_sim).
- **Reference code**:
  - Basic graph construction logic: `src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_generator_v3.cc/h`;
  - CCU microcode instruction to task graph: `src/plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/`;
  - AIV mode graph construction: not considered for now.
- **Graph construction scope (core design premise)**: db contains **single-rank task data**; this plugin performs **single-rank graph construction** -- this is one of the main differences from checker (multi-rank joint graph construction). **Cross-rank task edges (post-wait relationships) are all ignored** in the original graph construction process: only post-wait edges within a single rank are considered; cross-rank is not considered (see sections 5.1, 5.4, 6.6).
- **Relationship with checker**: light_performance graph construction module output structure is not identical to checker, but basic graph construction and instruction translation principles are the same; the former does not need checker's subsequent memory conflict verification / semantic verification operations, so appropriate simplification is possible (difference comparison see section 5.8, section 6.7).

---

## 2. Plugin Overall Design

### 2.1 End-to-End Data Flow and Module Division

```
+----------------------- light_performance plugin -------------------------+
|                                                                           |
|  +-----------+   +---------------+   +-----------+   +----------------+  |
|  | db_reader |-->| graph_builder |-->| simulator |-->| profiling      |  |
|  | (Chap 4)  |   | (Chap 5, 6)   |   | (Chap 7)  |   | (Chap 8)       |  |
|  +-----------+   +---------------+   +-----------+   +----------------+  |
|   sqlite read       SQE->Task mapping   simpy discrete    json + trace    |
|   BLOB decode       notify pairing      event             port bandwidth  |
|   operator grouping CCU microcode       sharing           HBM dynamic     |
|                     expansion                           sharing           |
|                                                          eid->port parse  |
|  Input: data/hccl_vm_data.db (+ data/hccl_sim.db, ranktable.json)         |
|  Output: (SimConfig, DependencyGraph, TopoResolver) -> makespan ->        |
|          perf_profiling.json (+ graph dump: DOT/JSON/HTML)                |
+---------------------------------------------------------------------------+
```

Data contracts between modules:

```
db_reader     --OpExecutionGroup(intermediate data model, 4.5)-->  graph_builder
graph_builder --(SimConfig, DependencyGraph)--------------------->  simulator   [output of this requirement]
simulator     --result dict(tasks with start/end + makespan + ports)-->  profiling/report
```

### 2.2 Plugin Directory Structure (Source Location: `src/plugin/light_performance`)

```
src/plugin/light_performance/
+-- __init__.py
+-- __main__.py             # Standalone CLI entry (section 3.3)
+-- main.py                 # @estimate plugin one-shot execution entry (section 3.1/3.2)
+-- manifest.json           # Plugin manifest (name=estimate; installed with package)
+-- pipeline.py             # Shared pipeline: db -> graph -> simulation -> perf_profiling.json
|                           #   (run_perf_sim.py and main.py share, section 3.1)
+-- db_reader.py            # Sub-Req 2: db reading and BLOB decoding (Chapter 4)
+-- sqe_group_merge.py      # SQE batch merge preprocessing: UDMA data tasks within same (stream, sqeGroupId) batch merged to 1 (section 5.1.1)
+-- graph_builder.py        # Sub-Req 2: SQE -> DependencyGraph (Chapter 5)
+-- graph_dumper.py         # Task graph dump: DOT + JSON + HTML
+-- ccu_graph_builder.py    # Sub-Req 3: CCU microcode -> task graph (Chapter 6)
+-- register_model.py       # CCU register/CKE execution model (section 6.4)
+-- sim_config.json         # Default latency/bandwidth config (section 7.2, units us/GBps; installed with package)
+-- bw_profile.json         # Default bandwidth utilization config (installed with package)
+-- simulator/              # Sub-Req 4: simulation module (ported from HCCLSIM, Chapter 7)
    +-- config.py           #   SimConfig (JSON units us/GBps)
    +-- task_model.py       #   Task / TaskType / TRANSFER_TYPES / INVALID
    +-- dependency_graph.py #   DependencyGraph (int stream id)
    +-- task_timing.py      #   Task latency formulas (single-point maintenance)
    +-- bandwidth_profile.py#   Bandwidth utilization table
    +-- topo_resolver.py    #   eid->ports topology resolution (ranktable.json)
    +-- input_builder.py    #   JSON task file construction + load_config
    +-- simulator.py        #   HcclSimulator / NetworkPort / _BandwidthManager
    +-- report.py           #   Text report + Chrome trace
    +-- __main__.py         #   hccl_sim compatible CLI (independent of db path)
```

> The `simulator/` subpackage is ported from `docs/light_performance/HCCLSIM/hccl_sim/`; interfaces follow HCCLSIM as the baseline (Chapter 7). The repository root `dump_graph.py` is a one-click dump entry (db -> DOT/JSON/HTML, optionally with simulation time).
> `manifest.json` and `main.py` (one-shot execution entry, Chapter 3) are placed directly in the package directory; at install time, the whole directory becomes the `plugin/light_performance` plugin directory (install layout see section 3.1; CMake rules see section 3.4).

### 2.3 Relationship with Existing Plugins

| Plugin | Responsibility | Relationship with light_performance |
|------|------|---------------------------|
| checker | Communication correctness verification (deadlock/memory conflict/semantic) | Reuses its graph construction principle; main difference: checker performs **multi-rank joint graph construction**, light_performance performs **single-rank graph construction** (cross-rank pairing edges not built; remote synchronization tasks retained, approximated by fixed remote latency); and no verification-type post-processing |
| runner | Virtual runtime (per-task simulated execution of data plane) | Also consumes opTask/ccuInstr; light_performance only reads, does not write, does not advance isDone |
| insight | Visualization | light_performance's profiling json / chrome trace can be consumed by it (future) |

> **Plugin tag and directory/module name**: This plugin's install tag is `estimate` (host `@estimate` command channel and mode gate are hard-coded by tag; see section 3.1), while the plugin directory and Python package name are both `light_performance` (`plugin/light_performance`) -- tag corresponds to "host routing identity", directory/package name corresponds to "code organization identity".

---

## 3. Plugin Engineering Design (Sub-Requirement 1)

### 3.1 Plugin Identification and manifest.json

**Plugin tag fixed to `estimate`** (not the module name light_performance): The host CLI hard-codes gate and path conventions for this plugin --

- The `hccl-vm plugin run @estimate` execution channel in `src/cmd/cmd_base_utils.cc:599-680` (`RunUserPlugin`) has special handling for `estimate`: only available in estimate mode (`sim::IsEstimateMode()`); plugin directory fixed to `<installRoot>/plugin/light_performance` (tag and directory name decoupled); directly reads its `manifest.json` and synchronously executes the `entry` command (see section 3.2 for details);
- Mode mutual exclusion: `@checker` is unavailable in estimate mode (`cmd_base_utils.cc:618-624`); `@estimate` is unavailable in non-estimate mode -- correctness verification and performance simulation belong to the two running modes' wrap-up actions respectively;
- Plugin scan/resident model (`src/plugin/hccl_plugin_manager.cc:94-169`, `src/plugin/hccl_plugin.cc`) matches tag by the manifest `name` field (scans `<installRoot>/plugin`, depth <= 2).

manifest.json (source and install both in the package directory: `src/plugin/light_performance/manifest.json` -> `plugin/light_performance/manifest.json`):

```json
{
    "name": "estimate",
    "version": "1.0.0",
    "entry": "python3 main.py",
    "dependency": {
        "min_core_version": "1.0.0"
    },
    "setting": {
        "db": "../../data/hccl_vm_data.db",
        "device": null,
        "iter": null,
        "sim_config": "sim_config.json",
        "bw_profile": "bw_profile.json",
        "ranktable": "",
        "out_dir": "data",
        "per_op_mode": true,
        "enable_trace": true,
        "dump_graph": false,
        "max_trace_events": 0
    }
}
```

- `name`: Plugin tag; host routes by it (`hccl-vm plugin run @estimate`).
- `entry`: Startup command within the plugin directory. @estimate goes through the one-shot synchronous execution channel (section 3.2); at execution, cwd is the plugin directory, and relative paths are resolved relative to the plugin directory.
- `setting`: Plugin-private configuration key-values, read by main.py itself (following checker's `SettingManager` self-managing configuration approach; Python side parses json directly). Path-type configs are resolved relative to the plugin directory (i.e., package directory): `db` defaults to `<installRoot>/data/hccl_vm_data.db` (same source as checker's `Loader::LoadOpTaskFile`); `sim_config`/`bw_profile` are latency/bandwidth and utilization configs in the section 7.2 format; defaults take `sim_config.json`/`bw_profile.json` in the same directory as manifest, installed with package; `ranktable` defaults to `ranktable.json` in the same directory as db; `device`/`iter` correspond to CLI's `--device`/`--iter`; `out_dir` is the output root directory (subdirectories built by run timestamp underneath); `per_op_mode` currently supports only per-operator (warns when false and executes per-operator); `enable_trace`/`dump_graph`/`max_trace_events` control additional artifacts.

Source and install layout (plugin directory = package directory, no nesting):

```
Source = install artifact structure (src/plugin/light_performance -> hccl_vm_install/plugin/light_performance):
plugin(light_performance)/            # plugin directory = light_performance package directory
+-- manifest.json                     # plugin manifest (name=estimate)
+-- main.py                           # @estimate one-shot execution entry (reuses pipeline.py)
+-- __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
+-- sim_config.json                   # default latency/bandwidth (section 7.2)
+-- bw_profile.json                   # bandwidth utilization
+-- simulator/                        # simulation subpackage (Chapter 7)
+-- data/                             # runtime output (perf_profiling.json etc., Chapter 8;
                                      #   produced only in install tree; source tree has no such directory)
```

> main.py is inside the package directory but executed as a script (`python3 main.py`): it adds the package's **parent directory** (`plugin/`; source tree is `src/plugin/`) to `sys.path` before `from light_performance.pipeline import ...` -- in-package relative imports are unaffected; source tree and install tree layout are consistent.
> The repository root `run_perf_sim.py` (manual entry) and `main.py` share the same pipeline implementation: main.py adds "install-tree path conventions + manifest setting reading" on top of it.
> Source tree debugging still uses run_perf_sim.py (section 3.3); in the install tree, `hccl-vm plugin run @estimate` triggers main.py.

### 3.2 Call Chain and Process Model (@estimate One-Shot Execution)

Host side requires only one path adaptation -- the @estimate one-shot execution channel is already implemented (`cmd_base_utils.cc:599-680`); this iteration changes the special-case branch's plugin directory from `plugin/estimate` to `plugin/light_performance` (tag remains `estimate`; directory is the package directory):

```
hccl-vm plugin run @estimate
    |
    v
RunUserPlugin("@estimate")                                  # subcmd_plugin.cc:55-57 routing
    +-- Gate: must be in estimate mode; otherwise error exit  # cmd_base_utils.cc:611-617
    +-- Read <installRoot>/plugin/light_performance/manifest.json  # missing file/entry field -> error
    +-- system("cd <installRoot>/plugin/light_performance && python3 main.py")   # synchronous blocking execution
       +-- Exit code non-0 -> HCCL_VM_ERROR; command returns failure overall
```

**Long-duration command client wait strategy** (`cmd_base_utils.cc` `StartHostClient`/`ServerListen`): In-session `hccl-vm` sends via POSIX message queue to host; **host responds only after `ParseCommand` returns (i.e., command execution completes)**; CCU MS mode microcode expansion can produce 10^5-level task counts (actual test 192,005 tasks; software simulation ~54s), far exceeding the typical 5s response cap. Therefore:

- Client, after recognizing `plugin run @estimate`, adopts a long-wait cap (default 30 minutes; env var `HVM_CLIENT_RESP_TIMEOUT_MS` overrides). During slice wait, every 30s prints to terminal "command still executing on host side" prompt (first at 5s); timeout prints clear warning. Other commands maintain 5s cap.
- Response carries the original command string (`ACK:<cmd>`). Client empties response queue before sending command, and only accepts responses strictly matching this command -- avoids misreading late responses from previous timed-out commands and returning early (especially when retrying the same command mid-interrupt).
- host prints `[HVM] @estimate performance simulation start/end` markers to terminal at simulation start/end (stdout; output with the report); this mechanism ensures **the prompt returns only after simulation truly ends**, no longer creating misunderstanding of "simulation already ended".

main.py (plugin side) execution flow:

```
RunEstimate():
    setting   <- parse manifest.json setting section (paths resolved relative to plugin directory)
    db        <- setting.db (default <installRoot>/data/hccl_vm_data.db, derived via __file__ as ../../data)
    config    <- setting.sim_config / bw_profile (fallback to built-in defaults when missing, section 7.2)
    ranktable <- ranktable.json in same directory as db (_find_ranktable default rule; consistent with host InitHvmCommEnv
                written data/ranktable.json, cmd_base_utils.cc:437-450; no config needed)
    groups    <- db_reader.load_op_executions(db)            # Chapter 4
    topo      <- TopoResolver(ranktable)                     # section 7.6
    for group in groups:                                    # per operator (section 5.1)
        config, dep_graph = graph_builder.build(group)      # Chapters 5, 6
        result = HcclSimulator(config, dep_graph, topo).run()   # Chapter 7
    profiling.dump(<plugin>/light_performance/data/<timestamp>/perf_profiling.json)   # Chapter 8
```

Comparison with checker resident process model (stdin single-line JSON command loop) and selection rationale:

| Dimension | checker (resident model) | estimate (one-shot model) |
|------|--------------------|----------------------|
| Launch method | `plugin install` fork+pipe resident (`hccl_plugin.cc:74-156`) | `plugin run` `cd && entry` synchronous execution (`cmd_base_utils.cc:655-660`) |
| Trigger protocol | stdin JSON: `{"action":"status","payload":{"status":"finish","data_id":...}}` | No protocol: entry starts; all input paths via convention + manifest setting |
| data_id | Message carries (host `DumpData` generates) | Not passed; output directory uses run timestamp as substitute (section 10.3 #13) |
| Lifecycle | Recycled on stop command / pipe EOF / host exit | Process natural exit; exit code is execution result |
| Adaptation reason | Verification must be triggered by host at flow end; may run multiple times | Simulation is offline batch processing: consumes already-landed db; runs once then exits |

> **Evolution direction** (not implemented in this iteration): If "flow end automatically triggers simulation by host" (resident model) is needed in the future, main.py adds stdin command loop to reuse the same pipeline -- host's message channel (`SendMessageToPlugin`) and resident launch (`RegisterPlugin`) are both already in place; at that time only need to switch @estimate in `RunUserPlugin` from special-case branch back to generic message path.

Data contracts (fixed conventions in install tree; all aligned with existing mechanisms):

| Path | Source | Description |
|------|------|------|
| `data/hccl_vm_data.db` | Written during host flow execution | Main simulation input; estimate mode produces **single-rank task data** (phantom peer); naturally aligned with section 4.6/5.1 single-rank graph construction premise |
| `data/ranktable.json` | Written by `InitHvmCommEnv` | eid->port resolution (section 7.6); `_find_ranktable` defaults to same directory as db; no explicit configuration needed |
| `plugin/light_performance/sim_config.json`, `bw_profile.json` | CMake installed with package directory (section 3.4) | Section 7.2 format (units us/GBps) |
| `plugin/light_performance/data/<timestamp>/` | Created at main.py runtime | Output directory. **Must not land under `data/`**: host `remove_all("data")` on each start (`src/main.cc:32`); data directory contents not preserved across sessions |

### 3.3 Command Line Design

Host integration mode (`plugin run` is the simulation entry, replacing manual run_perf_sim.py execution):

```bash
# (1) Start environment in estimate mode (choose one)
hccl-vm start <topo> --estimate             # interactive mode; PS1 shows (estimate)$>
hccl-vm run <topo> --estimate <sample command>     # one-shot mode
# (Within an already-started session, can also hccl-vm mode estimate switch, subcmd_mode.cc)

# (2) Run business -> produce data/hccl_vm_data.db (single-rank task list)

# (3) One-click performance simulation (only available in estimate mode; section 3.2 gate)
hccl-vm plugin run @estimate
# Artifacts: plugin/light_performance/data/<timestamp>/perf_profiling.json (+ trace / graph dump)
```

- The one-shot execution channel does not forward command line arguments; simulation parameters (target device, opIter filter, trace/dump switches, etc.) are configured via the manifest `setting` section (section 3.1); change config then re-run `plugin run @estimate` to take effect.
- `plugin install @estimate` is also executable (goes through `RegisterPlugin` resident launch channel); main.py one-shot completion and exit is acceptable degraded behavior (section 10.3 #14); regular use does not require install.

Standalone debug mode (does not depend on host; convenient for offline db analysis; implemented):

```bash
# Simulation + report (optional trace export, task graph dump)
python3 -m light_performance \
    --db hccl_vm_install/data/hccl_vm_data.db \
    [--config sim_config.json] [--bw-profile bw_profile.json] \
    [--ranktable ranktable.json] [--device 1] [--iter 0] \
    [--trace trace.json] [--dump-dir graph_dump] \
    [--max-trace-events 0] [-v] [-s]

# One-click simulation (run_perf_sim.py; outputs perf_profiling.json; shares pipeline with main.py)
python3 run_perf_sim.py hccl_vm_install/data/hccl_vm_data.db \
    [-o perf_profiling.json] [--config ...] [--bw-profile ...] [--trace-dir ...] [--dump-dir ...]

# One-click task graph dump (DOT + JSON, optional HTML; --no-sim skips simulation)
python3 dump_graph.py hccl_vm_install/data/hccl_vm_data.db [--html] [--no-sim]
```

| Parameter | Description |
|------|------|
| `--db` | hccl_vm_data.db path (required) |
| `--config` | Latency/bandwidth config (section 7.2 format; units us/GBps; defaults use built-in defaults) |
| `--bw-profile` | Bandwidth utilization config (defaults to 100% utilization) |
| `--jetty-num` | Override db-landed UB multi-channel (jetty) count, used for multi-jetty utilization scaling (defaults per-operator `opDetails.ubMultiChannelNum`; see section 7.2) |
| `--ranktable` | Topology file, used for eid -> physical port resolution (defaults to `ranktable.json` in same directory as db; when network tasks exist but no topology, **skips simulation and only dumps**) |
| `--device` | Target device (deviceId) for single-rank graph construction; defaults to first device in opDetails; needed only when db contains multi-device data |
| `--iter` | Simulate only operators of specified opIter (defaults all) |
| `--trace` | Export Chrome tracing JSON (multi-operator splits file by `_<commId>_<opIter>`) |
| `--dump-dir` | Task graph dump directory (per-operator output .dot/.json/.html, including simulation time) |
| `-v` / `-s` | Task-level detail / summary only |

### 3.4 Compile and Install Design (hccl_vm_install/plugin/light_performance)

In `src/plugin/CMakeLists.txt`, add install rule (following insight's directory install precedent; Python plugins need no compile target. Plugin directory is the package directory; manifest.json/main.py/default configs are all inside the package directory; one DIRECTORY rule carries all content):

```cmake
# estimate performance simulation plugin: plugin directory is light_performance package directory; install whole directory
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/light_performance"
    DESTINATION "plugin"
    FILE_PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
    PATTERN "__pycache__" EXCLUDE
    PATTERN "data" EXCLUDE)      # Runtime output directory not included in install package
```

Comparison with existing plugin install methods:

| Plugin | Install Method | Artifact |
|------|---------|------|
| checker | `install(TARGETS checker DESTINATION plugin/checker)` + `install(FILES manifest.json)` (`src/plugin/checker/CMakeLists.txt:67-82`) | Compile artifact + manifest |
| insight | `install(DIRECTORY insight DESTINATION plugin/visualization PATTERN frontend EXCLUDE)` (`src/plugin/CMakeLists.txt:26-39`) | Python directory + manifest |
| **estimate (this iteration)** | `install(DIRECTORY light_performance DESTINATION plugin)` (single rule; manifest/main.py/default configs all inside package directory, installed with package) | `plugin/light_performance/`: Python package + manifest (plugin directory and package directory unified; no nesting) |

Build and verify: `bash build.sh` (internally `cmake && make && make install`; install prefix defaults to in-repo `hccl_vm_install/`, `CMakeLists.txt:89-92`) then check install artifact:

```
hccl_vm_install/plugin/light_performance/
+-- manifest.json
+-- main.py
+-- __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
+-- sim_config.json / bw_profile.json
+-- simulator/ ...
```

- Package path: `build.sh --pkg` packages the entire `hccl_vm_install` into a tar; estimate plugin is distributed with it.
- Dependencies: python3 (>=3.8, including sqlite3 standard library) and **simpy** (discrete event simulation engine; current environment 4.1.2; `simulator/simulator.py` imports); deployment environment needs preinstallation (insight plugin similarly depends on python3; convention consistent); see section 10.3 #16.

---

## 4. db File Reading and Parsing (Sub-Requirement 2 - Upper Half; Focus of This Iteration)

### 4.1 db Location and Opening

- Path fixed to `<installRoot>/data/hccl_vm_data.db` (consistent with checker's `Loader::LoadOpTaskFile`).
- Read-only open: Python `sqlite3` standard library, `mode=ro` URI, to avoid write conflicts with running hccl-vm:
  ```python
  conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
  ```
- Auxiliary library `hccl_sim.db` (same directory): Used only to resolve `opDetails.commId` to communicator identity (`Communicator` table; `comm_id[128] + comm_hash` uniquely identifies); optional.

### 4.2 Table List and Association Model

hccl_vm_data.db tables are created by `OpDbOps::InitSchema()` (`src/runnerdb/modeldb/db_hccl_op_db_ops.cc:76-184`). Tables consumed by light_performance are:

| Table | Primary Key / Association | Content | Use |
|----|----------|------|------|
| `opDetails` | `id`; `pid` -> opTask table name | One row per operator per rank: pid/deviceId/rankId/commId/opIter/syncIter/streamId(main stream)/root/rankSize/srcRank/dstRank + `opDetail` BLOB (operator parameters) + `algorithmName` (algorithm selection result; DFX hijack/log dual-path truncation backfill; see note below) + `ubMultiChannelNum` (UB multi-channel/jetty count; proxy level0 stub reads env var `HCCL_UB_MULTI_CHANNEL_NUM` and lands in db; unset/invalid defaults to 1; see section 7.2 multi-jetty scaling) | Operator grouping, main stream identification, operator parameters, algorithm selection result, bandwidth utilization scaling |
| `opMemInfo` | `opDetailId` -> opDetails.id | input/output/ccl three memory segment addresses and sizes | Direction determination (section 5.5), data size verification |
| `opTask_P_<pid>` | `opDetailId` -> opDetails.id | SQE task queue: deviceId/streamId/taskType/isDone + `opTaskMeta` BLOB | **Main graph construction input** |
| `ccuChannels` | `(deviceId, dieId, channelId)` | CCU channel: src/dst die/device/rank + leid/reid(16B EID) + protocol + jettyNum + jettyId array | CCU channel -> peer/jetty mapping (Chapter 6) |
| `ccuInstrRes` | `(deviceId, dieId)` | Microcode instruction space per die: instrCount + instrSpace BLOB (32K x 32B) | CCU microcode source (Chapter 6) |
| `ccuInstr` | `ccuInstrResId` -> ccuInstrRes.id | One row per microcode dispatch (startId/instrInfoSize) | Only used to sense "new microcode dispatched"; decoding does not depend on it |
| `syncRecords` | -- | host-side stream synchronization records | Not used this iteration |
| `JettyMaps` | `opDetailId` | eid mapping recorded per operator | Backup (not used when ccuChannels covers) |

> Table structure definitions see `include/runnerdb/sim_op_db_types.h`; writer semantics see `src/runnerdb/tracedb/db_sim_op_db_ops.cc`.

> **algorithmName column source (algorithm selection result truncation)**: Dual-path truncation; both last-wins backfill `sim::UpdateOpDetailAlgName`, mutually backup, write same value.
>
> 1. **DFX hijack stub (main path)**: Business calls per-operator after algorithm selection completes `HcclDfxRegOpInfoByCommId(commName, &HcclDfxOpInfoCompat)` (hccl repository op_common.cc); its `algTag` carries the selected algorithm name (format `{opType_commName}_{algName}_{host|device}[_{dataType}...]`; graph mode `Graph_{algName}_{...}`). Level=2 mode (default) mounts only level0+level2; level1's dfx_stub is not in the process. `src/proxy/level2/hcomm_api_stub.cc` uses level2 strong symbol to preempt resolution (preload order precedes the real libhcomm.so loaded by business via python plugin chain), parses algorithm name from algTag and backfills, then via `DlsymRealWithFallback` (RTLD_NEXT + libhcomm.so SONAME fallback + dladdr to prevent self-call) **forwards real implementation, transparently returns value, does not change call semantics**. Structure layout replicated per hccl_res_dl.h (`static_assert` locks algTag offset 68); content verification (commId prefix + host/device suffix) filters op-begin and other algorithm-name-less registrations; fallback for business header drift. In Level=1 mode, this stub is shadowed by level1 dfx_stub (behavior same as older version); truncation falls back to log path.
> 2. **Log truncation (backup path)**: The tool is the implementer of the business log underlying interface (business `DlogRecord` weak symbol resolves to `src/proxy/level2/log_stub.cc`). level0 stub `RecordOpDbInfo` first inserts opDetails row (sets runnerdb global `g_currOpDetailId`) then calls real business; business internal selector output INFO-level algorithm selection log (`[HcclGetAlgRes] algName=` / `SelectMinCost: selected algName=` / old selector `the selected algo type is ` three marker types; same operator multiple matches overwrite later) extracted via `DlogPrintStub` entry matching. Truncation executes before message enters tool log output (immune to `log_config.yaml` level); proxy layer simultaneously provides `acllogCheckDebugLevel` always-pass stub (immune to business-side log level threshold).
>
> Old db auto-`ALTER TABLE` to add columns by `OpDbOps::InitSchema()`; plugin reads old db without columns falls back to empty value.

**Operator grouping model**: One operator instance = rankSize opDetails rows under the same `(commId, opIter)` (one per rank), each with an opMemInfo and a set of tasks (`WHERE opDetailId=?`; table name derived from that row's `pid` column as `opTask_P_<pid>`; reference checker `QueryCompositeOpDetailByOpIter`).

### 4.3 opTask Dynamic Table and Multi-pid Handling

- Each task-writing process creates its own table `"opTask_P_" + getpid()`; AICPU device-side process uses `getppid()` to share table with host.
- **Discover all task tables** (must enumerate; multi-pid is the norm; early sample db has `opTask_P_477776`/`opTask_P_477780` two tables; current sample is single table):
  ```sql
  SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'opTask_P_%'
  ```
- Consumption order: Within the same operator, by `id` auto-increment order (= SQE dispatch order; this is the basis for notify first-come-first-served pairing).

### 4.4 BLOB Decoding Specification (pack(1), Little Endian)

All BLOBs are C struct direct memcpy (no serialization framework); Python side uses `struct` module to decode. The following layouts are all verified against `hccl_vm_install/data/hccl_vm_data.db` (from v5 collection onward, sample opTaskMeta length 190B; historical samples 189B/188B).

#### 4.4.1 opTaskMeta (190 bytes)

Struct definition shared with writer: `include/sim_profiling_defs.h` (profiling reconciliation info single-source definition) and `src/plugin/checker/header/internal/task_meta_defs.h` (HcclTaskMetaData).

v5 collection struct (compared to v4's 189B version): After `sqeGroupId` adds `taskSrc` (1 byte) -- task source marker (`HcclTaskSrc`: 0=default SQE/aclrt native path, 1=DPU i.e. host CPU RoCE `ibv_dpu_stub` collection path), for performance simulation DPU sequence merge pass to precisely split by source (section 5.1.2). taskData start offset 69 -> 70.

v4 collection struct (compared to v3's 188B version): After `jettyId` adds `sqeGroupId` (1 byte) -- UDMA SQE batch grouping marker (all WQEs within one doorbell share same group); for performance simulation entry to batch-merge (section 5.1.1). taskData start offset 68 -> 69. Group number semantics: **Circular increment doorbell batch number 1..255** (255 wraps; collection side `device_sqe_parse_stub.cc` maintains global `g_currSqeGroupId`, `% 255 + 1`) -- batch number unique within 255 window; doorbells of other operators not visible in this operator's task list but similarly consume batch numbers, so same-stream-same-number within the list must be same doorbell; **0 reserved as "untagged"** (HcclTaskMetaData construction default); counter never takes 0; merge entry treats `sqeGroupId <= 0` all as not participating (section 5.1.1).

v3 collection struct (compared to v2's 185B version): After `taskType` adds `HcclTaskProfInfo` sub-structure (3 consecutive bytes; checker/runner forwards as opaque whole): `profTaskType` (aligned with hcomm TaskParamType value range 0~22) and `linkType`/`transportType` (aligned with hcomm DfxLinkType value range 0~9 / DfxTransportType value range 0~4; used for profiling data reconciliation). taskData start offset 65 -> 68.

> **Implementation status**: `db_reader.py` auto-identifies layout by BLOB length -- `>= 190B` uses v5 (`<BBBBQIQQIBB16s16s`; taskSrc decodes into `HcclTaskMetaData.src_type`; this branch **must precede v4 determination**; otherwise 190B would first hit the `>= 189` v4 branch causing eid/taskData offset misalignment); `>= 189B` uses v4 (`<BBBBQIQQIB16s16s`; sqeGroupId decodes into `RawTask.sqe_group_id`; src_type=0); `188B` uses v3 (`<BBBBQIQQI16s16s`; sqe_group_id filled -1 not participating in merge); `185~187B` uses old v2 (`<BQIQQI16s16s`; profInfo filled 0xFF=UNKNOWN) for backward compatibility; `< 185B` discard with warning. profInfo consumption (`task_type` display name + link/transport) **landed** (#18, section 11.4).

v2 collection struct (compared to earlier 169B version): Header adds `lcEid[16]` (explicit local EID); taskData members previously added `srcRankId/dstRankId` fields.

> **rank field fallback: srcRankId/dstRankId deprecated (current convention)**: The `srcRankId/dstRankId` fields previously added to taskData members have been rolled back and deleted; unified to use `srcDeviceId/dstDeviceId` -- collection side (estimate mode, `device_sqe_parse_stub.cc` etc.) writes communicator rankId directly into `srcDeviceId/dstDeviceId` fields, i.e. **deviceId field carries rankId** (peer rank unresolved is `0xFFFFFFFF`). `db_reader.py` after unpacking directly consumes as `src_rank_id/dst_rank_id` naming (reads `srcDeviceId` as `srcRankId`). This is a hard switch: Old BLOBs collected before field rollback (rank/device dual-field layout) are no longer supported.

```
Header 70B:  struct.unpack_from("<BBBBQIQQIBB16s16s", blob, 0)
+------+------+- ----------------+--------+-----------+-----------------------+
| Offset| Len | Field          | Type   | Sentinel  | Description           |
+------+------+- ----------------+--------+-----------+-----------------------+
| 0    | 1    | taskType     | u8 enum| --        | Redundant consistent with table column taskType |
| 1~3  | 3    | profInfo     | substruct| 0xFF      | HcclTaskProfInfo:      |
|      |  1   | .profTaskType| u8 enum| 0xFF      | Aligned with hcomm TaskParamType |
|      |  1   | .linkType    | u8 enum| 0xFF      | Aligned with hcomm DfxLinkType |
|      |  1   | .transportType|u8 enum| 0xFF      | Aligned with hcomm DfxTransportType |
| 4    | 8    | commId       | u64    | --        | Communicator ID         |
| 12   | 4    | rankId       | u32    | 0xFFFFFFFF| When default, backfill via device->rank |
| 16   | 8    | deviceId     | u64    | --        | Physical device ID     |
| 24   | 8    | streamId     | u64    | --        | Stream ID              |
| 32   | 4    | jettyId      | u32    | 0xFFFFFFFF| Invalid = not over network |
| 36   | 1    | sqeGroupId   | u8     | 0         | Doorbell batch number 1..255 circular increment (v4 new, section 5.1.1); 0 = untagged |
| 37   | 1    | taskSrc      | u8 enum| 0         | Task source (v5 new, section 11.4.6): 0=default, 1=DPU(host ibv) |
| 38   | 16   | rmEid[16]    | bytes  | all 0     | Remote EID             |
| 54   | 16   | lcEid[16]    | bytes  | all 0     | Local EID (v2 new)     |
| 70   | 120  | taskData     | union  | Interpret by taskType, see below          |
+------+------+-----------------+--------+-----------+-----------------------+
```

taskData union internal layout (relative offset 69):

| taskType | Member | Format String | Fields (in order) |
|----------|------|--------|------|
| 0/1 (notify) | `NotifyTask` | `<IQIBB` | srcDeviceId(carries rankId), notifyId, dstDeviceId(carries rankId), notifyCount, protocol |
| 3 (transMem) | `TransMemTask` | `<IQIQQB` | srcDeviceId(carries rankId), srcOffset, dstDeviceId(carries rankId), dstOffset, len, protocol |
| 2 (reduce) | `ReduceTask` | `<IQIQQBBBB` | srcDeviceId(carries rankId), srcOffset, dstDeviceId(carries rankId), dstOffset, dataCount(actually byte size), dataType, reduceOp, protocol, resv |
| 4 (ccu) | `CcuTask` | `<BBHHHII` + `<13Q` | dieId, missionId, timeout, instStartId, instCnt, key, argSize, args[13] |
| 5 (aiv) | `AivTask` | `<Q` | launchIdx |
| 6 (sync) | `SyncStreamTask` | `<Q` | syncIdx |
| 7 (writeWithNotify) | `WriteWithNotifyTask` | `<IQIQQQBBBBB` | srcDeviceId(carries rankId), srcOffset, dstDeviceId(carries rankId), dstOffset, len, notifyId, isReduce, dataType, reduceOp, protocol, notifyCount |

> `len` and `dataCount` are both **byte size** (reduce's field name has historical reasons; actually used as size). `WriteWithNotifyTask.len` is also byte size; `isReduce=1` means write payload is reduce data (`dataType`/`reduceOp` valid at this time); mapped as single `Write_With_Notify` task (section 5.2).

#### 4.4.2 taskType Enum Full Table

`HccLTaskMetaType` (`include/sim_common_defs.h:184-193`):

| Value | Enum | light_performance Processing |
|----|------|----------------------|
| 0 | `NOTIFY_WAIT` | -> local_wait / remote_wait (section 5.2) |
| 1 | `NOTIFY_RECORD` | -> local_record / remote_record |
| 2 | `REDUCE` | -> memcpy / local_copy (reduction approximated; section 5.7) |
| 3 | `MEM_CPY` | -> memcpy / local_copy (judged by jettyId) |
| 4 | `CCU_GRAPH` | -> CCU subgraph expansion (Chapter 6) |
| 5 | `AIV_GRAPH` | Not supported for now; skip and warn |
| 6 | `SYNC_STREAM` | -> Skip, do not generate node (section 5.6) |
| 7 | `WRITE_WITH_NOTIFY` | -> Single task mapping `Write_With_Notify` (network write + attached notify; not split; section 5.2) |

#### 4.4.3 opDetails.opDetail (28 bytes)

`OpDetails` (`task_meta_defs.h:141-165`): `<HHH` header + union:

```
opType u16@0 | dataType u16@2 | reduceType u16@4
opV1: count u64@6 | dataType u16@14 | reduceOpType u32@16          (total 28B)
opV2: sendCount u64@6 | sendDataType u16@14 | recvCount u64@16
      | recvDataType u16@24 | extInfo u16@26                       (AllToallV class)
```

opType is `HcclCMDType` (ALLREDUCE=2, ALLGATHER=6, REDUCE_SCATTER=7, ALLTOALL=10 etc.; full enum see `src/plugin/checker/header/external/hccl_types.h:162-189`); used for operator labeling in profiling report. `opExtInfo` BLOB (optional) stores V operator counts/displs, AllToallV send matrix, etc.; this iteration only forwards.

V1/V2 layout distinguished by opType (level0 stub writer-side convention, `hccl_op_stub.cc`): **V2 only AllToAll(10)/AllToAllV(8)/AllToAllVC(9)** (`BuildOpDetailsV2`; sendCount/recvCount are this rank's send/receive total element counts); **AllGatherV(16)/ReduceScatterV(17) are V1 + opExtInfo** (count = this rank side element count). db_reader decodes by opType choosing layout; V2 operators decode send/receive two-side element counts and types (`send_count/recv_count/send_data_type/recv_data_type`); `data_size/data_type` follow send side to keep old consumer compatible; `reduce_op` set 0.

#### 4.4.4 ccuChannels BLOB

| Field | Format | Description |
|------|------|------|
| `leid` / `reid` | Each 16B raw bytes | Source/destination endpoint EID (`URMA_EID_LEN=16`) |
| `jettyId` | `jettyNum x u32` little endian | Consecutive jetty number array (C struct reserves `u32[64]`; sample actual test jettyNum=1, jettyId=(1024,)) |
| `protocol` | int column | CommProtocol: HCCS=0/TCP=1/ROCE=2/UB_CTP=3/UB_TP=4/PCIE=5/SIO=6 |

Note: `srcRankId/dstRankId` two columns are actually filled with device_id (channel establishment precedes communicator; see writer `hccp_ccu_stub.cc:401-405` comment); rank semantics need correction via opDetails.

#### 4.4.5 ccuInstrRes instrSpace

- `instrSpace` = one die's **complete microcode instruction space snapshot**: `32K entries x 32B = 1 MiB` (`CCU_INSTRUCTION_NUM=32*1024`, `CCU_INSTRUCTION_SIZE=32`, `include/sim_common_defs.h:35-36`).
- The i-th instruction is at byte offset `i*32`; `instrCount = max(historical value, startId+instrCnt)`; **allows holes**.
- When reading, only take the first `instrCount` entries; CCU task's `instStartId/instCnt` are used directly as subscript range.
- `ccuInstr` table is only for runtime sensing (runner polls `COUNT(ccuInstr)+COUNT(ccuInstrRes)` change); light_performance decoding does not depend on this table.

### 4.5 Intermediate Data Model (db_reader Output)

```python
@dataclass
class OpExecutionGroup:          # One operator instance (single-rank view: only target device task data)
    comm_id: int
    op_iter: int
    op_type: int                 # HcclCMDType
    rank_size: int
    main_stream_ids: dict[int, int]      # deviceId -> main stream streamId (opDetails.streamId)
    tasks: list[RawTask]         # All SQEs of this operator; globally ordered
    mem_info: dict[int, MemInfo]         # deviceId -> input/output/ccl ranges
    channels: dict[tuple, ChannelInfo]   # (srcDev, srcDie, channelId) -> channel info
    instr_space: dict[tuple, list[bytes]]# (deviceId, dieId) -> 32B instruction list
    algorithm_name: str          # Algorithm selection result (opDetails.algorithmName; DFX/log dual-path truncation backfill; old db empty string)
    ub_multi_channel_num: int    # UB multi-channel (jetty) count (opDetails.ubMultiChannelNum; old db/zero value as 1)
    send_count / recv_count: int # Send/receive two-side element counts (V2 layout operators; V1 same as count)
    send_data_type / recv_data_type: int # Send/receive two-side data types (same as above)

@dataclass
class RawTask:
    global_id: int               # Global order across opTask_P_* tables (by table pid value + in-table id sort)
    op_detail_id: int
    device_id: int
    stream_id: int
    meta: HcclTaskMetaData       # 4.4.1 decoding result (dataclass form)
```

`meta` contains v2 new field: header `lc_eid` (explicit local EID; section 5.5 local eid parsing uses preferentially); NotifyTaskData/TransMemTaskData both carry `src_rank_id/dst_rank_id` (under estimate collection convention, unpacked from rankId value carried in deviceId field; see section 4.4.1 rank field rollback).

### 4.6 Data Integrity Verification

For each `OpExecutionGroup`:

1. **Single-rank filter (design premise)**: Performance plugin consumes only **single-rank task data** (sample db has only device 1; design norm, not data missing). If db contains multi-device data, filter opDetails and their tasks by target deviceId (`--rank`; defaults to first device). `rankSize` is only passed through as metadata; does not require `len(opDetails rows) == rankSize`.
2. **notify peer** (no backfill):
    - estimate collection convention (`device_sqe_parse_stub.cc`; both task types consistent): notify's `srcDeviceId` (carries rankId) = record side, `dstDeviceId` (carries rankId) = wait side; both explicitly filled by emitting side. Peer rank unresolved is `0xFFFFFFFF`.
    - Earlier version's `dstDeviceId == 0` backfill fallback has been deleted with rank field rollback (0 sentinel conflicts with valid rank 0; and emitting side always explicitly fills after, so no backfill needed).
    - `WRITE_WITH_NOTIFY` (7) carries notify facing **write peer** (wait side not in this db; single-rank simulation premise); does not participate in local pairing; its `dstDeviceId` (carries rankId) is the write destination.
    - Note: runner/checker consumer side (`DecodeAndExpandTaskMeta`) at decoding boundary splits 7 into [write, NOTIFY_RECORD] two sub-tasks; light_performance **does not split** -- single task mapping `Write_With_Notify` (section 5.2), ensuring simulation tasks one-to-one correspond with db tasks.
3. **Decoding length verification**: `len(opTaskMeta) >= 185` (auto-adapts to v4/v3/v2 layout by length; section 4.4.1); `len(opDetail) >= 28`; `len(instrSpace) >= instrCount*32`. If not satisfied, discard row and warn (checker does direct memcpy; here more lenient).

### 4.7 Decoding Example (Actual Test hccl_vm_data.db)

Sample db (REDUCE_SCATTER, opType=7, dataType=4/FP32, count=8, rankSize=2, devType=6/950) actual test decoding results:

- `opTaskMeta` length is constantly 188B (v3 collection sample; v4 collection onward is 189B; section 4.4.1); taskType distribution: 4 (CCU_GRAPH) and 6 (SYNC_STREAM) alternate.
- opDetails only device 1 (rank 0) -> single-rank data (design norm); main stream stream 2.
- Task sequence (per opIter main stream):

```
stream 2 (main stream):  CCU_GRAPH(die=1, mission=0, instStartId=14, instCnt=100, argSize=13)
                   SYNC_STREAM(syncIdx=n)          # microcode dispatch + stream sync
```

- CCU task args match data directory `sqe_info_device_1_die_1_mission_0_startId_14.txt` item by item (args[0]=281475405602881, args[1]=281475405602945 ...); `args[0]/args[1]` actual test is opMemInfo's inputAddr/outputAddr.
- `ccuInstrRes` (dev 1, die 1) instrCount=114 = instStartId(14) + instCnt(100).
- In early 169B sample containing MEM_CPY/NOTIFY, actual test: MEM_CPY's srcOffset/dstOffset can both fall into opMemInfo's input/output/ccl ranges (`input` contains `output`; ReduceScatter's output is the first 1/rankSize segment of input); network copies with jettyId, their offset also appear as "local range address" (communicator memory symmetric layout; peer buffer same VA as local). Therefore **direction determination cannot rely solely on address** (see section 5.5). This conclusion applies to v2 data as well.

#### 4.7.1 v2 AICPU Mode Sample (opTask_P_118800)

REDUCE_SCATTER, rankSize=8, count=32768 (FP32, output=128KB), 76 tasks, stream 2 (main) / 6 (scheduler) / 7..13 (one stream per peer). Key differences from CCU mode sample:

- taskType distribution: NOTIFY_WAIT x 30, NOTIFY_RECORD x 30, MEM_CPY x 8 (7 network writes jet=1..7 + 1 local), REDUCE x 7 (local), SYNC_STREAM x 1; no CCU_GRAPH; `ccuChannels` empty.
- **Direction all determined by rank dimension** (section 5.5 rule 1): 7 network MEM_CPY all are `srcRank=0/dstRank=1..7` (write); task offsets do not intersect opMemInfo ranges (range inference fails); `direction_inferred=0`.
- **notifyId reused across ranks**: Such as `RECORD(nid=30, dst=dev2)` (wait side at peer; skipped) and local `WAIT(nid=30)` (record side at peer; not in this db) appear in pairs; two rounds each 7 x 2; local paired 16 pairs (nid 3/13 main <-> scheduler, 15..27 scheduler -> peer stream, 6..12 peer stream -> scheduler); unpaired local WAIT 14 -> billed by `remote_wait_delay` (section 5.4).
- lcEid/rmEid and ranktable.json match verbatim (`...0040...01010001` -> port `1/0` etc.); network task ports all resolvable.

### 4.8 eid -> rank Index (`build_rank_index`, Peer Rank Resolution)

Peer rank resolution (section 5.5 SQE fallback, section 6.7 CCU main path) shares one eid(hex) -> rank_id index. Construction priority and basis:

1. **RankEidMap table (main path)**: runnerdb `hccl_sim.db` (same directory as hccl_vm_data.db) `RankEidMap` table. In estimate mode, Communicator table has only one row for this rank; peer rank uses this table as authoritative (`sim_models.h` comment; host precedent `GetRmtRankIdByEid`, `device_sqe_parse_stub.cc:889`). Row BLOB layout consistent with `sim::RankEidMap`: `id(u64) + rankId(u32) + eid(16B) + pad(4B)`; eid is "hex string byte order", same source and order as `ccuChannels.leid/reid`; Python side matches by `bytes.hex()` direct equality. This table is written by ranktable.json parsing (`topo_ascend_cluster_parser.cc:229`); mock-comm scenario fully overwrites with communicator initialization, so authoritative.
2. **ranktable.json (fallback)**: When RankEidMap empty / `hccl_sim.db` missing / table absent, expand `rank_list[].rank_id <-> level_list[].rank_addr_list[].addr` (addr_type=EID) to build isomorphic index. Actual test: both path indexes fully consistent (8 rank x 9 eid = 72 entries).

Read-only open (`file:...?mode=ro` URI); read failure (including table absent) degrades to fallback rather than errors. Note **`ccuChannels.srcRankId/dstRankId` cannot be directly used as communicator rank**: Writer side (`hccp_ccu_stub.cc:401-405`) explicitly fills global endpoint deviceId (channel establishment precedes communicator; same channel can be reused by multiple domains); offset exists against opDetails.rankId numbering. checker thus deprecated this field (`g_allRankChannelInfo` stores only dstDeviceId/dstDieId).

---

## 5. SQE Task Graph Generation (Sub-Requirement 2 - Lower Half; Focus of This Iteration)

### 5.1 Overall Pipeline

```
OpExecutionGroup
   |
   v
(0) SQE batch merge (sqe_group_merge.py, section 5.1.1): UDMA data tasks within same (stream, sqeGroupId) batch merged to 1
   |
   v
(0') DPU sequence merge (sqe_group_merge.py, section 5.1.2): DPU large message N-1 front blocks + tail block WRITE_WITH_NOTIFY merged to 1
   |
   v
(1) Per-SQE -> Task node (section 5.2 mapping table; CCU_GRAPH handed to Chapter 6 for expansion)
   |     stream_id: int (SQE stream uses original streamId directly; CCU skeleton task also belongs to its SQE original stream; section 6.5)
   v
(2) stream serial chain: group by (deviceId, streamId), global_id ascending -> dep_graph.add_stream_chain
   v
(3) notify pairing: Cross-rank tasks do not participate in pairing (nodes retained); only local notify first-come-first-served -> dep_graph.add_match (section 5.4)
   v
(4) Network task parameter completion: direction inference / eid resolution (section 5.5)
   v
(5) CCU subgraph expansion (Chapter 6): CCU_GRAPH placeholder -> microcode instruction interpretation -> internal nodes + CKE pairing
   v
(6) Output assembly: (SimConfig, DependencyGraph) -> HcclSimulator(config, dep, topo)
```

### 5.1.1 SQE Batch Merge (Pre-Graph-Construction Preprocessing)

Collection-side UDMA one doorbell (SQE) can carry multiple WQEs; tasks still land per WQE, marked with `sqeGroupId` for batch attribution (circular increment doorbell batch number 1..255, 255 wraps, never takes 0; 0 = construction default "untagged"; section 4.4.1). Before graph construction, split batches by **(streamId, sqeGroupId)** -- split key explicitly contains stream; merge all data tasks (MEM_CPY/REDUCE/WRITE_WITH_NOTIFY) within batch to single entry, aligned with hcomm batch movement profiling's `ExecProfilingAll` metric -- three entries (`pipeline.py` / `__main__.py` / `dump_graph.py`) call uniformly, ensuring simulation and dump metric consistent:

- **Stream-grouped isolation guarantee**: Split key contains streamId; **cross-stream tasks never merge into same batch** (one doorbell belongs to only one stream; single-doorbell batch is not split by key); group number (doorbell batch number) is unique within 255 window -- other operators' doorbells, though not visible in this operator's task list, similarly consume batch numbers; so **same-stream-same-number must be same doorbell**, no cross-doorbell mis-merge (multi-operator db safe).
- **Description information** (type/address/endpoint/notifyId/reduce parameters/profInfo) takes **the last** data task in batch; **length takes full-batch cumulative** -- consistent with `ExecProfilingAll(loc[insNum-1], rmt[insNum-1], totalSize, transferOp[insNum-1])` field by field; trace/dump display name similarly takes last task's op name (section 11.4.1). When batch mixes write/read directions or different dataTypes, similarly follows last task (single SQE batch usually homogeneous; mix is rare scenario).
- **Not participating in merge**: notify/SDMA/sync tasks (notifyId unique; pairing depends on order; local tasks retained in place; do not affect batch boundary); untagged/no-group-number tasks (`sqeGroupId <= 0`: 0 = untagged, -1 = v3 and earlier BLOB lacks this field) retained in place and **cut current batch** (cannot determine batch attribution; conservatively end current batch); UDMA non-data tasks (inline write notify) participate only in batch splitting; inline notifies within batch are placed after merged task (notify causality).
- Merged task occupies first data task slot of batch (batch head). Split key contains stream, ensuring in-batch tasks share same stream; stream serial chain (section 5.3) and notify pairing (section 5.4) relative order unaffected; local (non-UDMA) tasks do not participate in merge and do not reset batch boundary (retained in place; do not affect batch continuity).

### 5.1.2 DPU Sequence Merge (Pre-Graph-Construction Preprocessing)

DPU (host CPU RoCE) path large message WriteWithNotify sequence merge, executed after 5.1.1 (`pipeline.py` uniform call). Rules and semantics see section 11.4.6: Scan by DPU stream order **src/dst dual-address contiguous** MEM_CPY chain (only `src_type==1` DPU-source data tasks participate; other tasks retained in place, do not cut chain). Chain closed by address-contiguous WRITE_WITH_NOTIFY merged to single entry -- **the closing task is a natural terminator**; two independent WriteWithNotify calls cannot cross-merge (each call's tail block necessarily closes current chain); mis-merge only remains the theoretical scenario of "independent pure write's address恰好 contiguous with WriteWithNotify start address" (semantics equivalent to a contiguous data segment + closing notify; simulation result no difference). Chain without closing is not merged (independent pure writes retained per-entry `RDMASend`); merged task occupies closing task slot (notify causality); merged front blocks removed from original positions; unmerged tasks retain original stream order.

**Single-rank graph construction strategy (design premise; main difference from checker)**: db contains only single-rank task data; DependencyGraph describes only that rank's tasks and their dependencies. **Cross-rank pairing edges (post-wait relationships) all not built (task nodes retained; approximated by fixed remote latency)**:

- SQE level: RECORD with waitDev != this device (cross-rank post), WAIT with recordDev != this device (cross-rank wait) **retain task nodes** -- graph normally by stream order (src != dst expresses remote; uses fixed remote latency); but **do not build any pairing edges** -- only care about post-wait pairing within single rank (section 5.4).
- CCU level: Instructions posting to remote CKE; waits depending on remote CKE set **similarly generate task nodes** (Notify_Record/Notify_Wait; src=local, dst=INVALID -> fixed remote latency); but do not build cross-rank pairing edges (section 6.4/6.6).
- **Retain cross-rank data movement** (direction=write/read MEMCPY): They are the main body of network bandwidth usage from this rank's view. What is ignored is only the synchronization dependency edges, not the transfers themselves.
- makespan semantics: Single-rank view operator end-to-end duration = this rank's computation/copy + this side initiated network transfer + this rank synchronization wait; when peer is ready is not modeled (equivalent to assuming peer ready instantly).

**Simulation granularity**: Default per-operator (each `(commId, opIter)` one DependencyGraph, one simulation, one record), consistent with checker's per-operator verification metric; `--all-in-one` optional full mode (multiple operators on same stream naturally serial).

### 5.2 taskType -> hccl_sim Task Mapping

Aligned with HCCLSIM new task model (`TaskType`: `NOTIFY_RECORD`/`NOTIFY_WAIT`/`REDUCE_INLINE`/`MEMCPY`/`UB_INLINE_WRITE`/`UB_WRITE_OR_READ`/`WRITE_WITH_NOTIFY`; see section 7.3):

| db taskType | Judgment Condition | hccl_sim TaskType | Task Parameters | Latency Model |
|-------------|---------|-------------------|-----------|---------|
| NOTIFY_RECORD (1) | waitRank == local rank | `NOTIFY_RECORD` | notify_id=int, src=dst=local_rank | Fixed + trigger Event |
| NOTIFY_RECORD (1) | waitRank != local rank | `NOTIFY_RECORD` (cross-rank post: node retained, not paired) | notify_id=int, src=local_rank, dst=peer rank (unresolved falls back INVALID) | Fixed remote_record_delay; does not trigger Event |
| NOTIFY_WAIT (0) | recordRank == local rank | `NOTIFY_WAIT` | Same as above | Wait for Event |
| NOTIFY_WAIT (0) | recordRank != local rank | `NOTIFY_WAIT` (cross-rank wait: node retained, not paired) | notify_id=int, src=peer rank (unresolved falls back INVALID), dst=local_rank | Fixed remote_wait_delay |
| MEM_CPY (3) | jettyId == INVALID | `MEMCPY` | size=len, src/dst_rank=taskData endpoint rank (unfilled falls back local) | HBM dynamic shared bandwidth |
| MEM_CPY (3) | jettyId valid, write direction | `UB_WRITE_OR_READ` | size=len, src_rank=local, dst_rank=peer rank, src/dst_eid, jetty_id, protocol, is_write=True | Port write direction bandwidth shared |
| MEM_CPY (3) | jettyId valid, read direction | `UB_WRITE_OR_READ` | size=len, src_rank=peer rank, dst_rank=local, src/dst_eid, jetty_id, protocol, is_write=False | Port read direction bandwidth shared |
| REDUCE (2) | jettyId == INVALID | `REDUCE_INLINE` | size=dataCount, data_type=task-level dataType enum name, src/dst_rank same as MEMCPY local branch | HBM dynamic shared bandwidth |
| REDUCE (2) | jettyId valid | Same as MEM_CPY network branch (section 5.7 approximation) | Same as above (including data_type) | Same as above |
| WRITE_WITH_NOTIFY (7) | Always write direction (local src -> remote dst; no direction inference) | `WRITE_WITH_NOTIFY` (**single task; not split**) | size=len, notify_id=notifyId, src_rank=local, dst_rank=peer rank (unresolved falls back via eid->rank index), src/dst_eid, jetty_id, protocol, is_write=True; when isReduce=1, data_type=dataType enum name | Port write direction bandwidth shared + notify fixed record delay after write lands (remote) |
| CCU_GRAPH (4) | -- | (Expand; Chapter 6) | -- | -- |
| AIV_GRAPH (5) | -- | Skip | -- | Warn |
| SYNC_STREAM (6) | -- | **Skip** (host-side stream sync point; section 5.6) | -- | -- |

> waitRank/recordRank are taskData endpoint rankIds (carried in deviceId field; src=record side, dst=wait side; section 4.4.1/4.6); used only for **judging local/cross-rank**: Cross-rank notify tasks retain nodes at mapping stage but do not enter pairing (section 5.1 strategy); pairing only occurs between local notifies. notify local/remote is expressed in new model by `src_rank == dst_rank` (`is_local_notify`).
> **WRITE_WITH_NOTIFY (7) single task mapping** (one-to-one with db task; not split into write + NOTIFY_RECORD two sub-tasks): Write direction always write (emitting-side convention src local -> dst peer; no section 5.5 direction inference); goes through port write direction bandwidth shared; after write data lands, sends notify to peer; billed by fixed record delay (peer wait not in this db; single-rank simulation premise; same semantics as cross-rank record; self-write src==dst edge case uses local record delay); `notifyId` retained with task (trace/dump display); when `isReduce=1`, only affects `data_type` display (same enum name mapping as REDUCE branch).
> **Network task rank directly consumes taskData** (deviceId field carries rankId; emitting-side convention: write=src local/dst peer, read=src peer/dst local): write fills `dst_rank=peer rank`; read fills `src_rank=peer rank`. Peer rank unresolved (`0xFFFFFFFF` or equals local) falls back INVALID (section 10.1). Direction explicitly carried by `is_write`; simulator no longer relies on `src_rank==0` heuristic (section 10.3 #1 fixed; task.txt old input path still uses heuristic).
> Local eid preferentially takes SQE explicit lcEid (v2 collection new); default via jettyId -> ccuChannels channel leid derivation (section 5.5, section 10.1).
> This table's TaskType is internal latency model classification; trace/dump task display name already switched to profInfo.profTaskType's op name (section 11.4.1 mapping table; section 10.3 #18 landed); correspondence with this table's TaskType follows section 11.4.1 "Simulation TaskType" column.

### 5.3 Stream Serial Chain Construction

- Group by `(deviceId, streamId)`; within group by `global_id` ascending (= SQE dispatch order) form chain.
- Simulator launches one simpy process per chain for serial execution (Chapter 7) -- **does not need checker's mainStart virtual start and main stream special handling**: Simulation focuses on end-to-end makespan; each chain starts from time 0 in parallel; the first task's WAIT of secondary stream naturally forms actual synchronization with main stream.

### 5.4 Notify Pairing Algorithm (Local Only)

**Cross-rank post-wait edges all ignored** (section 5.1 strategy): Before pairing, first classify by section 4.6 peer backfill result; cross-rank notify tasks (RECORD with waitDev != this device; WAIT with recordDev != this device) retain nodes at section 5.2 mapping stage (graph by stream order), but do not enter pairing. Therefore pairing key simplifies from checker's triplet to single notifyId:

```
checker semantics:   pairing key = (recordDeviceId, waitDeviceId, notifyId)   <- multi-rank
This plugin:         first filter cross-rank -> pairing key = notifyId                    <- single rank
```

Consistent with checker's `AddLocalNotifyEdges` (dispatch order simulation + first-come-first-served consumption, `task_graph_generator_v3.cc:847-946`) in principle; light_performance simplifies to **two-pass scan + first-come-first-served**:

```
Input: All local notify tasks of some operator (cross-rank nodes retained but do not participate in pairing)
Pass 1: Scan by global_id ascending
    RECORD -> pending_records[notifyId].append(task_id)                        # queue
    WAIT   -> if pending_records[notifyId] non-empty:
                 rec = pop front; dep_graph.add_match(wait_id, rec_id)          # first-come-first-served
             else:
                 unmatched_waits.append(wait_id)                               # wait for "later" record
Pass 2: Process unmatched_waits (WAIT appearing before RECORD case)
    Replay by global_id order: WAIT unmatched hangs; when RECORD arrives, pair with earliest hanging WAIT
    (Successfully paired RECORD synchronously removed from pending_records, ensuring diagnostic statistics accurate)
Wrap-up: Still-residual WAIT -> normal (see below); still-residual RECORD -> warn; normal data should not have
```

Description:

- db's id order is dispatch order; and stream serial chain already ensures "tasks after WAIT in same stream do not execute first"; therefore no need for checker's full queue simulation and deadlock detection (deadlock exposed in simulation as makespan anomaly / task-not-executed form).
- Same notifyId with multiple RECORD/WAIT (such as early 169B sample n11/n15 repeatedly appearing on stream 6; main stream two iterations) naturally paired by first-come-first-served rule; **notifyId allowed to be reused**.
- **Unpaired local WAIT is normal semantics**: In v2 AICPU sample (section 4.7), notifyId has cross-rank reuse -- local `RECORD(nid=30, dst=dev2)` (wait side at peer; node retained but not paired) and local `WAIT(nid=30)` (record side at peer; not in this db) appear in pairs; that WAIT billed in simulation by `remote_wait_delay` fixed delay (section 7.6); warning only a hint.
- Early 169B sample's cross-rank pair: `RECORD(n18, s1->d2)` (notifying dev2) and `WAIT(n18)` (backfilled recordDev=2) both retain nodes; do not build cross-rank pairing edge: graph by stream order; `src != dst` uses fixed remote delay; data movement tasks on stream 8 (MEMCPY(jet=3) etc.) unaffected.

### 5.5 Network Task Direction and eid Resolution

**Direction determination** -- Network data tasks uniformly mapped to `UB_WRITE_OR_READ`; direction determines port billing direction (is_write) and rank endpoint values. taskData endpoint rankId carried by deviceId field (section 4.4.1 rank field rollback); emitting-side convention (`device_sqe_parse_stub.cc`): write = src local/dst peer, read = src peer/dst local:

| Priority | Rule | Basis |
|--------|------|------|
| 1 | srcRank == local and dstRank != local -> write; dstRank == local and srcRank != local -> read | Collected endpoint rankId (carried in deviceId field; section 4.4.1); actual test samples (section 4.7) all determined by this; no inference |
| 2 | When rank not distinguished (equal or both unfilled), use memory range: dst falls into local output range -> read; src falls into local input range -> write | Actual test sample induction (section 4.7): Fails when task offset does not intersect opMemInfo ranges |
| 3 | Fallback: Default write; diagnostic statistics `direction_inferred` count | -- |

> Network task on Task explicitly carries `is_write` (section 10.3 #1 fixed; CCU path section 6.7 synchronously fixed); simulator uses this to select write/read billing direction and port query eid (`src_eid`/`dst_eid`); no longer relies on `src_rank==0` heuristic; task.txt old input path lacks this field; falls back to heuristic determination.
> `WRITE_WITH_NOTIFY` (7) **does not participate in direction determination**: Semantics always write (local src -> remote dst); at graph construction directly set `is_write=True`.

**eid resolution** -- Network task via eid queries ranktable to get physical port (`TopoResolver.get_ports_by_eid`):

```
SQE:        Local eid preferentially takes SQE explicit lcEid (v2 new); default via jettyId
            --ccuChannels(this device outbound channel)--> leid(local eid)
            Peer preferentially takes channel reid; default falls back SQE rmEid; when channel missing, local eid degrades to any local channel leid
CCU microcode:   channelId -> channels[(dev,die,channelId)] -> leid/reid + jettyId list
            When channel missing, local eid preferentially takes that mission SQE's lcEid; degrades to same-die any channel leid
write direction: Simulator queries task.src_eid -> ports; read direction queries task.dst_eid -> ports
```

- `protocol`: Channel protocol column -> HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO (CommProtocol names; display only); `link_type`/`transport_type`: task profInfo (DfxLinkType/ProfTransportType names; section 11.4.2/11.4.3; `_add_task` uniformly fills; reconciles with msprof L1).
- Local eid unresolvable warns + `eid_unresolved` count; when network tasks exist but no ranktable, simulation overall skipped (TopoResolver directly throws on missing eid; see section 10.3).

**Peer rank resolution** -- SQE network task's peer rank (`_peer_rank`): taskData endpoint rankId (carried in deviceId field; emitting side already resolved by `GetRmtRankIdByEid`) prioritized; unresolved (UINT32_MAX / local value) falls back via eid->rank index (section 4.8) by peer eid; still none then INVALID + `rank_unresolved` count.

### 5.6 SYNC_STREAM Handling

Host-side stream sync point (sample main stream appears once per iteration); represents host waiting for this stream to complete; **does not represent device-side task**: Does not generate task node; does not consume time (makespan is stream chain last task end time).

- Diagnostic statistics `skipped_sync` count;
- Previously approximated by "unpaired local `NOTIFY_RECORD` fixed delay placeholder"; removed because it introduced false `local_record_delay` at stream tail (and appeared in trace as pseudo notify task with notify_id=invalid).

### 5.7 REDUCE Approximation

Reduce = network transfer + computation. Current model approximates by data movement:

- Network (jettyId valid): Billed by `UB_WRITE_OR_READ` port bandwidth delay (direction same as section 5.5 determination).
- Not network (jettyId INVALID): Mapped as `REDUCE_INLINE`; billed by HBM bandwidth dynamic sharing (`size / hbm_bandwidth`); computation delay not yet listed separately (future can calibrate by `dataType x dataCount`; see section 10.1 data_type field gap).

### 5.8 Comparison with checker Graph Construction and Simplification

| checker Step | light_performance | Reason |
|--------------|-------------------|------|
| mainStart virtual node + main stream entry special handling | Removed | Simulation only needs makespan; chain head WAIT already expresses sync |
| AddLocalNotifyEdges queue simulation + deadlock detection (GRAPH_DEADLOCK) | Two-pass scan first-come-first-served; no deadlock detection | No verification requirement; anomalies exposed as simulation result form |
| AddInterRankNotifyEdges cross-rank pairing (full-graph BFS simulation) | **Completely removed**: Cross-rank post-wait pairing edges all ignored (notify task nodes retained; graph by stream order) | Single-rank graph construction (core difference) |
| MemSlice address resolution + memory layout table | Only range attribution for direction determination (optional) | Simulation only needs len; no precise memory model needed |
| CheckSlaveTaskQueue secondary stream structure verification | Removed | Not a verification tool |
| Memory conflict / semantic verification / sync conflict / reachability / single-task check | All removed | Not a verification tool |
| CompactSyncNodes (SYNC_STREAM merging) | Skip, do not generate node (host-side sync point; not device-side task) | makespan is stream chain last task end time (section 5.6) |
| AIV subgraph expansion | Not supported (warn and skip) | Requirement explicitly not considered for now |
| CCU subgraph: queue parallelization transformation, loop merge, resource conflict verification | Queue parallelization simplified implementation (section 7.4.1: assembly-stage skeleton/unit split + fork/completion-set join with attach extra edges + `#v` virtual stream); loop merge equivalent implementation (section 6.5); resource conflict verification removed | Parallelization eliminates queue serialization distortion (section 7.4.1); no verification; node count controllable |

### 5.9 Output Assembly

```python
def build(group: OpExecutionGroup, sim_config: SimConfig | None = None
          ) -> tuple[SimConfig, DependencyGraph]:
    dep = DependencyGraph()
    # (1) SQE -> Task (CCU_GRAPH temporarily stored as placeholder)
    # (2) stream chain (int stream id)
    # (3) notify pairing (add_match)
    # (4) direction/eid completion
    # (5) CCU expansion (ccu_graph_builder.expand_ccu_graph(..., local_rank, alloc_stream))
    return sim_config, dep
```

- `SimConfig` source: `--config` / manifest setting specified JSON (section 7.2 format; units us/GBps), overlaid with bandwidth utilization config.
- Topology: `TopoResolver` built by CLI layer (ranktable location see section 3.3) and passed to simulator.
- Output immediately sendable to `HcclSimulator(config, dep_graph, topo).run()` (returns dict; see section 7.1).

---

## 6. CCU Mode Task Graph Generation (Sub-Requirement 3; Focus of This Iteration)

### 6.1 Organization of CCU Data in db

```
opTask_P_<pid> SQEs with taskType=4 (CCU_GRAPH)
    +-- taskData.ccu: {dieId, missionId, timeout, instStartId, instCnt, key, argSize, args[13]}
           |  instStartId/instCnt define instruction execution window [instStartId, instStartId+instCnt)
           |  args[13] consumed one-by-one by LOAD-class instructions (every 13 args consumed advances sqeIndex)
           v
ccuInstrRes (deviceId, dieId) -> instrSpace: 32K x 32B microcode instruction space
ccuChannels (deviceId, dieId, channelId) -> {dstDeviceId, dstDieId, leid/reid, jettyId[]}
    +-- Channel register values -> peer location and jetty
```

- SQEs grouped by `(dieId, missionId, queue)` into dispatch contexts (single-device graph construction; deviceId constant; queue is SQE stream) (checker's `CcuMissionKey` + latest node append semantics, `task_meta_translator_v3.h:47-62`): Same-key SQEs merge into the same context **only when contiguous with the previous one** (previous entry `instCnt==13` and `instStartId` head-to-tail contiguous, `IsContiguousCcuSqe`, `task_meta_translator_v3.cc:213-231`); otherwise **create new context** -- multiple dispatches of the same mission (instruction window wrap re-dispatch; typical "head SQE (13 Load SqeArgs) + body SQE" dispatched twice) each expand independently: Each context fully replays its own instruction window (including Load SqeArgs header; reinitializes loop counters etc. registers with **this dispatch's args**), avoiding the later dispatch using residual register state from the previous one and missing data movement (section 10.3 #20).
  - **args consumption** (checker `CcuGraphStateV3::GetSqe` semantics): SQEs within context arranged by dispatch order; args consumed slot-by-slot -- every 13 args consumed (read to last slot `sqeArgsId==12`) cursor advances to next SQE; in typical layout, head SQE's 13 Load SqeArgs consume head SQE's args[0..12]; subsequent Load instructions in body SQE window consume body SQE's args.
  - **Register/CKE bitmap globally shared across contexts** (checker `AllRankParamRecorder::Global()` semantics): Later dispatch's wait can be resolved by previous dispatch's residual CKE bits / residual post bookkeeping; consistent with checker's multi-graph shared recorder behavior.
  - **Same-queue dispatch contexts strictly advance serially in dispatch order** (checker queue semantics: single queue strictly by instruction order; same-stream later TaskCcuGraph must wait for previous to complete); cross-queue polling jointly advances (section 6.6).
- Subgraph attachment point: CCU_GRAPH SQE position in stream chain is subgraph entry; after expansion, this placeholder is removed (task graph does not introduce any placeholder nodes); subgraph attaches to SQE predecessor/successor tasks via extra edges (section 7.4.1).

### 6.2 Microcode Instruction Format (32B Fixed-Length)

Same source as `${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode` (this repo reference copy: `ai_skills/skills/convert-ccu-microcode/references/ccu_microcode_v1.h`):

```
CcuInstr = CcuInstrHeader (2B) + union { V1(A5) | V2(A6) | V3 } (30B)

CcuInstrHeader bit fields (little-endian u16):
    code     : 11 bit   (header & 0x7FF)      instruction code
    type     :  4 bit   ((header >> 11) & 0xF) instruction major class
    reserved :  1 bit

Instruction major classes:  LOAD_TYPE=0x0 | CTRL_TYPE=0x1 | TRANS_TYPE=0x2 | REDUCE_TYPE=0x3
```

Chip generation -> instruction set version mapping (checker `GetCcuInstrVersion`, `ccu_task_transform_v3.cc:175-184`): `Ascend950 -> A5/V1`, `Ascend960 -> A6/V2`; other devTypes fall back to A5. Derived from Device's soc_version (hccl_sim.db) or `opDetails.devType` (`expand_ccu_graph` selects `_MissionContext`/`_MissionContextA6` interpreter by `dev_type`). **Both instruction set generations implemented** (A6 ported from checker `ccu_task_transform_instruct_a6_v3.cc`; section 11.3.3). A5/A6 instruction set difference overview see section 6.8.

### 6.3 Instruction Type -> Task Mapping

#### 6.3.1 Master Table

| Instruction Major Class | Representative Instructions | light_performance Behavior | Output Task |
|----------|---------|----------------------|-----------|
| LOAD (register write) | LoadSqeArgsToXn/GSA, LoadImd, LoadX, arithmetic (A6 Add/Sub/...) | Update register model; data plane **does not produce nodes**; A6 LOAD-class commonly carries `setCKE` tail field (mask non-zero -> local `NOTIFY_RECORD`; checker `ProcessSetMask` semantics); A6 Load/Store (HBM <-> Xn data plane) does not model HBM content; registers not backfilled (only retain setCKE sync semantics; warn for trace) | -- (A6 setCKE non-zero produces Notify_Record) |
| CTRL: SetCKE/SetCKBit | Set CKE bit | Local -> `NOTIFY_RECORD` (register CKE bookkeeping; with mask); remote target (cross-rank post; SyncCKE etc.) -> remote `NOTIFY_RECORD` (not paired; section 6.6) | Notify_Record |
| CTRL: Wait/Jmp/Fence | Wait CKE bit / jump / fence | CKE wait (A5 embedded waitCKE field): Insufficient bits pause that queue pointer; bits sufficient generate `NOTIFY_WAIT` (split by contributing bits; with mask); wait depending on remote bits -> remote `NOTIFY_WAIT` (not paired; fixed delay; section 6.6). A6 independent Wait instruction (Xn value compare spin; checker `TransformWaitInstr`): Satisfied passes through no node; unsatisfied = depends on peer remote register write (SyncWtX/TransMem notify) -> top-level resolves via stall, generates remote `NOTIFY_WAIT` (fixed `remote_wait_delay`); within loop body directly generates node and continues. Jmp: A5 evaluates by `Xn[conditionXnId] != expectData` (checker `TransformJmpInstrA5` semantics); A6 supports 6 condition codes (==/!=/>/>=/</<=, >=6 unconditional) + relative/absolute two modes (jumpMode; target `Xn[relTarInstrXnId]`); target limited to current instruction window; out-of-range warns and passes through; within loop body passes through; with step budget defense against backward jump infinite loop. Fence (A6 only): checker does not support; warns and passes through (queue serial model naturally ensures order) | Notify_Wait |
| CTRL: ClearCKE | wait + clear bit | wait semantics; A6 follows checker `ClearWaitMask` semantics to clear consumed wait bits (`clearMask != 0` is checker unsupported scenario; warn + lenient clear target bits) | Notify_Wait |
| CTRL: Loop/LoopGroup | Loop | loopCnt from register; **merge expand** (section 6.5; failure fallback to full expand); A6 Loop `mode==1` is conditional expansion (`wishCKEBit == Xp.loopCtxId + 1024` hits only executes; always takes full-expand path) | None (body instructions mapped per this table; under merge path, data task size x total iterations) |
| TRANS (movement) | LocMem <-> LocMS/RmtMem/RmtMS (A5), local 4 codes + unified TransMem dmaOpCode (A6) | Map by role (section 6.3.2) | `Memcpy`/`Ub_Write_Or_Read` |
| TRANS: Sync (cross-rank sync) | SyncCKE/GSA/Xn (A5), SyncXnWtx/SyncAtx (A6) | Remote set/notify -> remote `NOTIFY_RECORD` (not paired; section 6.6); value sync is remote register write; no node built | Notify_Record (remote) |
| REDUCE | Add/Max/Min | Same as SQE REDUCE approximation (section 5.7); MS is rank-local buffer; always local HBM reduce | `Reduce_Inline` |

#### 6.3.2 TRANS Transfer Role Determination (Consistent with checker `ccu_loop_merge_v3.cc:73-144`)

By src/dst device attribution:

| src | dst | Role | hccl_sim Task |
|-----|-----|------|---------------|
| Local | Local | LOCAL_COPY | `Memcpy` (HBM dynamic shared) |
| Local | Remote | WRITE | `Ub_Write_Or_Read` (src_rank=local, dst_rank=peer, is_write=True) |
| Remote | Local | READ | `Ub_Write_Or_Read` (src_rank=peer, dst_rank=local, is_write=False) |

> Cross-die/card read/write two directions uniformly mapped `Ub_Write_Or_Read`; direction expressed via `is_write`/trace `direction` field; ports billed independently by write/read direction; peer rank resolved via channel reid -> eid->rank index (unresolved is INVALID). To preserve microcode-distinguishable buffer information, CCU-converted movement tasks carry `src_buf_type`/`dst_buf_type` (mem/ms; determined by instruction source/destination operand); SQE/CLI path same-name tasks have no such info; output invalid value N/A.

> Cross-rank **data movement** (WRITE/READ roles) retained -- it is the main body of network bandwidth usage; what is ignored is only cross-rank **synchronization edges** (post-wait; see section 6.4/6.6).

Key field sources:

- `size`: Length register (A5 `lengthXnId` direct read / A6 `xlId` via `GetXnId` -- bit15=1 register number offset by loop `xnIdOffset`) -> query register model.
- Peer location: A6 channel via **register value** (`Xn[GetXnId(xcId)]` value is channelId) -> `channels[(dev,die,channelId)]` -> `{dstDeviceId, dstDieId}`; A5 channel is instruction immediate field. A5 MS-class instruction `msId / 0x8000` = rank, `msId % 0x8000` = MS offset (MS page 4KB) -- **addressing encoding only**: MS is rank-directly-accessible local buffer (checker `GenSliceFromMs` + `AddLocalBatchReduce` always local for all msIds); MS high bits **not used as** local/remote task type criterion; cross-rank data movement expressed by Rmt* role instructions (go through channel). A6 TransMem's `msIdMode==1` marks MS end (write source / read destination is local MS; `src/dst_buf_type=ms`).
- eid: Channel leid/reid (hex string); local eid preferentially takes that mission SQE's `lcEid` (v2 new); default takes same-die channel leid approximation.
- A6 `TransMem`'s `dmaOpCode` (implemented; checker `TransformTransMemInstr`): 3=write, 5=write with notify (notify value writes remote Xn/CKE; target type unresolvable in single rank -> approximate as remote record), 6=read, 0x1A=remote atomic count (no data plane; approximate as remote record), 0x19 (compressed write+atomic) and others unsupported (warn and skip data plane; setCKE local post still executed to preserve sync skeleton); `udfEnable` (network reduce) billed as normal UB movement (section 5.7 approximation; peer reduce not modeled); `udfType != 0` warns and skips data plane.

### 6.4 Execution Model: Register + CKE Bookkeeping (register_model.py)

Python simplification of checker `AllRankParamRecorder` (`ccu_all_rank_param_recorder_v3.h:27-110`):

```python
class CcuRegisterModel:
    xn:    dict[(dev, die, id)] -> int      # general registers (length/loop counter etc.)
    cke:   dict[(dev, die, ckeId)] -> int   # CKE bitmap (set/clear)
    gsa:   dict[(dev, die, id)] -> int      # A5 GSA registers (A6 has no GSA; only xn/cke)
    seen_post: dict[key, list[PostRecord]]  # CKE post bookkeeping (for pairing)

    key = (waitDeviceId, dieId, ckeId)
    PostRecord = {record_device_id, cke_mask, remaining_mask, task_id, is_local}
```

- LOAD/arithmetic instructions "execute" on this model (read/write registers); uninitialized read -> warn and treat as 0 (checker errors and exits; here lenient handling to accommodate incomplete data). A6 arithmetic instructions (Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt) execute with 64-bit wrap semantics (Mul takes low 32-bit product; shift amount truncated by 0x3F; arithmetic shift interprets as signed); LoadX/StoreX indirect addressing target register number out of range (>16383) warns and skips write (checker errors and aborts).
- **A6 in-loop register number offset** (checker `GetXnId`/`UpdateXnId`): When payload register number field bit15=1, in expansion zone (`curLoopIdx >= expandOffset`), offset register number by `xnIdOffset x curExpandCnt`; bit15=0 uses original number directly. CKE/MS id loop offset same rule across both generations (`UpdateId`).
- **CKE pairing rules** (consistent with checker `GenWaitNode`, `ccu_task_transform_v3.cc:56-121`):
  ```
  post matches wait <=> post.waitDeviceId == wait.waitDeviceId
                    && post.dieId == wait.dieId
                    && post.ckeId == wait.ckeId
                    && (post.ckeMask & wait.ckeMask) != 0        # bits intersect
  After match: One wait instruction aggregated to at most 1 local wait node (mask = all contributing
          posts' bits ORed) + 1 remote wait node (bits depending on remote set)
          post.remaining_mask &= ~wait.ckeMask                  # one post's bits can be consumed by multiple waits in parts
          Emptied -> remove from seen_post
  Node dependency: Aggregated wait node builds edges to all contributing posts -- first contributing post goes
          through notify pairing (add_match; Event semantics); others via extra edges
          (child=wait, parent=post; "wait for all parents to complete" pure timing constraint)
  ```
- **CKE bookkeeping only registers this rank's posts** (single-rank graph construction): Cross-rank post generates remote `Notify_Record` node (src=local, dst=INVALID -> fixed remote delay) but does not register `seen_post` (set action occurs on peer rank's CKE); wait bits depending on remote set generate remote `Notify_Wait` node (fixed delay; not paired); see section 6.6.
- When bits not satisfied, that queue pointer pauses (wait for other queues of this rank's post to unlock) -- multi-queue joint advance see section 6.6 algorithm; when same queue has multiple dispatch contexts (same mission multiple dispatches; section 6.1), later context must wait for previous context to complete before advancing.
- Wrap-up verification: `seen_post` residual -> warn and count in diagnostic statistics (checker reports GRAPH_UNMATCHED; here downgraded).

### 6.5 Queue Model and Loop Handling

- **queue**: `CcuSqeParam` grouped by SQE's stream (checker is ccuParams[queueId]); each queue one serial chain. New task model `stream_id` is **int**: SQE stream uses original streamId directly; **CCU queue chain also belongs to its SQE original stream** -- whichever stream the CCU-type SQE is on, the expanded skeleton task is on that stream (`queue` is the original stream number; `alloc_stream("ccu_d{die}_q{queue}")` resolves back to original stream number via `ccu_queue_origin_stream`; same-name reused to merge same-queue multi-missions); only virtual streams newly created by queue parallelization allocated sequentially from base `1<<20`; logical names retained in `stream_labels` for dump/report display. Exception: When same original stream is **occupied by another die's** queue, the latter falls back to base allocation synthetic stream and warns (chaining would serialize cross-die parallel execution); at this time that stream's kind=ccu in `stream_meta`. Within same queue, instructions strictly in order (checker semantics: `ccu_task_transform_v3.cc:189`; "single queue strictly advances by instruction order; cross-queue coordination only through generated wait/post edges").
- **Assembly-stage queue parallelization split** (section 7.4.1): After interpret execution completes, each CCU queue chain further split into skeleton (wait / free set / remote sync nodes -- free set means record nodes not immediately following data task) and asynchronous data movement unit two classes: skeleton stays on queue serial chain (= SQE original stream chain; spliced back to placeholder position by dispatch order) expressing timing framework; asynchronous units reattached to virtual stream `ccu_d{die}_q{queue}#v{n}` (sequential allocation from base `1<<20`); reused across missions; not reused within mission; via fork extra edges connects to skeleton -- serial chain only constrains skeleton and unit internal order; eliminates "entire queue task-by-task serial" distortion.
- **loop expansion** (merge-first; aligned with checker `ccu_loop_merge_v3.cc`):
  - `Loop`/`LoopGroup` instruction iteration count from register; known at interpret time:
    - A5: LoopGroup parameters in Xn (LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41]) and Xm (gsaOffset/MSOffset/ckeOffset); Loop iteration count in Xn[xmId] low 13 bits;
    - A6: LoopGroup parameters changed to be carried by **Xn/Xm/Xp three registers** (checker `LoopGroupXnA6/XmA6/XpA6`) -- Xn: `loopInsCnt:10 | expandOffset:9 | expandCnt:9`; Xm: `ckOffset:10 | msOffset:11 | gsaOffset:32`; Xp: `xnIdOffset:32` (GetXnId offset; section 6.4); Loop iteration count in `Xn[GetXnId(xmId)]` low 13 bits; GSA stride in `Xn[xnId]` low 32 bits; context id in `Xn[xpId]` low 9 bits; `mode==1` is **conditional Loop** -- only expands when instruction `wishCKEBit == loopCtxId + 1024` (`A6_LOOP_BOARD_ENTRY`); otherwise this expansion skipped (checker `ProcessLoopIns`); conditional Loop always takes full-expand path (checker `CollectLoopExpandA6` rejects merge).
    Each Loop within LoopGroup first tries **merge path** (checker `TryProcessMergedLoopA5/A6` -> `LoopIterationMerge` / `LoopExpandMerge` semantics); if verification fails, fallback to per-iteration full expand.
  - **Merge precondition verification** (any unmet triggers WARN fallback; same semantics as checker "Skip merged-loop optimization"):
    1. Loop body contains only mergeable instructions:
       - A5: Simple TRANS / SYNC TRANS (SyncCKE/GSA/Xn) / REDUCE / ClearCKE (`CollectLoopCkeOpsA5` rule; SetCKE, LOAD, LOOP, JMP, TransMem with `setRmtCKEMask != 0`, ClearCKE with `clearMask != 0` all rejected);
       - A6 (`CollectLoopCkeOpsA6` + `Collect*LoopInstrA6` rules): Local TRANS 4 codes / TransMem (`udfType==0` and `dmaOpCode in {3,6}` -- 5 write-with-notify triggers remote post per iteration; rejected; channel must be resolvable, `IsExistRemoteDieInfo`) / REDUCE (`udfEnable` TransMem network reduce similarly folded as trans slot) / ClearCKE (`clearMask==0`); SetCKBit, LOAD, LOOP, JMP, Wait all rejected;
    2. wait/set masks **balance** within loop body (`ConsumeProducedMask`/`ProduceMask`: Each wait's bits must be produced by earlier set; at sequence end all produced bits exhausted);
    3. Cross-expansion CKE/MS resources no conflict (`CheckResourceConflict`: Different expansions after ckOffset/msOffset offset must not reuse CKE/MS numbers; A6's MS conflict check covers local movement/TransMem `msIdMode` MS end and reduce msId list);
    4. Movement/reduce length > 0 (`ValidateLoopLen`) and expansion count x iteration count > 0;
    5. **A6 additional**: in-loop wait CKE ID must not be reused (`CheckA6LoopWaitCkeIdUnique` hardware constraint; checker errors and aborts; here downgrades to full-expand fallback); each expansion's iteration count / length / channel value consistent (checker each expansion `loopCnt` equal + `InstrMerge`/`MemMerge` mergeability's performance-side approximation -- register numbers after xnIdOffset offset may point to different registers).
  - **Merge launch** (`EmitMergedLoopInstrsV3` semantics): Loop body by sample expansion (curExpandCnt=0) executes **once**; slot-by-slot wait -> data task -> set -> clear wait bits; data task size = single-time length x (expansion count x iteration count) (timing-equivalent folding; consistent with checker AddBatchTransMem/AddBatchReduce batch task); other expansions' CKE register bitmap replayed by or/clear idempotence, ensuring post-loop register state consistent with full expand (only PostRecord task bookkeeping not duplicated). All tasks within launch window (data tasks and folded notify tasks) carry `loop_cnt` (post-expansion loop count = N+1 after N expansions) and `loop_iter_cnt` (single loop self-loop count); total iterations = `loop_cnt` x `loop_iter_cnt`; preserving pre-merge loop details for subsequent simulation use; straight-line instructions, full-expand fallback, and SQE/CLI path tasks have both fields as 0 (invalid; trace uniformly outputs N/A).
  - **Quantified effect** (REDUCE_SCATTER actual-test db; 128 expansions x 50 iterations x 13 instructions): Task count 192,005 -> 66 (data tasks 10; local post/wait several; remote sync nodes 35 -- remote post 21 + remote wait 14; section 6.6); graph construction + simulation end-to-end ~55 s -> ~6 ms; makespan 70,113.650 us -> 2,753.711 us (this section's merge eliminates 128K task-level notify overhead and 64K small movements' per-task latency; final value after queue parallelization and remote sync nodes included; section 7.4.1).
  - A6 LoopGroup configuration verification (`loopInsCnt<=512`, `expandCnt<=511`, `expandOffset<=511` and `expandOffset<=loopInsCnt`; checker `TransformLoopGroupInstr`); when unmet, warns and skips entire LoopGroup (defends against out-of-control expansion from abnormal register values); "in-loop waitCKE must not reuse CKE ID" verification see above merge precondition #5 (`ccu_task_transform_instruct_a6_v3.cc:1126-1146`).

### 6.5.1 Merged Loop Pipeline Concurrency Recovery (Loop Tag + Feature Chain Resolution)

**Problem -- Folding pseudo-serial (merge not timing-equivalent for notify structure)**: Section 6.5 merge is timing-equivalent folding for **data tasks** (size x total iterations; total time unchanged under bandwidth-limited scenario), but **not** for loop body CKE wait/set handshake. In real microcode execution, LoopGroup's N expansions launch concurrently; each expansion internally advances by M inner iterations pipelined (typical 4KB scale per iteration). Loop body's `TransMemToMs -> setCKE -> waitCKE -> TransLocMsToRmtMem / TransLocMsToLocMem` is **per-iteration fine-grained handshake**: Iteration k+1's HBM->MS prefetch (memcpy engine) overlaps with iteration k's network write + local copy-out (channel / memcpy engine); steady-state throughput ~= max(phase totals). After merge folding, per-iteration handshake degrades to **single-pair** wait/record node; semantics becomes "front-segment all data (x N x M iterations) complete -> back-segment can start" **full barrier** -- pipeline concurrency folded to serial.

Taking `hccl_vm_install/data/hccl_vm_data.db` first loop template as example (microcode 0-10: `TransMemToLocMs(4KB, set 0:1) -> waitCKE(0:1) -> 7x TransLocMsToRmtMem(each 4KB, set 0:1..0x40) -> TransLocMsToLocMem(4KB, set 0x80) -> waitCKE(0:ff)`; LoopGroup 128 expansions x 256 inner iterations = 32768 times x 4KB = 128MB/task):

```
Current graph construction (after 7.4.1 parallelization; AA = loopgroup leading task representative,
                            i.e. the last node in queue skeleton before loop launch window):
  AA -> memcpy(mem->ms) -> record(0:1)                     <- prefetch unit (128MB, 95.95us)
  record(0:1) ->[pair]-> wait(0:1) -> 7x ub_write + memcpy(ms->mem)
                                    ^ full serial: back-segment 8 units fork from wait(0:1);
                                      via pairing edge waits for prefetch 128MB all complete
  multiple { record(0:xx) / record(0x80) -> wait(0:ff) }       <- loop-end barrier (retained)
```

- Real execution: steady-state ~= max(prefetch 95.95, 7-port write 2526.48) us;
- Current graph: serial 95.95 + 2526.48 us.
- **Quantified** (sample db ALLGATHER; before/after this section's fix actual test): makespan **2657.52 -> 2561.54 us**. Eliminates 95.95 us serial prefetch phase; after prefetch and local copy-out concurrent through HBM dynamic sharing, each 191.79 us (correct expression of pipeline steady state; section 7.5); 7-port network write 2526.48 us becomes only critical path. Error upper bound: When two phases belong to different engines (HBM prefetch vs network write) and durations comparable, loop body time overestimated by near **2x** (serial t1+t2 vs steady-state max(t1,t2)); same-engine phases (such as pure HBM prefetch+copy-out) serialization error absorbed by HBM dynamic sharing (total unchanged); error accumulates linearly with merged loop count.

**Correctness premise -- Merge verification invariant**: Section 6.5 merge precondition #2 (wait/set mask in-body balancing) guarantees merged loop body's **all wait bits produced by earlier sets within body and exhausted at end** -- i.e., all CKE dependencies within merge launch window are intra-loop per-iteration pipeline handshake; no real data dependencies across loops exist (otherwise verification fails and falls back to full expand). Therefore resolving "in-body middle wait and the set it consumes" does not break loop's external synchronization: **loop-end wait** (last wait within launch window; no same-loop data task after it) and **mission completion set** (skeleton tail + all unit tails; section 7.4.1) continue to carry post-loop instruction / SQE successor / same-queue next mission queue-level serial semantics.

#### Plan (Loop Tag + Feature Chain Identification + Delete-and-Reconnect; Iterate to Convergence)

1. **Loop tag**: `_MissionContext` maintains current launch window's `_loop_tag`, named `loopgroup_{i}_loop_{j}` (i = LoopGroup instruction execution sequence number within dispatch context, 0-based; j = in-group cur_loop_idx, 0-based; when loop body nests LoopGroup, each assigned by its launch window; no hierarchy). `_emit_merged_loop` sets within window; `add_task` defaults to inherit new field `Task.loop_tag` (same pattern as `loop_cnt`). **Only merge path tags**; full-expand fallback / SQE / CLI / straight-line instructions empty string (trace outputs N/A). Tag simultaneously serves resolution range limiting (cross-loop / cross-mission no mis-delete) and trace/dump tracing.
2. **Feature chain identification** (record during launch + judge before assembly; avoids reverse query of graph construction result): When `_gen_waits` pairing succeeds, record (wait_task_id -> all contributing post task_id) into `ctx.wait_posts` (aggregated wait's match + extra all contributors; section 6.4). Assembly stage (`expand_ccu_graph` assembly loop start; before skeleton/unit split) scans ctx.nodes sequence; feature chain hit must simultaneously satisfy:
   - W is local paired `Notify_Wait` and `loop_tag == T` (no remote wait within merge window -- verification #2 ensures in-body wait bits all locally produced);
   - W's all contributing posts R* are all `loop_tag == T` `Notify_Record`, and each in ctx.nodes **immediately follows same-tag data task M*** (unit tail record; i.e., folded producer task such as TransMemToMs folded memcpy; data task subset of `_QUEUE_DATA_TYPES`; not limited to Ub_Write_Or_Read -- Memcpy/Reduce_Inline also have this pseudo-serial);
   - **After W (in ctx.nodes order) exists same-tag data task** (mid-body determination; loop-end wait has no same-loop data task after it; naturally exempted; next loop's data task tag different; also not misjudged);
   - R* all do not immediately follow non-data nodes (free set / sync slot local set **does not hit** -- they wait for real dependencies such as remote sync; resolution prohibited).
3. **Delete and reconnect** (prune ctx.nodes; not post-assembly graph surgery):
   - **W always deleted**: Remove from ctx.nodes and dep task table; revoke its pairing edge (`remove_match`) and aggregated extra edges (`extra_edges[W]`) -- this is the essential operation breaking the barrier;
   - **R* depends on consumers**: R does not appear in other waits' `wait_posts` values -> delete together (sample db's set24 only consumed by wt25; directly deleted); otherwise **retain R, only delete W** (its retained edges unaffected; loop-end wait waits normally -- timing equivalent, semantics lossless conservative branch; when post bits partially consumed by multiple waits, takes this branch);
   - **Successor unit reconnect automatic**: After pruning, assembly logic rescans ctx.nodes; units originally forking from W automatically fork to W's preceding skeleton node (i.e., loopgroup leading task AA); when W is skeleton head (no preceding skeleton), that unit becomes entry unit; entry dependency expressed by SQE attach edge (section 7.4.1). **No edge-reconnect graph surgery needed** -- fork parent / skeleton chain / completion set naturally generated by existing assembly rules based on pruned sequence.
   - Register / CKE bookkeeping **zero changes**: Pairing consumption already completed during launch; pruning only graph structure operation; post-loop instruction wait matching uses bookkeeping result unaffected (deleted post's bits already consumed by W; no residual waiter).
4. **Iteration and statistics**: Repeat 2-3 until no new matches (handles multi-level chains in body such as prefetch -> wait -> middle segment -> wait -> end segment: After deleting first layer, second layer exposed as new chain head); `CcuBuildStats` adds `pipeline_chains_broken`; summary row shows as `pb=N`.

Post-fix sample task graph (comparing to this section start; deleted nodes are record(0:1)/wait(0:1); the original "wait(0:1) -> ub_write" gate taken over by AA; timing equivalent):

```
AA -> wait(0:ff)                                        <- loop-end barrier (skeleton)
AA -> memcpy(mem->ms)                                   <- prefetch unit (record deleted)
AA -> multiple { ub_write_or_read -> record(0:xx) }          <- 7-way writes; concurrent with prefetch
AA -> memcpy(ms->mem) -> record(0x80)                   <- copy-out unit; concurrent with prefetch
multiple { record(0:xx) / record(0x80) -> wait(0:ff) }      <- loop-end waits for all data complete
```

#### Implementation Details (Code Landing Points)

- `simulator/task_model.py`: `Task` adds `loop_tag` slot (default empty string); `to_trace_args` outputs `loop_tag` (empty string -> `N/A`; same metric as `loop_cnt`);
- `ccu_graph_builder.py`:
  - `_MissionContext.__init__`: `_loop_tag = ""`, `_lg_seq = 0`, `wait_posts = {}`;
  - `_exec_loop_group`: On entry, record `lg_tag = f"loopgroup_{self._lg_seq}_loop_{cur_loop_idx}"` (passed through to `_emit_merged_loop`); after LoopGroup processing, `self._lg_seq += 1`;
  - `_emit_merged_loop`: Within window, set `self._loop_tag = lg_tag` (`finally` reset; same pattern as `_loop_cnt`);
  - `_gen_waits`: After pairing success, `self.wait_posts[wait_task.task_id] = [p.task_id for p in posts]`;
  - `expand_ccu_graph` assembly loop start calls `_break_pipeline_chains(ctx, dep)` (returns deleted chain count; accumulated into `stats.pipeline_chains_broken`);
- `graph_dumper.py` / `simulator/report.py`: dump `nodes[]` and trace args supplement `loop_tag`; DOT/HTML legend synced (optional; presented with loop_cnt).

trace presentation (`export_trace`; display layer only; does not affect simulation): Tasks carrying `loop_tag` on virtual secondary stream have their display row reattached to this stream's loop tag row (row name is tag such as `loopgroup_0_loop_0`), indicating they are converted from in-loop instructions; other tasks on the same secondary stream retain original display row -- cross-mission reused mixed secondary streams thus split into original row + tag row. Display rows divided by `(stream_id, loop_tag)`; tasks from different virtual streams do not merge rows (retains inter-secondary-stream task division); main stream tasks (including loop-end wait) all retain original row; task `args.stream_id` retains real stream number.

```
_break_pipeline_chains(ctx, dep):
    broken = 0; changed = True
    while changed:
        changed = False
        for W in [n for n in ctx.nodes if type == Notify_Wait and n.loop_tag]:
            posts = ctx.wait_posts.get(W.task_id);  T = W.loop_tag
            if not posts:                              continue   # unpaired/remote
            if any R in posts: R.loop_tag != T:        continue   # cross-loop source
            if any R in posts: R's previous ctx.nodes node is not same-tag data task: continue
                                                                 # free set/sync
            if no same-tag data task after W:           continue   # loop-end barrier
            for R in posts:                                        # --- hit ---
                if R in other wait's wait_posts values: continue        # has retained consumer
                dep delete R; ctx.nodes remove R                       # unit tail taken over by M
            dep.remove_match(W); dep.extra_edges pop W
            dep delete W; ctx.nodes remove W
            broken += 1; changed = True; break                     # rescan
    return broken
```

Acyclicity guarantee: Pruning only deletes nodes and edges; does not add edges. Deleted W/R are all launch-window nodes; fork/join edges recomputed by assembly stage based on pruned sequence (section 7.4.1 acyclicity proof continues to hold); `_drop_cyclic_matches` (section 5.4) and simulator runtime deadlock detection fallback unchanged.

#### Plan Evaluation (Pros/Cons) and Alternatives

**Pros**:

1. Does not change section 6.5 merge semantics or checker alignment logic -- post-hoc resolution; risk isolated;
2. Prunes ctx.nodes before assembly (not surgery on graph construction result): fork parent / skeleton chain / completion set naturally generated by existing assembly rules; avoids "edge reconnect after node delete" graph surgery correctness burden (post-assembly surgery requires manual reconnect of fork edges, skeleton chain, completion set; error-prone);
3. Launch-phase records wait->posts pairing; identification does not depend on graph construction result's dependency shape -- insensitive to section 7.4.1 assembly rule evolution (fork parent selection, unit split adjustment);
4. Loop tag limits resolution range (same loop same mission); prevents cross-loop / cross-mission mis-delete; tag simultaneously feeds back to trace/dump tracing;
5. Deletes nodes rather than only deleting pairing edges: Avoids retaining a semantically dangling unpaired local wait (although its actual test takes `local_wait_delay` = 30ns; timing no difference; trace statistics metric cleaner).

**Cons / Risks**:

1. Feature chain is structural heuristic; identification conditions coupled with launch / assembly logic; requires maintenance as both evolve (coupling surface already compressed to ctx.nodes sequence one layer via "launch-phase record + pre-assembly pruning");
2. `Task` model field expansion; trace/dump/JSON presentation needs sync (Chapter 11 metric);
3. "Full concurrency + bandwidth manager arbitration" is pipeline **steady-state** approximation: Ignores pipeline fill/drain latency (single iteration 4KB scale; sub-microsecond; negligible) and per-iteration handshake **rate** overhead (if CKE engine throughput is bottleneck, needs dedicated rate model; current folding / deletion both count as single; same metric as pre-fix; registered section 10.3 #21);
4. Full-expand fallback path (`mf > 0`) does not go through this resolution -- its per-iteration wait/record is real 4KB-granularity handshake; but inter-iteration overlap similarly serialized by queue chain; this path node count large; current sample not triggered; as known residual (section 10.3 #21).

**Alternative plans comparison**:

| Alternative | Idea | Pros | Cons | Conclusion |
|------|------|------|------|------|
| B. Launch-phase direct judgment (no tag / no identification) | `_emit_merged_loop` directly judges mid-body wait by slot structure (has data slot after it) and skips emitting its wait/pairing | Simplest implementation; no Task field expansion; judgment info (slot order) at hand; no heuristic | Loses handshake node and loop tracing in trace; skipping emission changes register pairing bookkeeping (loop-end wait absorbs prefetch set; timing harmless but behavior difference from checker widens); free set / sync scenarios need additional exclusion conditions | Feasible simplified variant; selectable if trace presentation not needed |
| C. Segmented pipeline modeling | Data task folded into K segments (8~16); inter-segment retains wait/record; explicitly models fill/drain | Includes pipeline fill effect; more fine-grained | Node count x K; K value no physical basis; in steady state differs from full concurrency only by first/last segment | Over-design; rejected |
| D. Simulator soft dependency edge | New edge type "pipeline edge" (child starts after parent completes X%) | Graph structure unchanged | Simulator semantics complicated; X has no physical basis; and serial chain structure (unit forks from wait) still needs graph construction layer modification | Rejected |

Conclusion: Adopt original plan (tag + feature chain + delete); per above step 2/3 implementation locations and safety condition strengthening (launch-phase record + pre-assembly pruning + mid-body judgment + consumer conservative branch).

**Implementation status (landed)**: `Task.loop_tag` field (task_model / trace args / dump `nodes[]` / input_builder loopback) + `DependencyGraph.remove_task()` (along with cleaning extra edges and notify pairing) + `_break_pipeline_chains()` (called at assembly loop start; counts `pipeline_chains_broken`; summary shows `pb=N`) + trace loop secondary-stream rename presentation (`export_trace`; see above). Verification: Sample ALLGATHER db cross-engine scenario makespan 2657.521 -> 2561.541 us (56 -> 54 tasks; pb=1; 9 units all fork reconnect to preceding skeleton node rwt22; item-by-item consistent with manual prototype); archive regression -- same-engine phase scenario (mv->mv both HBM) makespan unchanged (zero-error scenario; theoretical expectation); SQE-only archive pb=0 unaffected. `loop_tag` retained as task persistent attribute for subsequent performance simulation (such as per-loop bandwidth/rate modeling; section 10.3 #21) consumption.

### 6.6 Graph Construction Algorithm (Pseudocode)

```
expand_ccu(device_id, die_id, sqe_params, instr_space, dep_graph, reg_model):
    SQEs grouped into dispatch contexts by contiguous segments (section 6.1: non-contiguous = new dispatch; independent window)
    Same-queue contexts serial by dispatch order; each queue takes first context into active set
    while active set non-empty:
        progress = False
        for q in active set's each queue current context:        # cross-queue polling
            while ptr[q] not at window end:
                instr = decode(instr_space[ptr[q]])
                if instr is WAIT and reg_model.cke bits insufficient:
                    break                              # this queue pauses; rotate to other queue
                nodes = transform(instr, reg_model)   # section 6.3 mapping; LOAD only changes registers
                for node in nodes: attach to queue q chain tail
                if instr is SET_CKE: reg_model.seen_post registers
                if instr is WAIT:     pair with seen_post -> aggregated wait node (section 6.4)
                if instr is LOOP:     merge path (section 6.5 verification) or interpret loop body per iteration
                ptr[q] += 1; progress = True
            if ptr[q] at window end: current context complete; same-queue next context enters active set
        if not progress:
            # Stall = exists wait depending on remote CKE (cross-rank; invisible in single-rank graph)
            For stalled queue: generate remote Notify_Wait node (notify_id+mask; fixed delay);
                then replay rest of this instruction (local set / data movement); continue advancing
    SQE stream attach (section 7.4.1): Placeholder removed from SQE stream (task graph does not introduce placeholder nodes);
        entry edge (SQE predecessor -> queue skeleton head) / exit edge (mission completion set -> SQE successor)
        expressed via extra edges
    Assembly split (section 7.4.1): Each mission queue sequence -> skeleton (stays on queue chain) + async units
        (attached to #v virtual stream; stream_id changed to point to virtual stream); fork (predecessor skeleton -> unit head) /
        completion set join (skeleton tail + all unit tails -> SQE successor / same-queue next mission entry)
        extra edges connect
```

- **Cross-rank post-wait does not build edges, but retains synchronization task nodes** (single-rank graph construction; main difference from checker): checker's `GenerateRawCcuGraphsV3` (`ccu_task_transform_v3.cc:1275-1406`) lets multi-rank CCU graphs advance alternately; cross-rank posts mutually unlock; this plugin has only single-rank instruction data:
  - Instructions posting to remote CKE / remote register notify -> generate remote `Notify_Record` node (src=local, dst=INVALID -> fixed `remote_record_delay`; does not register local CKE bookkeeping; does not participate in pairing):
    - A5: SyncCKE's `rmtCKEId/locCKEMask`, SyncGSA/SyncXn's `setRmtCKEId/setRmtCKEMask`, TransLocMSToRmtMS's `setRmtCKE` (consistent with checker `AddPost` trigger condition);
    - A6: TransMem `dmaOpCode=5` write-after-notify and `0x1A` remote atomic count, SyncXnWtx `notifyValid=1` notify, SyncAtx atomic sync -- remote target type (Xn/CKE) resolved by peer address mapping (checker `GetXnAndTypeIdByAddr`); unobtainable in single rank; all approximated as remote record (mask takes low 16 bits of notify value for trace display only).
    SyncGSA/SyncXn (A5) and SyncXnWtx main write (A6) value sync is only remote register write (checker builds node/writes state on peer graph); local side does not generate task separately.
   - wait required bits cannot be satisfied by this rank's post (stall detection found) -> that wait depends on remote -> generate remote `Notify_Wait` node (fixed `remote_wait_delay`; not paired); and replay rest of instruction; unsatisfied wait within loop body similarly generates remote wait node then continues. Partially satisfied: Satisfied bits pair normally with local post (aggregated to 1 local wait node; section 6.4); unsatisfied bits generate one remote wait node. A6 independent Wait instruction (Xn value compare) unsatisfied same mechanism: Top-level stall -> stall resolution generates remote `Notify_Wait` (task id tag `xwt`) then passes through.
    - Task graph does not spontaneously generate any node: All nodes come from original db data or microcode instruction transformation (attach dependencies expressed via extra edges; see section 7.4.1).
- This rank's post-wait pairs normally: Multiple queues of same device, multiple dies' CKE all within local graph; build edges per section 6.4 rules.

### 6.7 Comparison with checker CCU Graph Construction Simplification

**CCU task rank/eid/direction semantics**:

- **Channel-class network tasks** (write/read): Local rank takes opDetails.rankId; peer rank via channel peer eid (reid) queries eid->rank index (section 4.8); if not found fills INVALID + `rank_unresolved` count. Equivalent to checker `GetRemoteRankByChannel` (`g_allRankChannelInfo`: channelId -> peer); but uses RankEidMap's communicator rank instead of checker's physical dstDeviceId.
- **Reduce tasks** (REDUCE microcode): MS is rank-local buffer (checker for all msIds always `GenSliceFromMs` + `AddLocalBatchReduce`); reduce always local HBM operation; src/dst both fill local rank; cross-rank data already billed in preceding Rmt* role movement; reduce does not repeat network.
- **Local tasks** (local movement Memcpy, local reduce Reduce_Inline): src/dst both fill local rank.
- **is_write**: UB tasks produced by CCU all explicitly carry (write=True / read=False); does not rely on simulator rank heuristic.
- **Local rank self-verification**: Channel leid <-> index-resolved rank if inconsistent with opDetails.rankId, use opDetails as authoritative and warn (eid numbering misalignment defense; such as deviceId and rankId offset exists).

| checker Step | light_performance | Reason |
|--------------|-------------------|------|
| Two-stage (raw graph -> queue parallelization -> reconnect) | Single-stage interpret execution + assembly-stage queue parallelization split (section 7.4.1: `ApplyCcuQueueParallelizationV3` simplified port; skeleton/unit split + fork/join with attach extra edges + `#v` virtual stream) | No verification; no need to retain modifiable raw intermediate form; parallelization completed in one pass at assembly stage (reuse granularity etc. differences see section 7.4.1) |
| Loop merge (BatchTransMem/BatchReduce, resource conflict check, producedMasks balancing) | **Equivalent implementation** (section 6.5: precondition verification + folded launch + fallback full expand; memory slice merge not needed -- simulation only cares about data size and timing) | Aligned with checker semantics to eliminate node bloat |
| SQE grouped into TaskCcuGraph (`IsContiguousCcuSqe` contiguous only merged + `CcuMissionKey` latest node append) | **Equivalent implementation** (section 6.1: contiguous segment split dispatch contexts; each context independent window replay + sequential args consumption; register/CKE shared across contexts; same-queue contexts serial by dispatch order) | Aligned with checker multiple dispatch (window wrap re-dispatch) semantics; previously merging into single context lost second dispatch data movement (section 10.3 #20 fixed) |
| GetRemoteRankByChannel (channelId -> dstDeviceId/dstDieId; mapped to rank via Communicator table) | Peer eid (reid) -> RankEidMap/ranktable index -> communicator rank (section 4.8/6.7) | estimate mode Communicator table only this rank one row; ccuChannels.rankId field is endpoint deviceId; unusable |
| AllRankParamRecorder full address space mapping (XN/CKE/PFE/CHANNEL/JETTY/MISSION/MS ...) + HBM data content + remote address -> register reverse lookup (`GetXnAndTypeIdByAddr`) | Retains only xn/cke/gsa/channel four classes; A6 Load/Store HBM data plane and remote target reverse lookup not modeled (section 6.4; `_MissionContextA6` deviation list) | Simulation only needs length and peer; data content / remote register number do not affect timing |
| A6 checker error abort scenarios (unsupported dmaOpCode/dstType, clearType!=1, clearMask!=0, LoopGroup config illegal, LoadX/StoreX indirect number out of range, etc.) | Downgraded to warn + skip data plane (sync semantics still execute) or skip instruction | Performance tool tolerates incomplete/abnormal data; keeps queue skeleton intact |
| CcuTraceInfo tracing, ccu_convert_config.json generation | Removed | Verification/rewrite tool specific |
| CheckAllPostMatch error | Downgraded to warn + diagnostic statistics | Accommodates single-rank data |
| Cross-rank post unlocks other's wait (multi-rank joint advance) | **Does not build cross-rank edges**: Cross-rank post/wait generates remote sync node (fixed delay); when peer is ready approximated by fixed delay | Single-rank graph construction (core difference) |

### 6.8 A5 and A6 Instruction Set Difference Overview

| Dimension | A5 (V1/Ascend950) | A6 (V2/Ascend960) |
|------|--------------------|--------------------|
| LOAD class | 7 instructions: Includes GSA family | 18 instructions: Unified X register + LoadX/StoreX/ClearX/Nop + arithmetic (Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt) |
| CTRL class | 5 instructions: Loop/LoopGroup/SetCKE/ClearCKE/Jmp | 7 instructions: + SetCKBit/Wait/Fence |
| TRANS class | 12 instructions: Mem <-> MS all combinations + SyncCKE/GSA/Xn | 7 instructions: Local 4 kinds + unified TransMem (dmaOpCode distinguishes) + SyncWtX/SyncAtX |
| REDUCE class | Add/Max/Min (msId list addressing; <=8 MS) | ReduceAdd/Max/Min (same structure; no embedded waitCKE) |
| Loop register | v1.loop{xnId, startInstrId, endInstrId}; loopCnt in Xn | v2.loop{xpId(context), xnId, xmId(iteration count), mode}; mode==1 conditional expansion (section 6.5) |
| LoopGroup parameters | Xn: LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41]; Xm: gsaOffset/MSOffset/ckeOffset | Xn: loopInsCnt:10/expandOffset:9/expandCnt:9; Xm: ckOffset:10/msOffset:11/gsaOffset:32; Xp: xnIdOffset:32 |
| Register number addressing | Payload field direct | `GetXnId`: When bit15=1, offset by loop xnIdOffset (section 6.4) |
| Embedded sync field | TRANS/REDUCE commonly carries waitCKE + setCKE + clearType | Local TRANS/REDUCE only setCKE (wait borne by independent SetCKBit/ClearCKE/Wait instructions) |
| Channel/length value | Channel is instruction immediate field | Channel/length both Xn register values (`Xn[xcId]`/`Xn[xlId]`) |

Both instruction set generations already landed as graph construction (A6 interpreter `_MissionContextA6` reuses A5's task launch / CKE pairing / advance / merge launch infrastructure; only overrides instruction decoding and loop organization); A6 instruction -> task type master table see section 11.3.3; established deviation list from checker see `_MissionContextA6` class docstring and section 6.7 comparison table.

---

## 7. Performance Simulation Module (Sub-Requirement 4; Incorporated Based on HCCLSIM Baseline)

> This chapter is the design description of `simulator/` subpackage (ported from `docs/light_performance/HCCLSIM/hccl_sim/`); interfaces follow HCCLSIM as authoritative baseline. Differences from old version and outstanding items see Chapter 10.

### 7.1 Overview

```python
sim = HcclSimulator(config, dep_graph, topo_resolver)
result = sim.run(trace_filepath=None, max_trace_events=0)   # -> dict
```

Simulator receives three parameters:

| Parameter | Type | Content | Builder |
|------|------|------|--------|
| `config` | `SimConfig` | Latency, bandwidth, utilization config | Caller |
| `dep_graph` | `DependencyGraph` | Task nodes + dependencies | `graph_builder` (db path) / `input_builder` (JSON path) |
| `topo_resolver` | `TopoResolver` | eid -> physical port resolution (ranktable.json) | Caller |

`run()` returns result dict (no longer returns bare makespan):

```python
{
    "makespan": float,        # end-to-end duration (seconds)
    "makespan_us": float,     # (microseconds)
    "trace_file": str|None,   # trace output path
    "trace_events": int,      # trace event count
    "sim_time_s": float,      # software runtime (seconds)
    "tasks": list[Task],      # all tasks (including start/end_time)
    "tasks_by_stream": dict,  # int stream id -> [Task] (chain order)
    "ports": dict,            # tuple(ports) -> NetworkPort
}
```

Simulator does not accept task sequence list; does not infer dependencies -- all obtained from `dep_graph`.

### 7.2 Parameter One: `config: SimConfig`

**Unit convention**: Internal latency = seconds, bandwidth = B/s; **JSON config units latency = us, bandwidth = GBps** (`from_dict`/`to_dict` auto-convert). Port identifier is `"die_id/port_number"` string (such as `"1/0"`); consistent with ranktable.json.

#### Field Definitions

| Field | Type | Default | Description |
|------|------|--------|------|
| `local_record_delay` | float | `1e-6` | Local record fixed duration; JSON value unit us |
| `remote_record_delay` | float | `1e-6` | Remote record fixed duration; JSON value unit us |
| `remote_wait_delay` | float | `1e-6` | Remote wait / unmatched local wait fixed duration; JSON value unit us |
| `port_bandwidth` | dict | `{}` | `"die/port"` -> bandwidth (B/s). One eid can correspond to multiple ports; configured per port; total bandwidth summed |
| `default_port_bandwidth` | float | `53.125e9` | Unconfigured port default bandwidth (425 Gbps = 53.125 GBps); JSON value unit GBps |
| `hbm_bandwidth` | float | `1.4e12` | HBM bandwidth; used for Reduce_Inline/Memcpy; JSON value unit GBps |
| `bw_profile_set` | BandwidthProfileSet? | `None` | Bandwidth utilization config set; None means utilization 100% (including multi-jetty scaling step `jetty_utilization_step`; see below) |

#### Port Bandwidth Config Format

```json
{"port_bandwidth": {"1/0": 53.125, "0/7": 53.125}}
```

#### Bandwidth Utilization Config (`bw_profile_set`)

```
BandwidthProfileSet
+-- profile_low:   BandwidthProfile  <- used when shared port concurrency < 8
+-- profile_high:  BandwidthProfile  <- used when shared port concurrency >= 8
+-- THRESHOLD: 8
```

Each `BandwidthProfile` is a data size(32B~128MB) -> utilization(0~1) mapping table; query rounds down to largest tier not exceeding that data size (floor match).

JSON format:
```json
{
  "profile_low":  {"32": 0.30, "64": 0.35, "...": 0.90, "134217728": 0.95},
  "profile_high": {"32": 0.20, "64": 0.25, "...": 0.84, "134217728": 0.85},
  "single_port_utilization": 1.0,
  "jetty_utilization_step": 0.02745
}
```

> File's actual key names are `multi_port_utilization_low` / `multi_port_utilization_high` (`from_dict` simultaneously compatible with `profile_low` / `profile_high` old key names).

#### Multi-Jetty Scaling (`jetty_utilization_step`)

`multi_port_utilization_low` calibrated by **one channel one jetty** conventional configuration. When `HCCL_UB_MULTI_CHANNEL_NUM=n`, one channel attaches n jetties; concurrent movement capability approximately linearly improved; lightweight simulation does not build jetty-level model (jetty only used for eid/protocol resolution; see section 5.5); instead converts jetty count into `profile_low` table's overall lift:

```
util'(size) = min(1.0, util(size) + (n - 1) x jetty_utilization_step)
```

- `n` taken from `opDetails.ubMultiChannelNum` (proxy level0 stub reads env var and lands in db; section 4.2); per-operator effective; when `n <= 1` (unconfigured/invalid falls back to 1), uses original table directly; no lift;
- `jetty_utilization_step` value range `[0.0109, 0.044]`; defaults to mid value `0.02745`; configured only by `bw_profile.json` (out-of-range prints WARNING then clamped to interval bounds);
- Acts only on `profile_low`: `profile_high` is high-concurrency tier (port already saturated by multi-path transfer; multi-jetty no extra gain); single port / `d2h` uses constant utilization (`single_port_utilization` / `d2h_utilization`); both not scaled;
- Entry: `SimConfig.for_jetty_num(n)` -> `BandwidthProfileSet.for_jetty_num(n)` (`bandwidth_profile.py`); pipeline per-operator generates one scaled config copy; debug can use `--jetty-num` to override db value (`run_perf_sim.py` / `python -m light_performance`).

### 7.3 Parameter Two: `dep_graph: DependencyGraph`

#### Four-Layer Content

```
DependencyGraph
+-- tasks: dict[str, Task]                       <- task nodes (with all parameters)
+-- stream_chains: dict[int, list[str]]          <- serial dependencies (int stream id)
+-- notify_matches: dict[str, str]               <- notify dependencies (wait_task_id -> record_task_id)
+-- extra_edges: dict[str, list[str]]            <- additional dependency edges (child_task_id -> [parent_task_id, ...])
                                                  <- sources: CCU queue parallelization fork/join edges and SQE attach edges (section 7.4.1)
```

#### Layer 1: Task Nodes (`Task`)

Each Task carries all parameters needed for execution (invalid values unified as `INVALID = 0xFFFFFFFF`; eid empty string; string attributes `""`):

| Field | Type | Applicable Task Types | Simulator Use |
|------|------|------------|-----------|
| `task_id` | str | All | Dependency graph index |
| `task_type` | TaskType | All | Determines latency model (internal classification; section 7.3) |
| `task_name` | str | Display use | trace/dump display name: profInfo.profTaskType op name (`_add_task` uniformly fills; section 11.4.1); when empty, `display_name` falls back `task_type.value` (CCU sub-task/CLI input) |
| `stream_id` | **int** | All | Serial chain belonging |
| `notify_id` | int? | Notify_Record/Wait/Write_With_Notify | Dependency pairing identifier (CCU path = cke id); Write_With_Notify only display (its notify faces peer; does not participate in pairing) |
| `mask` | int | Notify_Record/Wait | notify bitmask (CCU path): notify_id + mask uniquely identifies one notify action; wait node carries aggregated mask (one wait instruction all contributing posts' consumed bits ORed; section 6.4); SQE path and 0 = unfilled |
| `src_rank` / `dst_rank` | int? | notify/transfer | Local judgment (src==dst); read direction local judgment |
| `size` | int | Memcpy/Reduce_Inline/Ub_*/Write_With_Notify | Compute latency + query utilization |
| `port` | str/list? | (reserved field) | Port resolved in real time by eid; db path not filled |
| `src_eid` / `dst_eid` | str | Ub_*/Write_With_Notify | eid -> ranktable -> physical port |
| `transport_type` / `link_type` | str | Display use | msprof reconciliation dimensions (profInfo transportType/linkType names; section 11.4.2/11.4.3; `_add_task` uniformly fills; CCU sub-tasks not filled) |
| `data_type` | str | Display use | trace/report labeling |
| `protocol` | str | Display use | Channel CommProtocol name (section 11.4.2 note; empty when no channel info) |
| `jetty_id` | int? | Ub_*/Write_With_Notify | trace labeling |
| `is_write` | bool? | Network tasks | Billing direction and port query eid selection (section 10.3 #1) |
| `start_time` / `end_time` | float | All | Simulator fills (-1 = not executed) |

7 TaskTypes:

| TaskType | Latency Model | Description |
|----------|---------|------|
| `NOTIFY_RECORD` | Fixed delay (local also triggers Event) | Local = `is_local_notify`(src==dst); unpaired local record only fixed delay |
| `NOTIFY_WAIT` | Wait Event / fixed delay | matched -> wait record's Event; unmatched/remote -> fixed `remote_wait_delay` |
| `REDUCE_INLINE` | HBM dynamic shared | `size / hbm_bandwidth`; shares bandwidth with other HBM tasks |
| `MEMCPY` | HBM dynamic shared | Same as above (HBM copy; not network transfer) |
| `UB_INLINE_WRITE` | Fixed delay (same as Notify_Record: local/remote record_delay) | notify inline write; SQE graph construction path does not produce; only old input path (task.txt) uses |
| `UB_WRITE_OR_READ` | Port bandwidth shared (write/read two directions billed independently) | Network data send/receive; `is_write` explicitly carried by graph construction (SQE network tasks two directions uniformly mapped to this type) |
| `WRITE_WITH_NOTIFY` | Port write direction shared + post-write notify fixed delay | Network write + attached notify single task; one-to-one with db `WRITE_WITH_NOTIFY` (7); notify by remote record fixed delay (peer wait not in this db) |

#### Layer 2: Serial Dependencies (`stream_chains`)

```
int stream id -> [task_id_0, task_id_1, ..., task_id_n]
```

- List order = execution order: task[i] must complete before task[i+1] executes
- Each stream_id launches one simpy process; executes by chain order `yield`
- Different streams naturally parallel
- db path stream id addressing: SQE stream uses original streamId directly; **CCU queue skeleton chain belongs to its SQE original stream** (section 6.5; whichever stream the CCU-type SQE is on, expanded task is on that stream; only when original stream occupied by another die's queue does it allocate synthetic stream); virtual streams allocated from base `1<<20` (section 6.5; virtual stream logical name with `#v{n}` suffix); logical names retained via `stream_labels`; `GraphBuilder.stream_meta` on top gives real/ccu/virtual three classifications and virtual stream's parent_sid (section 7.4.1; points to its queue's stream id; normally SQE original stream); for report/trace/dump labeling

#### Layer 3: Notify Dependencies (`notify_matches`)

```
wait_task_id -> record_task_id
```

Explicit 1:1 pairing: Which wait waits for which record. When same notify_id has multiple records, only one pairs with wait; unpaired records take fixed delay.

#### Layer 4: Additional Dependency Edges (`extra_edges`)

```
child_task_id -> [parent_task_id, ...]
```

- Semantics: child must wait for **all** parents to complete before starting execution (pure timing constraint; does not carry record/wait event semantics -- event pairing still expressed by `notify_matches`).
- Current sources: CCU queue parallelization fork edges (predecessor skeleton -> unit head); mission completion set join edges (skeleton tail + unit tails -> SQE successor / same-queue next mission entry); CCU subgraph attach edges on SQE stream (entry/exit; see section 7.4.1); and aggregated wait node multi-parent edges (wait -> all contributing posts; section 6.4 -- first contributing post goes through `notify_matches`; others through this layer).
- Writer side `add_edge` auto-dedup (same child-parent pair only recorded once); simulator queries via `get_extra_parents` / `extra_parent_ids`.
- Acyclicity: Above edges always connect "forward" along original instruction order (skeleton before unit; unit before join point; placeholder position before-and-after attach; post registered before wait instruction that consumes it); graph construction direction guarantees acyclicity; included in `_drop_cyclic_matches` cycle detection fallback (section 5.4).

### 7.4 Simulator Consumption Flow

```python
def run(self, trace_filepath=None, max_trace_events=0):
    # 1. Retrieve all task nodes
    tasks = list(dep_graph.tasks.values())

    # 2. Create sync primitives: Only matched wait creates Event
    for wait_task_id in dep_graph.matched_wait_ids:
        notify_events[wait_task_id] = env.event()
    for parent_id in dep_graph.extra_parent_ids:  # all parent tasks of extra edges
        task_done_events[parent_id] = env.event() # must pre-build before stream processes start:
                                                  # parent completes first, child builds Event later -> permanent wait (section 7.4.1)

    # 3. HBM bandwidth manager (Reduce_Inline + Memcpy shared; utilization always 1.0)
    hbm_mgr = _BandwidthManager(env, config.hbm_bandwidth, empty utilization table)

    # 4. Each stream launches one simpy process; executes serially by chain order
    for task in chain:
        for pid in dep_graph.get_extra_parents(task.task_id):
            yield task_done_events[pid]           # extra edges (section 7.4.1); start_time excludes wait
        if NOTIFY_RECORD:
            yield timeout(calc_notify_record_delay(task, config))
            if local and matched: trigger matched wait's Event
        elif NOTIFY_WAIT:
            if local and matched: yield notify_events[task_id]   # wait Event
            else:              yield timeout(remote_wait_delay)
        elif REDUCE_INLINE / MEMCPY:
            yield hbm_mgr.send(task_id, size)                 # HBM dynamic shared
        elif UB_INLINE_WRITE / UB_WRITE_OR_READ / WRITE_WITH_NOTIFY:
            is_write = task.is_write  # explicit direction; old input path None falls back heuristic
            eid = src_eid if is_write else dst_eid
            ports = topo_resolver.get_ports_by_eid(eid)       # missing throws
            port = lookup/create NetworkPort(ports, total bandwidth)
            yield port.send(task_id, size, is_write)          # port bandwidth shared
            if WRITE_WITH_NOTIFY:
                yield timeout(calc_write_with_notify_delay(task, config))
                                                           # notify after write lands:
                                                           # fixed record delay
                                                           # (peer wait not in this db)
        task.end_time = env.now
        if task_id in task_done_events: task_done_events[task_id].succeed(env.now)
    # 5. makespan = max(end_time); assemble result dict (including trace export)
```

Latency formulas uniformly converged in `task_timing.py` single-point maintenance.

#### 7.4.1 CCU Queue Parallelization and Virtual Streams (extra edges)

**Problem** -- CCU queue chains produced by section 6.5/6.6 are task-by-task serial flows: All data movement, wait, set of same queue execute in order. On real hardware, queue's asynchronous movement units execute concurrently; serialization causes triple distortion:

| Distortion | Mechanism |
|------|------|
| makespan overestimate | Asynchronous movement separated by skeleton (wait/set) one-by-one; cannot overlap |
| Utilization tier distortion | Same-queue concurrent movement degrades to one-by-one execution; `_BandwidthManager` observed port concurrency always 1; always queries profile_low tier (section 7.5) |
| Port peak concurrency statistics distortion | `peak_concurrency` always 1; cannot reflect concurrent shared bandwidth behavior (section 8.2 port-level metrics) |

**checker reference** -- `ApplyCcuQueueParallelizationV3` (`ccu_task_transform_v3.cc:1052`): After raw graph construction, second-pass transformation; `CollectAsyncBoundRecordNodesV3` collects asynchronous movement nodes; `QueueIdReuseAllocatorV3` reuses virtual queue id by `[startNodeId, afterNode]` interval; fork/join edges reconnect to skeleton.

**Simplified port** (`ccu_graph_builder.py` assembly loop; completed in one pass at assembly stage; no raw intermediate form) -- splits each mission's queue node sequence:

1. **Unit identification**: `_QUEUE_DATA_TYPES` (`Memcpy`/`Ub_Inline_Write`/`Ub_Write_Or_Read`/`Reduce_Inline`) data tasks + immediately following consecutive `NOTIFY_RECORD` chain = one **async unit**; other nodes (wait / free set / remote sync nodes) are **skeleton**; stay on queue serial chain -- queue chain is the skeleton itself; no longer introduces any head/tail placeholder nodes.
2. **Fork edge**: `add_edge(unit head, preceding skeleton node)` -- unit must wait for queue to advance to its predecessor skeleton before starting (adjacent units share same fork parent node); unit appearing before skeleton (mission first node is data task) has no skeleton predecessor; its entry dependency expressed by attach edge instead.
3. **Completion set join**: Mission completion set = skeleton tail + all unit tails; via extra edges joins to same-queue next mission entry (skeleton head + pre-skeleton unit head; queue serial semantics; skeleton tail order already covered by queue chain serial) and SQE stream successor task (exit edge; added by graph_builder attach stage).
4. **Virtual stream reattach**: Each queue one set of counters/reuse pool (`vstream_seq`/`vstream_free`/`vstream_busy`); unit task's `stream_id` changed to point to `ccu_d{die}_q{queue}#v{n}` (same `1<<20` base space); unit chain registered into `result.virtual_chains` and `add_stream_chain` -- serial chain only constrains skeleton and unit internal task order.
5. **SQE stream attach** (`graph_builder._expand_ccu` consumes `result.attach`): CCU placeholder removed from SQE stream chain (task graph does not produce any placeholder/spontaneous nodes); entry edge = SQE predecessor task (including immediately preceding CCU mission completion set) -> skeleton head and pre-skeleton unit head; exit edge = mission completion set -> SQE successor task. All via extra edges; timing semantics: SQE predecessor (and preceding mission completion set) completes before subgraph can start; subgraph mission completion set all complete before SQE successor task can start. All-task-as-placeholder SQE stream removes empty chain.

**CKE timing preserved**: Split does not change `add_match` pairing -- whichever unit record falls into, whichever skeleton wait stays on, does not affect event pairing (wait via notify_events waits cross-stream). In actual test sample, wrap-up memcpy unit's fork parent wt28 happens to be on preceding read unit tail set's bookkeeping wait; cross-stream event + fork edge together guarantee "read complete -> wrap-up copy" original timing. **Exception**: Merged loop launch window's mid-body wait/record pairing is folded pseudo-serial barrier (per-iteration pipeline handshake folded to full barrier); resolved by section 6.5.1 feature chain before assembly pruned -- this section split based on pruned sequence; skeleton/unit/fork structure unaffected.

**Reuse rule differences from checker**:

| Dimension | checker (V3) | light_performance |
|------|--------------|-------------------|
| Reuse granularity | afterNode interval (join successor node) level | mission level (when entering new mission, preceding all released) |
| Intra-mission reuse | Allowed (DAG explicit edges; no pseudo-dependency) | Not reused (units within same mission concurrent to each other; under chain model, reusing same stream would introduce pseudo-serial) |
| Residual strictness | GRAPH_UNMATCHED error exit | Tolerant (performance tool; section 6.4 downgrade warn) |

Cross-mission reuse safety: Units reusing same virtual stream serial within that stream chain (later mission's unit chain after earlier mission's unit chain); time axis no overlap.

**Acyclicity guarantee**: fork (skeleton -> unit head) and completion set join edge (unit tail -> SQE successor / next mission entry) always connect "forward" along original instruction order; split does not change original order; `_drop_cyclic_matches` cycle detection simultaneously counts extra edges (section 5.4) as fallback; simulator has runtime deadlock detection (task `end_time<0` errors exit).

**Statistics and labeling**: `CcuBuildStats` carries parallelization (`parallel_units`/`virtual_streams`/`fork_edges`/`join_edges` -- join_edges now counts completion set join and SQE attach edges), remote sync (`local_posts`/`remote_posts`/`wait_matched`/`remote_waits`) etc. counts; summary row shows as `pu/vs/lp/rp/wm/rw`; `GraphBuilder.stream_meta` classifies by logical stream name -- `s{n}` -> real; containing `#` -> virtual (with parent_sid); others -> ccu; consumed by three places: report (`Stream count: N (including virtual M)`; virtual stream line labeled `vir_stream {n} ({label})`; n numbered from 0 by allocation order; over 64 streams folds detail); trace (virtual stream thread_name `vir_stream {n} ({label})` + task args `virtual_stream: true`); dump (streams[] with kind/virtual/parent_sid; extra_edges list; DOT purple dashed cluster + blue dotted extra edges; HTML badge + dotted arc). stream_meta default (simulator standalone CLI old input) not labeled; backward compatible.

**Quantified effect** (sample db REDUCE_SCATTER iter0; 66 tasks): Queue parallelization makes makespan 20,404.20 -> 5,149.36 us (~3.96x); subsequent two corrections further drop to **2,753.71 us**: (1) Remote MS reduce correction (section 6.3.2; reduce always local HBM) -- original second phase 128MB reduce billed as network read ~2526 us; corrected to HBM ~96 us; (2) Cross-rank sync node completion (section 6.6) -- 14 waits depending on remote set previously skipped (read unit launched early at t~=0); corrected to simulate by fixed remote delay. 10 async units reattached to 10 virtual streams (fork edges 10; attach/join edges generated by SQE neighbor and queue serial situation) -- first phase 8 units concurrent (7 network read units each ~2526 us; belonging to 7 independent ports `1/0`..`1/6` + 1 local memcpy ~96 us); then reduce ~96 us and wrap-up memcpy ~96 us serial join. Note single-port read peak concurrency still 1 -- benefit comes from **cross-port overlap**, not single-port concurrency improvement.

### 7.5 Bandwidth Sharing Mechanism (`_BandwidthManager`)

#### Generic Manager (Port and HBM Reuse Same Model)

- **Single-manager-process model**: When new transfer joins, `interrupt` manager process; manager process evenly divides bandwidth by current concurrency N: `bw_per = total_bandwidth / N`.
- **Utilization query basis is `min_remaining`**: Same group of concurrent tasks queries utilization table by "minimum remaining data size" (concurrency <8 uses profile_low; >=8 uses profile_high); within group same basis.
- **profile_low table already scaled by jetty count**: Before query, pipeline already used `SimConfig.for_jetty_num(opDetails.ubMultiChannelNum)` to generate config copy; multi-jetty (`HCCL_UB_MULTI_CHANNEL_NUM>1`) table values overall lifted; manager itself unaware of jetty (section 7.2 "Multi-jetty scaling").
- Effective bandwidth = `(total_bandwidth / N) x utilization(min_remaining, N)`; compute earliest completion time; timeout settles progress; recalculate on join/complete.
- **HBM dynamic sharing**: `Reduce_Inline` and `Memcpy` share one HBM manager (utilization always 1.0) -- when multiple HBM tasks concurrent, evenly share HBM bandwidth (different from old version's per-task independent `size / hbm_bandwidth`; see section 10.3).

#### Dual-Direction Independent

Each `NetworkPort` contains write and read two independent managers; bandwidth independent non-interfering: When configuring port bandwidth B, write direction can use B; simultaneously read direction can also use B.

#### Multi-Port Aggregation

When one eid corresponds to multiple physical ports, `port_key = tuple(port_ids)` as logical port; total bandwidth = sum of each port's bandwidth (`get_ports_total_bandwidth`).

### 7.6 Parameter Three: `topo_resolver: TopoResolver`

- Input `ranktable.json` (`TopoResolver(ranktable_path=...)`); provides **eid (16B hex string) -> physical port list (`["die/port", ...]`)** resolution.
- Transfer task routing: Write direction queries `task.src_eid`; read direction queries `task.dst_eid`.
- **When eid cannot find port, directly throws** (no jettyId fallback; no default port) -- therefore db path overall skips simulation when "network tasks exist but no ranktable" (section 3.3); db graph construction counts and warns on unresolvable eid tasks (section 5.5).
- Port bandwidth queries `SimConfig.port_bandwidth` per resolved port and sums.

### 7.7 Build Methods and CLI

All input methods uniformly build `(config, dep_graph)`:

| Input Source | Build Function | Return | Status |
|--------|---------|------|------|
| **hccl-vm db (new this iteration)** | `graph_builder.build(group)` | `(config, dep_graph)` | Chapters 5, 6; main input |
| JSON task file | `load_dep_graph(filepath)` + `load_config(...)` | `(config, dep_graph)` | Retained (hccl_sim compatible) |
| task.txt / algorithm-generated etc. old input sources | -- | -- | **Removed with HCCLSIM version** (see section 10.3) |

Build flow unified to:
1. Create `Task` node; call `dep_graph.add_task(task)`
2. By in-stream task order call `dep_graph.add_stream_chain(stream_id, task_ids)`
3. By notify pairing result call `dep_graph.add_match(wait_task_id, record_task_id)`

CLI entry:

```bash
# db input (light_performance main entry; see section 3.3 for details)
python3 -m light_performance --db data/hccl_vm_data.db \
    --config sim_config.json --bw-profile bw_profile.json --ranktable ranktable.json -v

# One-click task graph dump (repository root)
python3 dump_graph.py data/hccl_vm_data.db --html

# hccl_sim compatible entry (JSON task file)
python -m simulator --tasks tasks.json --config sim_config.json \
    --bw-profile bw_profile.json --ranktable ranktable.json -v --trace trace.json
```

`--config` overrides each mode's default latency/bandwidth (units us/GBps); `--bw-profile` independently loads utilization config; report and trace generated by `report.py`'s `generate_report` / `export_trace` from `run()` result dict.

trace file is Chrome Trace Event format: `ts`/`dur` value units **us** (format specification convention); file top-level carries `"displayTimeUnit": "ns"` specifying viewer (chrome://tracing etc.) Start/Wall Duration **display** unit as ns (default ms).

`export_trace` display adaptation conventions:

- **tid sequential numbering**: Thread rows mapped by stream id ascending to tid `0,1,2,...` (original streamId 2/6..13 directly as tid; string-sorting viewers would place 10..13 before 2); real stream id retained in thread_name label and `args.stream_id`.
- **Endpoint rounding**: `ts`/`end` each rounded then subtract to get `dur` -- `ts` and `dur` each independently rounded produces 0.001us pseudo-overlap; same-tid overlap slices nested by viewer as sub-rows (line-wrapped display).
- **Virtual stream labeling** (section 7.4.1): db path via `export_trace(..., stream_meta=...)` passes stream meta info; virtual stream thread_name changed to `vir_stream {n} ({label})` (n by sid ascending i.e. allocation order numbered from 0; escapes 2^20-base large-value stream numbers; real stream number see in-line task `args.stream_id`); its task args appends `virtual_stream: true`; real/ccu stream names unchanged (ccu queue chain's logical name only appears in text report line). stream_meta default (simulator standalone CLI old input) not labeled; backward compatible.
- **notify task args carry mask**: CCU path's `Notify_Record`/`Notify_Wait` task args append `mask: "0x..."` (notify_id + mask uniquely identifies one notify action; section 7.3); SQE path and mask=0 does not output this field.
- **Operator summary info first row (op_summary)**: pipeline per-operator calls `export_trace(..., op_summary=...)`; each operator (multi-operator merged export per pid process group) uses tid 1 + `thread_sort_index` 0 to retain `op_summary` first row; task row tid and sort_index deferred (presentation layer only; tid avoids 0 -- some viewers such as Perfetto treat tid 0 as idle thread; `thread_sort_index` is Chrome Trace standard row order control); summary event placed at front of event array; triple guarantee op_summary always first row. First row places one `X` summary slice spanning operator start to end (start time and duration are slice's own ts/dur; takes earliest task start to latest end); row label `"<op_name> | <algorithm_name>"`; args carry: `algorithm_name` (algorithm selection result; opDetails.algorithmName log truncation backfill (DFX main path; section 4.2 note)), `data_type`/`data_count` (element count; V2 layout operators is send/receive two-side element count and type; others is count and single-side type), `task_count` (post-graph-construction total task count; CCU mode is microcode instruction translated to task graph count); and op_type/comm_id/op_iter/rank/rank_size/input_bytes/output_bytes (opMemInfo's inputSize/outputSize; computed from business API parameters at collection time) context; summary events not counted in `max_events`.

### 7.8 Alignment Adaptation Completion Status

Old version (this repo pre-modification) differences from HCCLSIM baseline digested in this alignment:

| # | Item | Handling |
|---|----|------|
| 1 | Task model 6 types -> new 6 types (local/remote record/wait merged to Notify_*; LOCAL_COPY split to Memcpy/Reduce_Inline; network transfer changed to Ub_*; later with db new WRITE_WITH_NOTIFY expanded to 7 types; section 10.3 #19) | `task_model.py` replaced per HCCLSIM |
| 2 | stream_id str -> **int**; notify_id str -> **int** (INVALID=0xFFFFFFFF) | `dependency_graph.py` / graph construction side synced |
| 3 | Port specified by `port` int field -> **eid real-time resolution** (write queries src_eid / read queries dst_eid) | `topo_resolver.py` + graph construction side fills eid |
| 4 | Unit system: JSON latency s -> **us**; bandwidth B/s -> **GBps**; port key int -> **"die/port"** string | `config.py` from_dict converts |
| 5 | HBM latency `size/bw` independent -> **dynamic shared** (Reduce_Inline+Memcpy share manager) | `_BandwidthManager` reused |
| 6 | `run()` returns makespan -> **result dict**; simulator two params -> **three params** (+topo_resolver) | `simulator.py` |
| 7 | Latency formulas scattered -> `task_timing.py` single point | Incorporated |

Outstanding unmatched items (including `src_rank==0` hard-code, eid no fallback, broadcast pairing only triggers first wait, etc.) uniformly moved to Chapter 10 tracking.

---

## 8. Performance Profiling Output (Sub-Requirement 5; Conceptual)

### 8.1 Output Location and Files

```
<installRoot>/plugin/light_performance/data/<timestamp>/
+-- perf_profiling.json      # Main artifact: performance metrics + task graph summary
+-- trace_<commId>_<opIter>.json   # (optional) Chrome tracing; chrome://tracing visualization
+-- light_performance.log    # Runtime log
```

- Directory naming: `hccl-vm plugin run @estimate` one-shot execution channel does not pass data_id (section 3.2); uses run timestamp as substitute; standalone debug mode (run_perf_sim.py) specifies output path by `-o`.
- Output fixed in plugin data directory not `data/`: host clears `data/` on each start (`src/main.cc:32`); artifacts under plugin directory preserved across sessions; `archive.sh` archive only covers logs/ and data/; plugin artifacts not included in auto-archive (section 10.3 #15).

### 8.2 Metric Definitions (Draft)

| Level | Metric | Source |
|------|------|------|
| Operator | opType/opIter/rankSize/dataSize, makespan, software simulation duration | opDetails + `sim.run()` result dict |
| Operator | Operator summary info first row (algorithm_name/data_type/data_count; start/end and duration carried by slice ts/dur; section 7.5 op_summary) | opDetails.algorithmName (DFX/log dual-path truncation) + simulation task timeline |
| Operator | Total transfer volume, effective bandwidth utilization (total transfer / (makespan x port bandwidth)) | Summarize result["tasks"] |
| Stream | Each stream start/end time, duration, task count + virtual stream label (section 7.4.1) | result["tasks_by_stream"] + stream_meta |
| Graph | Async unit count, virtual stream count, fork edge count, completion set / attach edge count (section 7.4.1); local/remote post count (lp/rp); local wait paired count (wm); remote wait count (rw) | CcuBuildStats (parallel_units / virtual_streams / fork_edges / join_edges / local_posts / remote_posts / wait_matched / remote_waits; summary row `lp/rp/wm/rw`) |
| Port | Each port peak concurrency, write/read transfer volume and average bandwidth | result["ports"] (NetworkPort) |
| Task | Per-task start/end/duration; transfer task average bandwidth | result["tasks"] (existing trace logic) |

### 8.3 perf_profiling.json Format Draft

```json
{
  "meta": {"data_id": "1234", "db": "...", "sim_config": {...}, "generated_at": "..."},
  "ops": [
    {
      "comm_id": 1, "op_iter": 0, "op_type": 7, "op_name": "REDUCE_SCATTER",
      "rank_size": 4, "data_size": 33554432,
      "makespan_us": 15230.5,
      "total_bytes": 522190080,
      "streams": [{"stream_id": 2, "label": "s2", "kind": "real", "start_us": 0.0, "end_us": 15230.5, "task_count": 30}],
      "ports": [{"ports": ["1/0"], "write_bytes": 272629760, "read_bytes": 0,
                 "peak_concurrency": 3, "avg_bw_Gbps": 143.2}],
      "graph": {"task_count": 40, "stream_count": 5, "notify_match_count": 44,
                "cross_rank_notify": 40,
                "virtual_stream_count": 10, "extra_edge_count": 20,
                "dump_files": ["graph_comm1_iter0_dev1.dot"],
                "trace_file": "trace_1_0.json"}
    }
  ]
}
```

> `cross_rank_notify` is retained cross-rank notify node count (diagnostic info): These nodes graph normally by stream order; take fixed remote delay; but do not participate in local pairing (no cross-rank record-wait edge); this count used to troubleshoot data/classification anomalies (such as local notify misjudged as cross-rank). `kind`/`virtual_stream_count`/`extra_edge_count` are CCU queue parallelization (section 7.4.1) new fields: stream classification real/ccu/virtual; virtual stream count and extra edge (fork/join/attach) count.

---

## 9. Appendix

### 9.1 checker Graph Construction Module -> light_performance Graph Construction Module Master Comparison

| Aspect | checker (C++) | light_performance (Python) |
|------|---------------|---------------------------|
| Objective | Correctness verification DAG (mainStart reachable; supports various checks) | Simulation DependencyGraph (three layers: tasks / stream_chains / notify_matches) |
| Input | db (via StorageManager/sim::Query) | Same db (sqlite3 direct read + struct decode) |
| Graph construction scope | **Multi-rank joint graph construction** (all rank tasks in same DAG) | **Single-rank graph construction** (cross-rank pairing edges not built; remote sync tasks retained) |
| SQE translation | TaskMetaTranslatorV3 (including MemSlice address resolution) | Section 5.2 mapping table (only takes len/jettyId/notify fields) |
| WRITE_WITH_NOTIFY consumption | Decode boundary split: `DecodeAndExpandTaskMeta` expands to [write, NOTIFY_RECORD] two sub-tasks | **Not split**: Single task mapping `Write_With_Notify` (section 5.2); one-to-one with db task; trace/dump/statistics granularity consistent |
| Dependency edges | stream chain + local/cross-rank notify edges (queue simulation) | stream chain + only local notify first-come-first-served (two-pass scan) + extra edges (CCU fork/join/attach; section 7.4.1) |
| CCU | Two-stage expansion + queue parallelization + loop merge + CKE bookkeeping | Single-stage interpret execution + assembly-stage queue parallelization (section 7.4.1) + loop merge (section 6.5) + CKE bookkeeping (Chapter 6) |
| Post-verification processing | Memory conflict / semantic / sync conflict / reachability / single-task check | None |
| Output | Verification conclusion + dump | (SimConfig, DependencyGraph) -> makespan -> profiling json |

### 9.2 Glossary

| Term | Description |
|------|------|
| SQE | Submit Queue Element; AICPU/SDMA-side submitted task unit; i.e., one row of opTask table |
| WQE | Work Queue Element (UB/DB mode); not involved this iteration |
| notify | AICPU-side event synchronization primitive; record sets / wait waits; paired by (recordDev, waitDev, notifyId) |
| CKE | CCU-side event synchronization register (bitmap); post sets / wait waits; paired by (waitDev, dieId, ckeId, mask) |
| jetty | Network card (communication engine) port number; determines memcpy's bandwidth attribution |
| die | Chip die; CCU microcode space and CKE both organized by (device, die) |
| mission | CCU one task dispatch (instStartId/instCnt define instruction window; args[13] pass parameters) |
| makespan | Task graph end-to-end duration (latest task end time) |
| Main stream / secondary stream | opDetails.streamId marked operator main stream; others are secondary streams (head WAIT tail RECORD structure) |

---

## 10. HCCLSIM Alignment Outstanding Items and To-Match List

> This chapter records **field gaps, task type mismatches, and other conflict points** confirmed during this "db graph construction path + design document" alignment to HCCLSIM baseline. Current implementations all landed per "default fill invalid value / approximate mapping"; retained here for subsequent item-by-item digestion jointly with collection side and simulation module.

### 10.1 Task Field Availability Table (db Graph Construction Path)

New model Task fields' source in db graph construction path and default values (invalid value convention: int field `INVALID=0xFFFFFFFF`; eid empty string; other strings `""`):

| Task Field | db Source | Availability | Default/Approximate Value | Subsequent Match Path |
|-----------|--------|--------|------------|-------------|
| `task_id` / `stream_id` | Global order / streamId+synthetic base | Available | -- | -- |
| `notify_id` | opTaskMeta `NotifyTask.notifyId` | Available | -- | -- |
| `size` | MEM_CPY `len` / REDUCE `dataCount` / CCU length register | Available | -- | -- |
| `jetty_id` | SQE table column / ccuChannels.jettyId array | Available | INVALID (non-network tasks) | -- |
| `link_type` | opTaskMeta `HcclTaskProfInfo.linkType` (v3; enum name mapping section 11.4.2) | Available (v3) | "" (v2 old BLOB no profInfo displays `Unknown`) | -- (section 11.4 already consumed) |
| `protocol` | ccuChannels.protocol (CommProtocol enum name mapping) | Available | "" (when no channel info; trace displays `N/A`) | -- (migrated from `link_type`; section 11.1 plan a) |
| `src_eid` (local; write) | SQE `lcEid` (v2 collection new; section 4.4.1) prioritized; default jettyId -> this device outbound channel `leid` derivation | Available (v2) / approximate (fallback) | empty -> any local channel leid fallback | -- (v2 already supplemented explicit field) |
| `dst_eid` (peer) | Channel `reid` prioritized; fallback SQE `rmEid` | Approximate | "" (tasks without channel info) | Collection side supplements peer explicit eid |
| `src_rank` / `dst_rank` | taskData endpoint rankId (carried in deviceId field; section 4.4.1 rank field rollback) directly consumed: write=src local/dst peer, read=src peer/dst local; CCU/unresolved via eid->rank index fallback (section 4.8) | Available / index fallback | When both explicit value and index unresolved, INVALID + `rank_unresolved` count | -- (section 5.2/5.5/6.7) |
| `transport_type` | opTaskMeta `HcclTaskProfInfo.transportType` (v3; enum name mapping section 11.4.3) | Available (v3) | "" (v2 old BLOB no profInfo displays `Unknown`) | -- (section 11.4 already consumed) |
| `data_type` | REDUCE task-level `dataType` (v2 new; enum name mapping int8..bfp16); MEM_CPY task-level no source | Partially available (v2) | REDUCE fills enum name; MEM_CPY "" | MEM_CPY collection side supplement |
| `port` | (reserved field) | Not filled | None | Simulator resolves port in real time by eid (section 7.6) |

**Instruction-level gaps (CCU microcode expansion)**: Channel-class network task peer rank already resolved by eid->rank index (section 4.8/6.7); reduce tasks always local (MS is rank-local buffer; section 6.3.2); no peer eid need. UB tasks produced by CCU all explicitly carry `is_write` (section 6.7); direction determination does not rely on rank value.

### 10.2 Task Type Mismatch Table

| db taskType / Old Type | New Model TaskType | Mismatch Handling |
|---------------------|----------------|-----------|
| SYNC_STREAM (taskType=6) | **No corresponding type** | Skip; do not generate node (host-side stream sync point; not device-side task; section 5.6); statistics `skipped_sync` |
| AIV_GRAPH (taskType=5) | None | Skip + warn (old and new models consistent; both unsupported) |
| WRITE_WITH_NOTIFY (taskType=7; db new) | `WRITE_WITH_NOTIFY` (new; 7th kind) | Single task mapping (**not split** into write + NOTIFY_RECORD): Write always write direction goes through port bandwidth shared; attached notify by fixed record delay (section 5.2; section 10.3 #19 supported) |
| Old `REMOTE_RECORD` / `REMOTE_WAIT` | Merged into `NOTIFY_RECORD` / `NOTIFY_WAIT` (src != dst expresses remote) | db single-rank path's cross-rank notify generates remote node by `src != dst` (section 5.2; retains node; not paired) |
| Old `MEMCPY` (network transfer) | Semantics changed: New `Memcpy` = HBM copy | Network transfer changed to `Ub_Write_Or_Read` (write/read direction distinguished); `Ub_Inline_Write` retains only notify inline write semantics (fixed delay) |
| Old `LOCAL_COPY` | Split to `Memcpy` (pure copy) / `Reduce_Inline` (reduce) | REDUCE not network -> Reduce_Inline |
| Old `MemcpyDirection` enum (direction field) | Deleted | Direction expressed by task type + rank/eid value (section 5.5) |
| Old `sync_stream_delay` config item | Removed along with type gap cancellation | Merged into local_record_delay reuse |

### 10.3 Other Conflict Points (Subsequent Match)

1. ~~**`src_rank == 0` judge local hard-code** (`simulator.py` TRANSFER branch): Read-direction task if peer happens to be rank 0 and explicitly filled would be misjudged as write.~~ **Fixed**: Task adds explicit `is_write` field (db graph construction path fills; task.txt old path None falls back heuristic); simulator prioritizes consuming; taskData's peer rankId (carried in deviceId field) simultaneously directly filled into src/dst_rank (section 5.2/5.5).
2. **TopoResolver no fallback**: eid not found in ranktable directly throws (no jettyId -> port mapping; no default port). db path avoids by "network tasks exist and no ranktable -> skip simulation only dump" (section 3.3); future can consider connecting ccuChannels/JettyMaps to build fallback mapping.
3. **Broadcast pairing only triggers first wait** (matched record's `break`): When multiple waits pair with same record, subsequent waits' Event not triggered; end_time remains -1 (task silently lost).
4. **`is_matched_record` linear scan**: `dependency_graph.py` does O(R x M) traversal over matches; large-scale db (tens of thousands of tasks) has performance risk; recommend maintaining record -> wait reverse index (simultaneously solves #3).
5. **SimConfig unit system and port key change**: JSON latency s -> **us**; bandwidth B/s -> **GBps**; port key int -> **"die/port"** string; old config files incompatible (this iteration already synced sample `src/plugin/light_performance/sim_config.json`).
6. **HBM dynamic sharing semantics change**: Old version per-task independent `size / hbm_bandwidth`; new version when Reduce_Inline+Memcpy concurrent evenly shares HBM bandwidth (utilization always 1.0 manager); result metric differs.
7. **Utilization query basis `min_remaining`**: Same-group concurrent tasks query utilization tier by minimum remaining data size; different from old version "each task queries by own data_size" -- small tasks within group drag down large tasks' utilization tier.
8. **Old input sources removed**: task.txt parsing, AllToAll generator, built-in demo removed from input_builder with HCCLSIM version; log replay debug requires separate restoration if needed.
9. **TopoResolver no net_type filter**: When same eid has multiple network type entries, no priority rule.
10. **profiling json not implemented** (sub-requirement 5): Current output is task graph dump (DOT/JSON/HTML) + text report + Chrome trace; perf_profiling.json format see Chapter 8 draft.
11. ~~**local_rank fallback convention**: When opDetails.rankId missing, falls back by `deviceId-1` (following task.txt path convention); deviceId and rank numbering no necessary relationship; ranktable can refine.~~ Peer rank already resolved by eid->rank index (RankEidMap main path + ranktable fallback; section 4.8); no longer relies on numbering convention; local still uses opDetails.rankId as authoritative (channel leid cross-verification; section 6.7).
12. ~~**CCU queue parallelism not modeled**: CCU queue chain is independent synthetic stream executing in parallel; real hardware queue concurrency resource constraint not modeled (checker side has queue parallelization transformation; this plugin section 6.7 already simplified).~~ **Queue parallelization implemented** (section 7.4.1: assembly-stage skeleton/unit split + fork/join/attach extra edges + `#v` virtual stream cross-mission reuse). Still not modeled: Real hardware queue concurrency resource upper limit -- virtual stream count not limited (checker V3 similarly not limited); future can cap by chip generation.
13. **@estimate one-shot channel does not pass data_id**: `plugin run @estimate` special-case branch directly executes entry (`cmd_base_utils.cc:626-663`); does not go through generic message path's `DumpData` (whose generated `data_id` currently fixed value "1234"); therefore output directory named by run timestamp (section 8.1); if switching to resident model in future, can align data_id semantics.
14. **`plugin install @estimate` semantics degradation**: install goes through `RegisterPlugin` resident launch (fork+pipe); main.py one-shot execution completes then exits after stdin EOF; MonitorThread recycles -- behavior harmless but inconsistent with install's resident semantics; if truly resident needed, must implement stdin command loop (section 3.2 evolution direction).
15. **Plugin artifacts not included in auto-archive**: `archive.sh` only archives `logs/` and `data/`; `plugin/light_performance/data/` perf_profiling.json etc. artifacts do not enter archive directory (benefit: not affected by start's `remove_all("data")`); if archive needed, must extend archive.sh.
16. **simpy deployment dependency**: Target environment needs simpy preinstalled (not distributed with hccl_vm_install offline wheel); main.py at startup import failure should give clear error and install hint (section 3.4).
17. ~~**db_reader.py not synced to v3 collection layout**: C side `HcclTaskMetaData` already extended to 188B (after taskType adds `HcclTaskProfInfo` 3 bytes; section 4.4.1); `db_reader.py` still decodes by 185B (`TASK_META_LEN=185`; `<BQIQQI16s16s`) -- length gate `len(blob) >= 185` cannot block 188B BLOB; reading v3 collected db would **fields overall misaligned** (starting from commId). Need sync to 188B/`<BBBBQIQQI16s16s` and decode profInfo three fields (value range/names see section 11.4).~~ **Synced**: `db_reader.py` auto-identifies v3 (188B/`<BBBBQIQQI16s16s`; profInfo three fields into `HcclTaskMetaData`) and old v2 (185B) layouts by BLOB length; see section 4.4.1 implementation status.
18. ~~**JSON task type not consuming profInfo** (remainder narrowed): trace args' `transport_type`/`link_type` already switched to profInfo driven; CommProtocol migrated to new field `protocol` (section 11.4.2/11.4.3; section 11.1 plan a landed); **remainder** is `task_type` field -- currently still derived by TaskType enum + jettyId/direction condition (section 11.2); has **semantic misalignment** with collection side profInfo's hcomm op name (such as `Ub_Inline_Write` bi-directionally refers to different tasks; section 11.4.4); after switching to `profInfo.profTaskType` driven can one-to-one correspond with msprof timeline op name; target state comparison (including each value's correspondence with simulation TaskType) see section 11.4.1.~~ **Landed (all remainder wrap-up)**: `Task` adds `task_name` display name (`task_model.py`; `display_name` = `task_name or task_type.value`); `graph_builder._add_task` uniformly injects `prof_task_type_name(profInfo.profTaskType)`; trace event name / `cat` (`report.py export_trace`); `args.task_type` (`to_trace_args`); verbose detail and dump (DOT/JSON/HTML legend) all switched to op name. Internal `TaskType` retired to latency model classification (section 11.4.5 #3); latency/pairing/billing zero changes. v2 old BLOB (profInfo=0xFF) displays `Unknown` (same metric as link/transport); CCU microcode sub-task has no profInfo; falls back internal name (section 11.3). Misalignment elimination comparison see section 11.4.4 implementation record.
19. ~~**WRITE_WITH_NOTIFY (taskType=7) decoding missing**: `db_reader.py`'s `HccLTaskMetaType` IntEnum only 0~6; `decode_op_task_meta` encountering 7 directly raises and discards (counted as warning).~~ **Supported (single task mapping)**: `db_reader.py` decodes `WriteWithNotifyTask` (section 4.4.1) and single-task transparently passes; `graph_builder` maps to `WRITE_WITH_NOTIFY` TaskType (trace display name `Write_With_Notify`; section 5.2); **one-to-one** with db task; latency = port write direction bandwidth shared + post-write notify fixed record delay (section 7.3/7.4). **Does not adopt split strategy** (checker/runner side's "split into [write, NOTIFY_RECORD] two sub-tasks"; and this plugin's early decode boundary split both deprecated): Split scatters task granularity; makes trace/dump/statistics unable to one-to-one correspond with db tasks. Collection side profInfo=TASK_WRITE_(REDUCE_)WITH_NOTIFY (op name `Write_With_Notify` / `Reduce_With_Notify`; hcomm source latter misspelled `Reuce_With_Notify`; see section 11.4.5); both profInfo values mapped to same `Write_With_Notify` task (isReduce only affects data_type display).
20. ~~**Same mission multiple dispatch SQE merged to single context**: CCU expansion merges all SQEs of same `(dieId, missionId, queue)` into one dispatch context (single instruction window [first instStartId, end); single SQE args consumption) -- does not satisfy checker `IsContiguousCcuSqe`'s "contiguous only merged" constraint (`task_meta_translator_v3.cc:213-231`). Typical victim scenario: Instruction window wrap re-dispatch (head SQE 13 Load SqeArgs + body SQE dispatched twice; second time uses new args); under old implementation, second dispatch: (1) Window start counted by head SQE; end-segment JMP crosses entire data movement instructions; (2) SQE args already exhausted by first consumption; subsequent Load takes 0 value (GSA addresses all 0); (3) Loop counter/CKE residual first state -- second dispatch's Ub/Memcpy/local post overall lost; network data volume only 3/10 of checker's.~~ **Fixed (contiguous segment split)**: Splits by `instStartId`/`instCnt` continuity into multiple dispatch contexts (section 6.1); each context independent window replay (including Load SqeArgs header) + independent sequential args consumption (`GetSqe` semantics); register/CKE bitmap shared across contexts (`AllRankParamRecorder` semantics); same-queue contexts serial by dispatch order; multi-segment task id disambiguated by `_s{seq}` suffix (`ccu_graph_builder.expand_ccu_graph`). After fix, aligned with checker single-pass graph stream-by-stream: record local/remote 54/54; wait local/remote 27/36; Ub 39/39; Memcpy 6/6; network+local data volume 1088MB (2333 verification environment ALLGATHER dev1). Same root-cause **wait node launch deviation** also fixed: Old implementation split by "per contributing post one wait node" (one wait instruction can produce N nodes; inconsistent with checker `GenWaitNode`'s "per instruction at most 1 local + 1 remote node"; this environment shows local wait 48 vs 27); now aggregated to single node (mask = all contributing posts' bits ORed; first contributing post goes through notify pairing; others via extra edges; section 6.4); after fix local wait 27/27 aligned.
21. **Merged loop folded pseudo-serial residual gap**: Section 6.5.1 feature chain resolution covers merged path's mid-body wait/record (folded pseudo-serial barrier); but two residuals: (1) Full-expand fallback path (`mf > 0`)'s per-iteration wait/record is real 4KB granularity handshake; its inter-iteration overlap similarly serialized by queue chain (node count large; current sample not triggered; not handled for now); (2) Per-iteration handshake **rate** overhead not modeled -- folding and resolution both count as single (if CKE engine throughput is bottleneck, needs dedicated rate model such as ns/time x 32768 times scale evaluation).

---

## 11. db Task Type and JSON Artifact Task Type Mapping Master Table

> This chapter is the type mapping quick reference; answers three questions: (1) Tasks in db database (`opTask_P_<pid>` table taskType column) ultimately convert to which task types in JSON artifact (section 11.1, 11.2); (2) CCU mode microcode instructions and JSON task type correspondence (section 11.3); (3) Collection-side profiling reconciliation info (`HcclTaskProfInfo`) and JSON display type correspondence and target state (section 11.4). Implementation basis: `src/plugin/light_performance/graph_builder.py` (SQE mapping; `_map_tasks`/`_map_transfer`); `src/plugin/light_performance/ccu_graph_builder.py` (CCU instruction mapping; `transform` family); `src/plugin/light_performance/simulator/task_model.py` (TaskType enum). Same source as section 5.2 (SQE mapping + latency model) and section 6.3 (instruction major class mapping); here reorganized by "db -> JSON" correspondence; reverse index supplemented.
>
> **profInfo background**: From v3 collection onward, db task's opTaskMeta BLOB carries `HcclTaskProfInfo` sub-structure (section 4.4.1; single-source definition at `include/sim_profiling_defs.h`; checker/runner forwards as opaque whole); three fields `profTaskType`/`linkType`/`transportType` value range and name one-to-one aligned with hcomm `TaskParamType`/`DfxLinkType`/`DfxTransportType` -- i.e., msprof timeline op name and L1 detail linkType/transportType field. This chapter 11.1~11.3 describes **current TaskType implementation** (not yet consuming profInfo); 11.4 gives three-field full comparison and switch target state (to-do see section 10.3 #17~#18).

### 11.1 Task Type Carriers in JSON Artifact

Current implementation JSON artifacts total three types; task types within all taken from `TaskType` enum values (string form; `task_model.py:20-26`):

| JSON Artifact | Format | Task Type Field | Generator Module |
|-----------|------|----------------|---------|
| `perf_profiling.json` (main artifact; section 8.1) | Chrome Trace Event | `traceEvents[].name` / `.cat` / `.args.task_type` | `simulator/report.py` (`export_trace`; event name is task display name: profInfo op name; section 11.4) |
| `trace_comm<commId>_iter<opIter>.json` (optional per-operator trace) | Chrome Trace Event | Same as above | Same as above |
| Task graph dump `graph_comm<commId>_iter<opIter>_dev<dev>.json` (optional; `--dump-dir`) | `light_perf.graph_dump/2` | `nodes[].type` | `graph_dumper.py` |

> dump JSON's notify nodes carry `nodes[].mask` when mask non-zero (section 7.3 Task.mask); HTML view tooltip synchronously displays; `perf_profiling.json`/trace's task args also contain `mask` (section 7.7).

Full set of task types appearing in JSON (7 kinds):

| TaskType Value (Presented in JSON) | Latency Model | Semantics |
|--------------------------|---------|------|
| `Notify_Record` | Fixed delay (local also triggers Event) | notify set (record / CKE post) |
| `Notify_Wait` | Wait Event / fixed delay | notify wait (wait / CKE wait) |
| `Reduce_Inline` | HBM dynamic shared | Local reduce (computation approximated as movement; section 5.7) |
| `Memcpy` | HBM dynamic shared | Local copy (HBM; not network) |
| `Ub_Inline_Write` | Fixed delay (same as Notify_Record) | notify inline write (old input path only) |
| `Ub_Write_Or_Read` | Port bandwidth shared (write/read directions billed independently) | Network data send/receive (SQE network tasks two directions uniformly mapped) |
| `Write_With_Notify` | Port write direction bandwidth shared + post-write notify fixed delay | Network write + attached notify (one-to-one with db `WRITE_WITH_NOTIFY`; section 5.2) |

> Section 8.3's `perf_profiling.json` structured format (meta/ops array) is draft; current main artifact actually Chrome tracing format (section 10.3 #10); regardless of which format, task types are the above 7 TaskType values; mapping table in this chapter uniformly applies to them.

**Type-related fields in trace args (current state and profInfo consumption)**: Task's `to_trace_args` (`task_model.py`) outputs four type-related fields; all four fields already switched to profInfo driven (section 10.3 #18 all landed; section 11.4):

| trace args Field | Current Source | Status |
|----------------|---------|--------------------------|
| `task_type` | `profInfo.profTaskType` op name (`prof_task_type_name`; `graph_builder._add_task` uniformly fills as `Task.task_name`; when `display_name` empty, falls back internal TaskType name) | **Landed** (one-to-one with msprof timeline op name; section 11.4.1; v2 old BLOB is `Unknown`) |
| `transport_type` | `profInfo.transportType` name (`ProfTransportTypeName`; section 11.4.3; `graph_builder._add_task` uniformly fills) | **Landed** |
| `link_type` | `profInfo.linkType` name (`ProfLinkTypeName`; section 11.4.2; `graph_builder._add_task` uniformly fills) -- **note different concept from `protocol` (CommProtocol)** | **Landed** (plan a: overwrite with DfxLinkType name) |
| `protocol` (new field) | Channel `CommProtocol` enum name (HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO; filled via `graph_builder._resolve_eids` / CCU `_channel_eids`) | **Landed** (CommProtocol migrated from `link_type`; `N/A` when no channel info) |

> `link_type` overwrite (plan a) implemented: Semantics aligned with msprof L1 detail field name; avoids same-name different-meaning; existing CommProtocol info not lost; moved to `protocol` field. CCU microcode sub-task is simulator semantic artifact; no profInfo (section 11.3); its `link_type`/`transport_type` is `N/A`; `protocol` retains channel CommProtocol name.

### 11.2 SQE Path: db taskType -> JSON Task Type Mapping Master Table

Forward mapping (db task -> JSON task display name). "JSON task type" column is trace/dump displayed op name (already switched to `profInfo.profTaskType` driven; section 10.3 #18 landed); "internal TaskType" (latency model classification; determines billing/pairing) keeps section 11.2 original judgment conditions (jettyId/direction/recordDev); correspondence with op name see section 11.4.1 "Simulation TaskType" column:

| db taskType (opTask.taskType column) | Judgment Condition | JSON Task Type (op name) | Collection-side profInfo.profTaskType (op name) | Description |
|----------------------------------|---------|--------------|--------------------------------------|------|
| `NOTIFY_WAIT` (0) | recordDev == this device | `Notify_Wait` | `TASK_NOTIFY_WAIT` -> `Notify_Wait` OK | Local wait: Participates in first-come-first-served pairing (section 5.4); waits for Event |
| `NOTIFY_WAIT` (0) | recordDev != this device | `Notify_Wait` | `TASK_NOTIFY_WAIT` -> `Notify_Wait` OK | Cross-rank wait: Node retained not paired; src != dst -> fixed `remote_wait_delay` |
| `NOTIFY_RECORD` (1) | waitDev == this device (SDMA/local notify) | `Notify_Record` | `TASK_NOTIFY_RECORD` -> `Notify_Record` OK | Local record: Participates in pairing; triggers Event |
| `NOTIFY_RECORD` (1) | waitDev != this device (SDMA/local notify) | `Notify_Record` | `TASK_NOTIFY_RECORD` -> `Notify_Record` OK | Cross-rank post: Node retained not paired; src != dst -> fixed `remote_record_delay` |
| `NOTIFY_RECORD` (1) | UBDMA WQE `inlineEn==1` (notify inline write) | `Ub_Inline_Write` | `TASK_UB_INLINE_WRITE` -> `Ub_Inline_Write` OK | notify inline write (name returned; section 11.4.4); internal TaskType still Notify_Record (fixed record delay) |
| `REDUCE` (2) | jettyId == INVALID | `Reduce_Inline` | `TASK_REDUCE_INLINE` -> `Reduce_Inline` OK | Local reduce: HBM dynamic shared billing |
| `REDUCE` (2) | jettyId valid + write direction | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` -> `Ub_Reduce_Inline` OK | Network reduce approximation (section 5.7); internal TaskType=Ub_Inline_Write; port write direction billing |
| `REDUCE` (2) | jettyId valid + read direction | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` -> `Ub_Reduce_Inline` OK | Same as above; internal TaskType=Ub_Write_Or_Read; read direction billing |
| `MEM_CPY` (3) | jettyId == INVALID (same device) | `Memcpy` | `TASK_SDMA` -> `Memcpy` OK | Local copy: HBM dynamic shared billing; profInfo.linkType=ONCHIP |
| `MEM_CPY` (3) | jettyId == INVALID (cross-device SDMA SQE) | `Memcpy` | `TASK_SDMA` -> `Memcpy` OK | Cross-chip SDMA movement; also Memcpy; difference expressed by profInfo.linkType=PCIE / transportType=SDMA (section 11.4) |
| `MEM_CPY` (3) | jettyId valid + write direction | `Ub_Write_Or_Read` | `TASK_UB` -> `Ub_Write_Or_Read` OK | Network send (direction not encoded in type name; expressed by direction/is_write); internal TaskType=Ub_Inline_Write |
| `MEM_CPY` (3) | jettyId valid + read direction | `Ub_Write_Or_Read` | `TASK_UB` -> `Ub_Write_Or_Read` OK | Network receive; internal TaskType=Ub_Write_Or_Read |
| `WRITE_WITH_NOTIFY` (7) | Always write direction (no direction inference; section 5.5) | `Write_With_Notify` (single task; not split); when `udfFlag==1`, `Reduce_With_Notify` | `TASK_WRITE_WITH_NOTIFY` -> `Write_With_Notify`; when `udfFlag==1`, `TASK_WRITE_REDUCE_WITH_NOTIFY` -> `Reduce_With_Notify` (hcomm source misspelled `Reuce_With_Notify`; section 11.4.5) | Write+notify combined task (`WriteWithNotifyTask`; section 4.4.1): Port write direction bandwidth shared + post-write notify fixed record delay; both profInfo values mapped to same internal WRITE_WITH_NOTIFY (latency model consistent; display name and isReduce distinguish) |
| `CCU_GRAPH` (4) | -- | (Placeholder; not directly produced) | `TASK_CCU` -> `Ccu` | Expands to CCU subgraph; final artifact see section 11.3 |
| `AIV_GRAPH` (5) | -- | None (skip) | `TASK_AIV` -> `AivKernel` | Warn + `skipped_aiv` count (not supported for now) |
| `SYNC_STREAM` (6) | -- | None (skip) | `UNKNOWN` -> `Unknown` | Host-side stream sync point; not device-side task (section 5.6); hcomm profiling has no corresponding type |

> Cross-rank judgment (waitDev/recordDev) see section 4.6; network direction judgment (write/read) see section 5.5.

Reverse index (JSON task display name <- all sources; including CCU path and profInfo dimension; SQE path display name = profInfo op name; CCU path display name = internal TaskType name; section 11.3):

| JSON Task Display Name | SQE Source (db taskType) | Collection-side profInfo Source (op name) | CCU Source (microcode instruction; section 11.3) |
|--------------|------------------------|------------------------------|------------------------------|
| `Notify_Record` | `NOTIFY_RECORD` (1); local/cross-rank (SDMA/local notify) | `Notify_Record` (TASK_NOTIFY_RECORD OK) | SetCKE/SetCKBit local post; TRANS/REDUCE/LOAD-class instruction embedded `setCKE` field (A6 LOAD-class also carries); cross-rank set/notify (A5 Sync family setRmtCKE; A6 TransMem notify/0x1A atomic, SyncWtX notify, SyncAtx -- remote approximation); subgraph attach launch-record / completion-record |
| `Notify_Wait` | `NOTIFY_WAIT` (0); local/cross-rank | `Notify_Wait` OK | Various instructions' embedded `waitCKE` field; ClearCKE; A6 independent Wait instruction (Xn value compare; unsatisfied -> remote wait); subgraph attach completion-wait / launch-wait |
| `Reduce_Inline` | `REDUCE` (2); jettyId invalid | `Reduce_Inline` OK | REDUCE-class instructions (msId list all local) |
| `Memcpy` | `MEM_CPY` (3); jettyId invalid (including cross-chip SDMA) | `Memcpy` OK | TRANS local movement instructions (A5 4 codes: 0x0/0x2/0x5/0xA; A6 4 codes: 0x0/0x2/0x5/0x6) |
| `Ub_Inline_Write` | `NOTIFY_RECORD` (1) + `inlineEn==1` (notify inline write) | `Ub_Inline_Write` (TASK_UB_INLINE_WRITE OK; name returned) | -- (CCU network movement uniformly mapped `Ub_Write_Or_Read`; section 11.3.1) |
| `Ub_Write_Or_Read` | `MEM_CPY` (3) / `REDUCE` (2); jettyId valid (write/read direction) | `Ub_Write_Or_Read` (TASK_UB OK; direction expressed by direction/is_write) | TRANS network movement instructions (A5 write role 3 codes: 0x3/0x7/0x9; read role 4 codes: 0x1/0x4/0x6/0x8; A6 unified TransMem 0x10: dmaOpCode 3/5=write, 6=read; including udfEnable network reduce approximation); REDUCE class with remote MS |
| `Ub_Reduce_Inline` | `REDUCE` (2); jettyId valid (network reduce) | `Ub_Reduce_Inline` (TASK_UB_REDUCE_INLINE OK) | -- |
| `Write_With_Notify` | `WRITE_WITH_NOTIFY` (7); isReduce=0 | `Write_With_Notify` (TASK_WRITE_WITH_NOTIFY OK) | -- |
| `Reduce_With_Notify` | `WRITE_WITH_NOTIFY` (7); udfFlag==1 | `Reduce_With_Notify` (TASK_WRITE_REDUCE_WITH_NOTIFY OK) | -- |

### 11.3 CCU Path: Microcode Instruction -> JSON Task Type Mapping

In db, `CCU_GRAPH` (4) SQE itself is only placeholder (`CcuPlaceholder`; section 6.1); final JSON task produced by `ccuInstrRes` instruction space microcode interpretation (Chapter 6). **A5 (V1; Ascend950; devType=6) and A6 (V2; Ascend960; devType=8) two instruction set generations both implemented** (selected by `opDetails.devType`; other devTypes fall back A5): A5 master table see section 11.3.1; A6 master table see section 11.3.3.

> profInfo correspondence: CCU placeholder SQE's `HcclTaskProfInfo` whole is `TASK_CCU` -> `Ccu` (linkType/transportType is UNKNOWN; microcode has no link dimension). Expanded microcode sub-tasks (`Memcpy`/`Ub_Write_Or_Read`/`Notify_*` etc.) are simulator semantic artifacts; **no corresponding ProfTaskParamType**; display name falls back internal TaskType name (section 10.3 #18 implementation time established trade-off; section 11.4.4); msprof side CCU tasks via microcode-level detail (`CcuProfilingInfo`: CCU_TASK/WAITCKE/LOOPGROUP three classes; reported with TASK_CCU) express equivalent granularity info; reconciliation uses whole SQE's `Ccu` as alignment anchor.

#### 11.3.1 A5 Instruction Master Table (Instruction -> JSON Task Type)

Microcode instruction = 2B header (type:4bit major class + code:11bit instruction code; section 6.2) + 30B payload:

| Instruction Major Class (type) | Instruction (code) | JSON Task Type | Description |
|----------------|------------|--------------|------|
| LOAD (0x0) | LoadSqeArgsToGSA (0x0), LoadSqeArgsToXn (0x1), LoadImdToGSA (0x2), LoadImdToXn (0x3), LoadGsaXn (0x4), LoadGsaGsa (0x5), LoadXx (0x6) | **None** (does not generate node) | Only reads/writes register model (xn/gsa); for subsequent instructions to get length, loop counter etc. |
| CTRL (0x1) | SetCKE (0x2) | `Notify_Record` | Local CKE post (ckeId as notify_id; ckeMask as mask; src=dst=local); embedded `waitCKE` field can additionally produce `Notify_Wait` |
| CTRL (0x1) | ClearCKE (0x4) | `Notify_Wait` | wait semantics (embedded `waitCKE` field); clear action itself does not produce node |
| CTRL (0x1) | Loop (0x0) | **None** | Must be triggered by LoopGroup; appearing alone is abnormal; warn and skip |
| CTRL (0x1) | LoopGroup (0x1) | (no direct node) | Loop control: Merges/expands loop body (section 6.5); in-body instructions mapped per this table; under merge path, data task size x total iterations; launched tasks carry loop_cnt (post-expansion loop count) / loop_iter_cnt (single loop self-loop count); total iterations = product of two fields |
| CTRL (0x1) | Jmp (0x5) | **None** | Conditional jump: When `Xn[conditionXnId] != expectData`, pc jumps to instruction pointed by `Xn[dstInstrXnId]` (checker `TransformJmpInstrA5` semantics; target limited to current instruction window; out-of-range warns and passes through); within loop body passes through (merge precondition verification already rejected loop body containing JMP); with step budget defense against backward jump infinite loop |
| TRANS (0x2) | LocMemToLocMs (0x0), LocMsToLocMem (0x2), LocMsToLocMs (0x5), LocMemToLocMem (0xA) | `Memcpy` (src/dst_buf_type labels mem/ms) | Local movement (LOCAL_COPY role; section 6.3.2); HBM dynamic shared |
| TRANS (0x2) | LocMsToRmtMem (0x3), LocMemToRmtMem (0x9), LocMsToRmtMs (0x7), RmtMemToLocMs (0x1), RmtMsToLocMem (0x4), RmtMsToLocMs (0x6), RmtMemToLocMem (0x8) | `Ub_Write_Or_Read` (is_write distinguishes direction; src/dst_buf_type labels mem/ms) | Cross-die/card movement (write/read roles unified type; section 6.3.2); peer rank resolved via channel reid -> eid->rank index (section 4.8/6.7) |
| TRANS (0x2) | SyncCKE (0xB), SyncGSA (0xC), SyncXN (0xD) | Local post part -> `Notify_Record`; remote post part -> `Notify_Record` (remote; not paired) | Remote set fields (SyncCKE's `rmtCKEId/locCKEMask`, SyncGSA/SyncXN's `setRmtCKEId/setRmtCKEMask`) generate remote record node (src=local; dst=INVALID -> fixed `remote_record_delay`; does not register local bookkeeping); counted as `remote_posts` (section 6.6); when SyncGSA/SyncXN carries second remote post similarly generates; embedded `waitCKE` can produce `Notify_Wait` |
| REDUCE (0x3) | Add (0x0), Max (0x1), Min (0x2) | `Reduce_Inline` | Reduce computation approximated as movement (section 5.7); MS always rank-local buffer (msId high bits only addressing encoding; section 6.3.2); local HBM billing |
| (Subgraph attach) | -- (synthesized at graph construction stage; not microcode) | None (does not produce task node) | Attach dependencies all expressed via extra edges (section 7.4.1): Entry edge SQE predecessor -> skeleton head / pre-skeleton unit head; exit edge mission completion set -> SQE successor; task graph does not spontaneously produce any node |

#### 11.3.2 Embedded Synchronization Fields (Cross-Instruction Common)

TRANS / REDUCE / SetCKE / Sync-class instruction payloads commonly carry synchronization fields; **overlapping produce** JSON tasks with main semantics:

| Payload Field | JSON Task Type | Behavior |
|---------|--------------|------|
| `waitCKEId` / `waitCKEMask` | `Notify_Wait` | When bits satisfied, pairs with local post bookkeeping (aggregated to single node: mask = all contributing posts' consumed bits ORed; first post add_match; others extra edges; section 10.3 #20); when bits insufficient: top-level instruction blocks this queue (after stall detection found, generates remote wait node and replays instruction rest; `remote_waits`); within loop body cannot block; directly generates remote wait node and continues |
| `setCKEId` / `setCKEMask` | `Notify_Record` | Local CKE post; registers seen_post bookkeeping (section 6.4) |
| `clearType` bits / SetCKE clear-bit flag | None (only register clear) | After wait completes, clears bits; does not produce node |

> Therefore one TRANS instruction can produce 1 data movement task (`Memcpy`/`Ub_Write_Or_Read`); and by embedded fields additionally produce several `Notify_Wait` (paired by bits; 0..n) and 1 `Notify_Record`. Under merged loop (section 6.5), each slot similarly retains wait/set tasks; only data tasks folded to size x total iterations.
>
> **A6 difference**: Embedded `waitCKE` only exists in SetCKBit/ClearCKE (CTRL class); local TRANS/REDUCE and TransMem/Sync class only carry tail `setCKEId/setCKEMask` (@26/28; additionally produce `Notify_Record`); wait semantics borne by independent Wait instruction (Xn value compare).

#### 11.3.3 A6 (Ascend960) Instruction Master Table (Instruction -> JSON Task Type)

devType=960 (`DEV_TYPE_960`) corresponds to V2/A6 instruction set; **implemented** (ported from checker `ccu_task_transform_instruct_a6_v3.cc`; interpreter `_MissionContextA6`). Microcode instruction = 2B header (isomorphic to A5) + 30B V2 payload (`ccu_microcode_v1.h` CcuV2 struct; pack(1)); register number commonly via `GetXnId` (bit15 addressing mode; section 6.4).

| Instruction Major Class (type) | Instruction (code) | JSON Task Type | Description |
|----------------|------------|--------------|------|
| LOAD (0x0) | LoadSqeArgsToX (0x1), LoadImdToX (0x2), LoadX (0x6), StoreX (0x7), ClearX (0x8), Nop (0x9), arithmetic Add (0xD)/Sub (0xE)/Mul (0xF)/And (0x10)/Or (0x11)/Not (0x12)/Xor (0x13)/Shl (0x14)/Shr (0x15)/Popcnt (0x16) | Data plane **none**; when `setCKEMask != 0` additionally `Notify_Record` | Unified Xn register read/write (no GSA): SQE args sequentially consumed (section 6.1); immediate load; indirect read/write (LoadX/StoreX; oMode 0=immediate offset/1=register offset); range clear (ClearX); 64-bit wrap arithmetic (Mul takes low 32-bit product; shift amount 0x3F truncated; Shl/Shr support logical/arithmetic/rotate three forms); all except Nop carry tail setCKE field (checker `ProcessSetMask`; mask non-zero -> local post node) |
| LOAD (0x0) | Load (0xA), Store (0xB) | Data plane **none**; setCKE additionally `Notify_Record` | HBM <-> Xn data plane: HBM content not modeled (checker via AllRankParamRecorder reads/writes data); registers not backfilled; warn for trace; only retains sync semantics |
| CTRL (0x1) | SetCKBit (0x2) | `Notify_Record` (+ embedded `waitCKE` additionally `Notify_Wait`) | wait -> local post -> when clearType==1 clears wait bits (checker only supports clearType=1; others warn and skip clear) |
| CTRL (0x1) | ClearCKE (0x4) | `Notify_Wait` (embedded `waitCKE`) | After wait, per checker `ClearWaitMask` semantics clears consumed wait bits; `clearMask != 0` is checker unsupported scenario (warn + lenient clear target bits) |
| CTRL (0x1) | Loop (0x0) | **None** | Must be triggered by LoopGroup; appearing alone warns and skips. `mode==1` conditional Loop: Only expands when `wishCKEBit == Xp.loopCtxId + 1024` hit (section 6.5) |
| CTRL (0x1) | LoopGroup (0x1) | (no direct node) | Loop control: Parameters from Xn/Xm/Xp three registers (loopInsCnt:10/expandOffset:9/expandCnt:9 etc.; section 6.5); merge/expand same as A5 (merge verification includes A6 additional rules: wait CKE ID not reused; cross-expansion iteration count/length/channel consistent); config illegal (loopInsCnt>512 etc.) warns and skips |
| CTRL (0x1) | Jmp (0x5) | **None** | Conditional jump: conditionType 0~5 (==/!=/>/>=/</<=); >=6 unconditional; jumpMode 0=relative pc (mod 0x10000) / 1=absolute instruction number; target takes `Xn[relTarInstrXnId]`; limited to current instruction window; out-of-range warns and passes through; within loop body passes through; step budget defense against infinite loop |
| CTRL (0x1) | Wait (0x7) | When unsatisfied, remote `Notify_Wait` | Xn value compare spin (conditionType same as Jmp; default by equal); satisfied passes through no node; unsatisfied = depends on peer remote register write -> top-level stall resolution / within loop body directly generates remote wait node (fixed `remote_wait_delay`; task id tag `xwt`; counted in `remote_waits`) |
| CTRL (0x1) | Fence (0x8) | **None** | checker does not support (error aborts); here warns and passes through (queue serial model naturally ensures order) |
| TRANS (0x2) | TransLocMemToLocMS (0x0), TransLocMSToLocMem (0x2), TransLocMSToLocMS (0x5), TransLocMemToLocMem (0x6) | `Memcpy` (src/dst_buf_type labels mem/ms) | Local movement (length `Xn[GetXnId(xlId)]`); no embedded waitCKE; tail setCKE additionally `Notify_Record` |
| TRANS (0x2) | TransMem (0x10) | `Ub_Write_Or_Read` (dmaOpCode 3/5=write, 6=read); notify/atomic -> remote `Notify_Record` | Unified network movement: Channel = `Xn[GetXnId(xcId)]` register value -> peer rank/eid resolution same as A5 (section 4.8/6.7); when `msIdMode==1`, write source / read destination is local MS (buf_type=ms); dmaOpCode 5 write-after-notify and 0x1A remote atomic count (no data plane) approximated as remote record (mask = low 16 bits of notify value); `udfEnable` network reduce billed as normal UB movement (section 5.7 approximation); 0x19 and other dmaOpCode / `udfType != 0` warn and skip data plane (setCKE post still executed) |
| TRANS (0x2) | SyncXnWtx (0xD), SyncAtx (0xE) | Remote `Notify_Record` (approximation); SyncAtx also has setCKE -> local `Notify_Record` | Cross-rank register sync: SyncWtX `notifyValid==1` notify and SyncAtx atomic write approximated as remote record (remote target type unresolvable in single rank; section 6.6); main value sync is remote register write, no node built (A5 SyncXn same metric); SyncWtX tail setCKE field checker does not consume (local does not post); channel unresolvable warns and skips (`eid_unresolved`) |
| REDUCE (0x3) | ReduceAdd (0x0), ReduceMax (0x1), ReduceMin (0x2) | `Reduce_Inline` | Local MS batch reduce (msId[8] list + `Xn[XnIdLength]` length); no embedded waitCKE; tail setCKE additionally `Notify_Record`; MS always local HBM billing (section 6.3.2) |
| (Subgraph attach) | -- (synthesized at graph construction stage; not microcode) | None (does not produce task node) | Same as A5 (section 7.4.1 extra edge attach) |

> A6 established deviations from checker (HBM data plane, remote target type resolution, checker error scenario lenient downgrade etc.) summarized in `_MissionContextA6` class docstring and section 6.7 comparison table; loop merge/pipeline concurrency recovery (section 6.5/6.5.1), queue parallelization (section 7.4.1), SQE grouping and stall resolution (section 6.1/6.6) etc. framework mechanisms shared across both generations.

### 11.4 HcclTaskProfInfo Three-Field Comparison (profInfo -> JSON Display)

From v3 collection onward, db tasks carry `HcclTaskProfInfo` (section 4.4.1; single-source definition `include/sim_profiling_defs.h`). This section gives three fields' "value -> enum -> JSON display string" full comparison and each value's correspondence with simulation TaskType (latency model classification; section 7.3); C++ side name functions (`ProfTaskParamTypeName`/`ProfLinkTypeName`/`ProfTransportTypeName`) are the unique authoritative source of display strings.

> **Implementation status**: `task_type`/`transport_type`/`link_type`/`protocol` four fields all landed per this section comparison (Python side display name mapping in `db_reader.py`'s `PROF_*_NAMES`; character-by-character consistent with C++ name functions; section 11.4.5; `graph_builder._add_task` uniformly fills -- `task_type` via `Task.task_name`/`display_name`; section 10.3 #18 wrap-up completed).

#### 11.4.1 profTaskType (ProfTaskParamType, 0~22 -> msprof timeline op name)

This table supplements each value's correspondence with simulation TaskType (`task_model.py`; latency model classification) -- previously db task and simulation task type matched via db taskType column (section 11.2); after profInfo consumption, this table's `profTaskType -> simulation TaskType` is the second matching dimension (judgment conditions unchanged; same db taskType diverted to different TaskType rows by jettyId/direction noted in-line):

| Value | Enum | JSON Display String (op name) | Simulation TaskType (latency model classification) | light_perf Occurrence Scenario |
|---|---|---|---|---|
| 0 | TASK_SDMA | `Memcpy` | `Memcpy` (jettyId invalid; HBM billing) / `Ub_Inline_Write`/`Ub_Write_Or_Read` (jettyId valid by direction; port billing) | Local copy, cross-chip SDMA SQE (linkType distinguishes; section 11.4.2) OK |
| 1 | TASK_RDMA | `RDMASend` | `Memcpy` (jettyId invalid; HBM billing approximation) | DPU host ibv ordinary read/write (`RDMASend`; hcomm UbMemTransport::Write/Read same metric) OK |
| 2 | TASK_REDUCE_INLINE | `Reduce_Inline` | `Reduce_Inline` (jettyId invalid; HBM billing) / `Ub_Inline_Write`/`Ub_Write_Or_Read` (network approximation; section 5.7; port billing) | Local reduce OK |
| 3 | TASK_REDUCE_TBE | `Reduce_TBE` | -- (does not appear) | 910 generation reserved; does not appear |
| 4 | TASK_NOTIFY_RECORD | `Notify_Record` | `Notify_Record` | SDMA/local notify record OK |
| 5 | TASK_NOTIFY_WAIT | `Notify_Wait` | `Notify_Wait` | notify wait OK |
| 6/7 | TASK_SEND_NOTIFY / TASK_SEND_PAYLOAD | `Send_Notify` / `Send_Payload` | -- (does not appear) | hcomm reserved; does not appear |
| 8 | TASK_WRITE_WITH_NOTIFY | `Write_With_Notify` | `Write_With_Notify` | WRITE_WITH_NOTIFY (7) whole task (single task mapping; section 10.3 #19 supported) OK |
| 9 | TASK_WRITE_REDUCE_WITH_NOTIFY | `Reduce_With_Notify` (tool-side corrected spelling; hcomm source/msprof is misspelled `Reuce_With_Notify`; section 11.4.5) | `Write_With_Notify` (isReduce=1 reduce write payload; data_type distinguishes display) | WRITE_WITH_NOTIFY (7) isReduce=1 |
| 10 | TASK_CCU | `Ccu` | (CCU placeholder whole SQE; expands to Chapter 6 subgraph; section 11.3) | CCU placeholder whole SQE OK |
| 11/12 | TASK_AICPU_KERNEL / TASK_AICPU_REDUCE | `AicpuKernel` / `Aicpu_Reduce` | -- (not consumed) | AICPU kernel dispatch (goes through device process; currently no collection point / not consumed) |
| 13 | TASK_AIV | `AivKernel` | -- (skip) | AIV kernel task (skip) |
| 14 | TASK_UB_INLINE_WRITE | `Ub_Inline_Write` | `Notify_Record` (notify fixed record delay; display name returned to op name OK) | UBDMA `inlineEn==1` notify inline write |
| 15 | TASK_UB_REDUCE_INLINE | `Ub_Reduce_Inline` | `Ub_Inline_Write` (write) / `Ub_Write_Or_Read` (read) (network reduce approximation; port billing) | UBDMA `udfFlag==1` network reduce |
| 16 | TASK_UB | `Ub_Write_Or_Read` | `Ub_Inline_Write` (write direction) / `Ub_Write_Or_Read` (read direction) (direction expressed by direction/is_write) | UBDMA ordinary read/write WQE (write/read two directions same value) |
| 17~22 | TASK_DPU_KERNEL / THREAD_FENCE / CHANNEL_FENCE / INLINE_WRITE / NOTIFY_WAIT / WRITE_WITH_NOTIFY | `DpuKernel` / `Dpu_ThreadFence` / `Dpu_ChannelFence` / `Dpu_Notify_Record` / `Dpu_Notify_Wait` / `Dpu_Write_With_Notify` | `Dpu_Notify_Record`->`Notify_Record`, `Dpu_Notify_Wait`->`Notify_Wait` (cross-rank fixed delay), `Dpu_Write_With_Notify`->`Write_With_Notify` (single task port billing; section 11.4.6); `DpuKernel`/Fence class current collection path does not produce | host CPU RoCE (ibv_dpu_stub) task OK (section 11.4.6) |
| 0xFF | UNKNOWN | `Unknown` | -- (SYNC_STREAM etc. already skipped; no node produced) | SYNC_STREAM, level1 channel API task, v2 old BLOB etc. unlabeled source |

> `task_type` already switched to op name (section 10.3 #18 wrap-up completed); this table's "Simulation TaskType" column now is internal latency model classification index (section 11.4.5 #3); OK rows op name and TaskType value consistent.
> After SQE batch merge (section 5.1.1), merged task's op name and link/transport take **the last** data task in batch's profInfo (description follows last task; length takes full-batch cumulative); v3 and earlier BLOBs have no sqeGroupId so not merged; displayed per-task.

#### 11.4.2 linkType (ProfLinkType, 0~9 -> hcomm DfxLinkType enum name)

| Value | Enum | JSON Display String | light_perf Occurrence Scenario |
|---|---|---|---|
| 0 | LINK_ONCHIP | `ONCHIP` | Local Memcpy/Reduce_Inline/Notify_* |
| 1 | LINK_HCCS | `HCCS` | Reserved (current collection path does not produce) |
| 2 | LINK_PCIE | `PCIE` | Cross-chip SDMA SQE (same-device judged as local ONCHIP; section 4.4.1 collection rule) |
| 3 | LINK_ROCE | `ROCE` | host CPU RoCE (DPU; ibv_dpu_stub) task OK (section 11.4.6) |
| 4/5/6 | LINK_SIO / HCCS_SW / STANDARD_ROCE | `SIO` / `HCCS_SW` / `STANDARD_ROCE` | Reserved |
| 7 | LINK_UB | `UB` | Network task (UBDMA WQE all classes) |
| 8 | LINK_UBoE | `UBoE` | UBoE channel (collection side temporarily does not distinguish UB/UBoE; temporarily does not appear) |
| 9 | LINK_RESERVED | `RESERVED` | Reserved |
| 0xFF | UNKNOWN | `Unknown` | CCU/AIV/SYNC_STREAM, level1 channel API task |

> **Implementation status**: trace args' `link_type` field filled per this table (`prof_link_type_name`; `graph_builder._add_task` uniformly injects; section 11.1 plan a); this table and original `link_type` carried `CommProtocol` channel protocol (HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO) are **two different concepts**: CommProtocol describes channel establishment protocol; DfxLinkType describes task execution link (profiling dimension). CommProtocol name already migrated to `protocol` field (`N/A` when no channel info).

#### 11.4.3 transportType (ProfTransportType, 0~4 -> msprof L1 detail transportType value range)

| Value | Enum | JSON Display String | light_perf Occurrence Scenario |
|---|---|---|---|
| 0 | TRANSPORT_SDMA | `SDMA` | Cross-chip SDMA SQE |
| 1 | TRANSPORT_RDMA | `RDMA` | Reserved (910 generation RDMA semantics) |
| 2 | TRANSPORT_LOCAL | `LOCAL` | Local task (Memcpy/Reduce_Inline/Notify_*) |
| 3 | TRANSPORT_UB | `UB` | Network task (**latency model anchor**: port bandwidth shared billing; section 5.5) |
| 4 | TRANSPORT_ROCE | `ROCE` | host CPU RoCE (DPU; ibv_dpu_stub) task OK (section 11.4.6) |
| 0xFF | UNKNOWN | `Unknown` | Unlabeled task |

> **Implementation status**: trace args' `transport_type` field filled per this table (`prof_transport_type_name`; `graph_builder._add_task` uniformly injects). Same-named `Memcpy` task's local/cross-chip difference expressed by linkType+transportType: ONCHIP+LOCAL (local) vs PCIE+SDMA (cross-chip) -- trace args now losslessly distinguishes these two previously indistinguishable task classes in JSON.

#### 11.4.4 Ub_Inline_Write Bi-Directional Semantic Misalignment and Switch Implementation Record

**Historical problem**: Before switch, JSON's `Ub_Inline_Write` (network write direction send) and hcomm profiling's `Ub_Inline_Write` (TASK_UB_INLINE_WRITE; notify inline write) **refer to different tasks**, and in reverse misalignment:

| Task (collection semantics) | db taskType | ProfTaskParamType (op name) | Pre-switch JSON Type |
|---|---|---|---|
| UBDMA ordinary write WQE (network send; write direction) | MEM_CPY (3) | `TASK_UB` -> `Ub_Write_Or_Read` | `Ub_Inline_Write` <- name misalignment |
| UBDMA network reduce (write/read direction) | REDUCE (2) | `TASK_UB_REDUCE_INLINE` -> `Ub_Reduce_Inline` | `Ub_Inline_Write` / `Ub_Write_Or_Read` <- name misalignment |
| UBDMA inline write notify (`inlineEn==1`) | NOTIFY_RECORD (1) | `TASK_UB_INLINE_WRITE` -> `Ub_Inline_Write` | `Notify_Record` <- name misalignment |

**Implemented (section 10.3 #18 wrap-up)** -- trace/dump display name switched to `profInfo.profTaskType` op name (`Task.task_name`; `graph_builder._add_task` uniformly injects); three misalignments all eliminated:
1. Network data task (TASK_UB) write/read uniformly displays `Ub_Write_Or_Read`; **direction no longer encoded into type name**; expressed by trace args existing `direction` (write/read) and `is_write` field -- latency model unchanged; still billed separately by port write/read direction bandwidth shared (section 5.5).
2. `Ub_Inline_Write` name **returned** to notify inline write task (db taskType=NOTIFY_RECORD + inlineEn==1); one-to-one with msprof timeline.
3. Network reduce displays `Ub_Reduce_Inline` (TASK_UB_REDUCE_INLINE); retains recognizable reduce semantics.
4. `Notify_Record`/`Notify_Wait`/`Reduce_Inline`/`Memcpy` four classes originally already consistent with op name; switch no change (`Memcpy` corresponds to TASK_SDMA).

**Implementation trade-offs (established decisions)**:
- v2 old BLOB (no profInfo; profTaskType=0xFF) displays `Unknown` -- same metric as `link_type`/`transport_type`'s v2 behavior (strict semantics; no internal name fallback).
- CCU microcode sub-task (section 11.3) has no profInfo; display name falls back internal TaskType name -- simulator semantic artifact has no reconciliation need; maintains status quo (network movement unified `Ub_Write_Or_Read`; direction expressed by `direction`/`is_write`; section 11.3.1).
- Internal `TaskType` enum retired to latency model classification (section 11.4.5 #3); simulation/pairing/billing zero changes.

#### 11.4.5 String Consistency Convention (Reconciliation Basis)

1. **Python-side display strings must be character-by-character consistent with C++ side three name function outputs** (`sim_profiling_defs.h` single source); including hcomm source's established spellings (`Ub_Write_Or_Read`, `Dpu_Notify_Record` etc.) -- this is the basis for field-by-field reconciliation with msprof timeline / L1 detail. **Single exception**: `TASK_WRITE_REDUCE_WITH_NOTIFY`'s op name tool side unified to corrected spelling `Reduce_With_Notify` (hcomm source both places are misspelled `Reuce_With_Notify`: `profiling_common.h:42` and `dfx_profiling_handler_lite.cc:67`; msprof raw data shows misspelled version); cross-system reconciliation encountering this string needs to do `Reuce_With_Notify <-> Reduce_With_Notify` difference mapping.
2. trace args field-level switch (**all landed**; section 10.3 #18 wrap-up): `transport_type` <- transportType name, `link_type` <- linkType name (`graph_builder._add_task` uniformly injects; value range tables see section 11.4.2/11.4.3); CommProtocol migrated to new field `protocol` (plan a; section 11.1); `task_type` <- profTaskType op name (`Task.task_name`/`display_name`; section 11.4.4 implementation record).
3. `TaskType` enum already retired to internal latency model classification (`task_name` bears display duty); trace event name (`traceEvents[].name`) and `args.task_type` both switched to op name; CCU microcode sub-task/CLI input has no profInfo; `display_name` falls back internal name.

#### 11.4.6 DPU (host CPU RoCE) Task Collection and Merge

host DPU transfer (hcomm `HostCpuRoceChannel`; pod network cross-node path) ibv hijack tasks (`ibv_dpu_stub.cc`) previously did not label profInfo (displays `Unknown`); WriteWithNotify split into two tasks; peer resolution loopbacks to local. This section is the alignment implementation record for this path.

**profInfo labeling** (aligned with hcomm `host_cpu_roce_channel.cc` DFX metric; link/transport always ROCE/ROCE):

| ibv Operation | db taskType | profTaskType (op name) |
|---|---|---|
| Pure imm notify (len=0 WRITE_WITH_IMM / hybrid) | NOTIFY_RECORD | `TASK_DPU_INLINE_WRITE` -> `Dpu_Notify_Record` |
| Write+imm (len>0 WRITE_WITH_IMM) | **WRITE_WITH_NOTIFY (7) single task** | `TASK_DPU_WRITE_WITH_NOTIFY` -> `Dpu_Write_With_Notify` |
| Pure RDMA write/read (no imm) | MEM_CPY | `TASK_RDMA` -> `RDMASend` (aligned with hcomm `UbMemTransport::Write/Read` precedent) |
| poll_cq pairing | NOTIFY_WAIT | `TASK_DPU_NOTIFY_WAIT` -> `Dpu_Notify_Wait` |

**WriteWithNotify merged single task**: Whole operation produces only one taskType=7 task (`WriteWithNotifyTaskData`: src/dst rank, offsets, len, notifyId, protocol); aligned with hcomm whole operation only reports one `Dpu_Write_With_Notify` metric; `DpuPendingNotify` pairing mechanism unchanged. Simulation side `_map_write_with_notify` goes through port write direction bandwidth shared (local RoCE eid -> ranktable `d2h` port; when `d2h` port bandwidth not configured, falls back `default_port_bandwidth`).

**Real peer resolution (phantom echo problem; socket timing binding)**: estimate mode socket layer constructs phantom peer (local Send mirror replayed as Recv; `hccp_ra_socket_stub.cc`); "remote gid" exchanged by QP handshake is actually local self; `RaTypicalQpModify`'s gid->IP resolution back to local. Fix (zero-byte read; only relies on API call timing):
- `RaGetSockets` resolves real peer EndPoint from `SocketInfoT.remoteIp` (hcomm link config; client/server roles both carry); stores into FdInfo (peerDeviceId/peerEid/localEid);
- `RaGetQpAttr` anchor (hccp_stub; executes only within estimate branch) records QP awaiting binding (`DpuRecordPendingBindQp`; `bindAttempted` prevents Modify duplicate query attr then mis-binding);
- Same-thread next `RaSocketSend` (i.e., hcomm ExchangeData's attr pack -> Send timing) via `BindDpuPendingQps` binds QP to that socket's real peer: `DpuQpInfo` adds `peerRankId`/`lcEid` (peer rank from FdInfo's peerEid queries `RankEidMap`; estimate communicator table only this rank; lcEid for WRITE_WITH_NOTIFY port billing).
`RaTypicalQpModify` maintains `remoteDeviceId`'s loopback plane semantics (`DpuPendingNotify` pairing depends; under estimate = local); task metadata uniformly takes bound field.

**NotifyWait precise synthesis (hcomm primitive interception)**: Correct semantics is "only peer post-inserted pending generates NotifyWait for me". Under phantom, peer does not run; that notify has no real source; and hcomm `NotifyWait` does imm hard verification on poll result (`wc.imm_data == dpuNotifyId`; mismatch immediately errors). Implementation (`hcomm_api_stub.cc` intercepts 5 hcomm northbound primitives; real-body pointer static one-shot dlsym resolution; **normal mode first line short-circuit pure forward; zero overhead**):
- post-class primitives (`HcommChannelNotifyRecordOnThread` / `HcommWriteWithNotify(+-Nbi)OnThread`; input carries `remoteNotifyIdx`) -> `InsertDpuPendingNotifyEstimate` records `channelSlots[channel][idx] = (imm, notifyId)` slot mapping -- under echo `remoteDpuNotifyIds_ == localDpuNotifyIds_`; post-side idx->imm mapping is wait-side expected mapping (same idx same slot value);
- wait-class primitives (`HcommChannelNotifyWaitOnThread(+-WithDefaultTimeout)`; input carries `localNotifyIdx`) -> `ConsumeDpuPendingNotifyAndEmit` precise path: expected imm = `channelSlots[channel][idx]` **precisely known**; directly synthesizes "virtual peer post" to construct wc return; imm must pass hcomm hard verification; NotifyWait node records local real wait event; only virtualized is "notify arrival" this external fact (isomorphic to socket echo virtualizing "peer response");
- Design constraint: estimate mode's wait consumption **only goes through precise path** -- all wait entries must be intercepted (single-card phantom FIFO loopback consumption unreliable; send/receive asymmetric imm mismatch/timeout; already removed with old mechanism); unintercepted entry -> `pollCq::miss` **ERROR** (deduplicated per qpn+idx combination once; prevents busy-loop flooding log) -> hcomm timeout exposes; accordingly add interception in `hcomm_api_stub.cc`.

**taskSrc source marker (BLOB v5)**: `HcclTaskMetaData` adds `taskSrc` (0=default, 1=DPU); DPU stub emitted tasks set 1; for Python side to precisely split by source (does not rely on transport==ROCE etc. link semantic criteria; AICPU RoCE etc. future sources unaffected).

**DPU sequence merge pass (section 5.1.2; `sqe_group_merge.py::merge_dpu_write_sequences`)**: hcomm `WriteWithNotify` for len > maxMsgSize large messages split into N-1 ordinary RDMA_WRITE + 1 closing WRITE_WITH_IMM (whole operation still reports only one); collection side per-WR lands in db as N-1 MEM_CPY + 1 WRITE_WITH_NOTIFY. pass scans by DPU stream order src/dst dual-address contiguous MEM_CPY chain; closed by address-contiguous WRITE_WITH_NOTIFY merged to single entry (len full-chain cumulative; description follows closing task including notifyId; occupies closing slot); chain without closing not merged (independent pure writes retained per-entry `RDMASend`); only `src_type==1` data tasks participate; other tasks retained in place; do not cut chain.

> **Verification** (222 network 8MB all_reduce): First round (20260921) 15 `Unknown` -> 0 (`Dpu_Notify_Record` x5 / `Dpu_Notify_Wait` x9 / `Dpu_Write_With_Notify` x4; link=ROCE); peer rank=2/4/6 (real cross-pod peer); `eid_unresolved=0`; WRITE_WITH_NOTIFY enters `d2h` port billing. Final version (20260922; after primitive interception): estimate (-np 1) makespan 166.596us; primitive interception 16 times; precise synthesis 11 times; miss=0; normal (-np 8) checker All Success (primitive pure forward; interception log 0 entries); single-machine 8p (118) SQE path regression no change.

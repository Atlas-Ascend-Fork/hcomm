# light_performance Plugin Design Document (Including Simulator Design)

> Document location: `docs/light_performance/simulator_design.md`
> Requirement source: `docs/light_performance/HCCLSIM/Requirement Description`
> Reference implementation: `src/plugin/checker/src/framework/task_graph_generator_v3/` (graph generation principles),
> `docs/light_performance/HCCLSIM/hccl_sim/` (simulator prototype, **authoritative baseline for simulation module**)
> Alignment leftovers and to-match list: Chapter 10
> db task / CCU instruction -> JSON task type mapping summary table: Chapter 11

---

## 1. Requirement Background and Document Description

### 1.1 End-to-End Requirement

Integrate light_performance (a lightweight performance simulation tool) as an independent plugin into the current project, completing the following flow end-to-end:

```
hccl-vm flow execution --generates--> task sequence / instruction db files
        |
        v
light_performance plugin reads db files --generates--> task graph required for performance simulation
        |
        v
Performance simulation module performs performance simulation based on task graph
        |
        v
Outputs performance profiling json file
```

### 1.2 Sub-Requirement Decomposition and Chapter Mapping

| Sub-Requirement | Content | Status | Chapter in This Document |
|--------|------|------|---------|
| Sub-req 1 | Plugin engineering: command line, compilation and installation, complete plugin architecture | Command line and installation design already detailed (@estimate one-shot execution channel) | Chapter 3 |
| Sub-req 2 | Integration with hccl-vm: db file reading, task graph generation | **Detailed design in this iteration** | Chapters 4, 5 |
| Sub-req 3 | CCU mode task graph generation (SQE queue + microcode instruction sequence -> task graph) | **Detailed design in this iteration** | Chapter 6 |
| Sub-req 4 | Performance simulation module (task graph parsing, performance simulation) | **Aligned and incorporated per HCCLSIM baseline in this iteration** | Chapter 7 |
| Sub-req 5 | Performance profiling json output (performance metrics, task graph) | High-level design | Chapter 8 |

> Chapter 10 records the **field gaps, type mismatches, and other conflicts** during this alignment
> (differences between the db-to-graph path and the HCCLSIM simulation model) for future matching reference. Chapter 11 Section 11.4 provides the mapping between v3-collected
> profInfo (`HcclTaskProfInfo` three fields) and JSON display types, and the target state.

### 1.3 Delivery Scope of This Iteration

This iteration focuses only on **db file reading/parsing and the graph generation module**:

- **Input**: db files in the hccl_vm_install/data directory (format and data reference `hccl_vm_install/data/hccl_vm_data.db`).
- **Output**: Task graph `DependencyGraph` and `SimConfig` required by the performance module (hccl_sim).
- **Reference code**:
  - Basic graph generation logic: `src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_generator_v3.cc/h`;
  - CCU microcode to task graph: `src/plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/`;
  - AIV mode graph generation: not considered for now.
- **Graph generation scope (core design premise)**: db contains **single-rank task data**; this plugin performs **single-rank graph generation** -
  this is one of the main differences from checker (multi-rank joint graph generation). In the original graph generation process, **cross-rank task edges
  (post-wait relations) are all ignored**: only intra-rank post-wait edges are of concern; cross-rank ones are not considered
  (see Sections 5.1, 5.4, 6.6).
- **Relationship with checker**: the output structure of light_performance's graph generation module is not identical to checker's, but the basic graph generation and
  instruction conversion principles are the same; the former does not need checker's subsequent memory conflict validation / semantic validation operations, so it can be appropriately simplified
  (difference comparison in Sections 5.8 and 6.7).

---

## 2. Plugin Overall Design

### 2.1 End-to-End Data Flow and Module Partitioning

```
+--------------------------- light_performance plugin --------------------------+
|                                                                           |
|  +-----------+   +---------------+   +-----------+   +----------------+   |
|  | db_reader |-->| graph_builder |-->| simulator |-->| profiling      |   |
|  | (Ch 4)    |   | (Ch 5, 6)     |   | (Ch 7)    |   | (Ch 8)         |   |
|  +-----------+   +---------------+   +-----------+   +----------------+   |
|   sqlite read       SQE->Task map      simpy discrete     json + trace    |
|   BLOB decode       notify pairing      event port bw sharing             |
|   op grouping       CCU microcode       HBM dynamic sharing               |
|                     expansion           eid->port resolution              |
|  Input: data/hccl_vm_data.db (+ data/hccl_sim.db, ranktable.json)          |
|  Output: (SimConfig, DependencyGraph, TopoResolver) -> makespan ->         |
|          perf_profiling.json (+ graph dump: DOT/JSON/HTML)                 |
+----------------------------------------------------------------------------+
```

Data contracts between modules:

```
db_reader     --OpExecutionGroup(intermediate data model, Section 4.5)---------->  graph_builder
graph_builder --(SimConfig, DependencyGraph)--------------------------------->  simulator   [output of this iteration's requirement]
simulator     --result dict(tasks with start/end + makespan + ports)--------->  profiling/report
```

### 2.2 Plugin Directory Structure (Source location: `src/plugin/light_performance`)

```
src/plugin/light_performance/
+-- __init__.py
+-- __main__.py             # Independent CLI entry (Section 3.3)
+-- main.py                 # @estimate plugin one-shot execution entry (Sections 3.1/3.2)
+-- manifest.json           # Plugin manifest (name=estimate; installed with package)
+-- pipeline.py             # Shared pipeline: db -> graph -> simulation -> perf_profiling.json
|                           #   (shared by run_perf_sim.py and main.py, Section 3.1)
+-- db_reader.py            # Sub-req 2: db reading and BLOB decoding (Chapter 4)
+-- sqe_group_merge.py      # SQE batch merge preprocessing: within same (stream, sqeGroupId) batch, UDMA data tasks merged to 1 (Section 5.1.1)
+-- graph_builder.py        # Sub-req 2: SQE -> DependencyGraph (Chapter 5)
+-- graph_dumper.py         # Task graph dump: DOT + JSON + HTML
+-- ccu_graph_builder.py    # Sub-req 3: CCU microcode -> task graph (Chapter 6)
+-- register_model.py       # CCU register/CKE execution model (Section 6.4)
+-- sim_config.json         # Default latency/bandwidth config (Section 7.2, units us/GBps; installed with package)
+-- bw_profile.json         # Default bandwidth utilization config (installed with package)
\-- simulator/              # Sub-req 4: simulation module (ported from HCCLSIM, Chapter 7)
    +-- config.py           #   SimConfig (JSON units us/GBps)
    +-- task_model.py       #   Task / TaskType / TRANSFER_TYPES / INVALID
    +-- dependency_graph.py #   DependencyGraph (int stream id)
    +-- task_timing.py      #   Task latency formulas (single-point maintenance)
    +-- bandwidth_profile.py#   Bandwidth utilization table
    +-- topo_resolver.py    #   eid->ports topology resolution (ranktable.json)
    +-- input_builder.py    #   JSON task file construction + load_config
    +-- simulator.py        #   HcclSimulator / NetworkPort / _BandwidthManager
    +-- report.py           #   Text report + Chrome trace
    \-- __main__.py         #   hccl_sim compatible CLI (independent of db path)
```

> The `simulator/` subpackage is ported from `docs/light_performance/HCCLSIM/hccl_sim/`; its interface takes
> HCCLSIM as the baseline (Chapter 7). The repository root's `dump_graph.py` is the one-click dump entry
> (db -> DOT/JSON/HTML, optionally with simulation timestamps).
> `manifest.json` and `main.py` (one-shot execution entry, Chapter 3) are placed directly in this package directory;
> at installation, the entire directory becomes the `plugin/light_performance` plugin directory
> (installation layout in Section 3.1, CMake rules in Section 3.4).

### 2.3 Relationship with Existing Plugins

| Plugin | Responsibility | Relationship with light_performance |
|------|------|---------------------------|
| checker | Communication correctness validation (deadlock/memory conflict/semantics) | Reuses its graph generation principles; main difference: checker performs **multi-rank joint graph generation**, light_performance performs **single-rank graph generation** (does not build cross-rank pairing edges, remote synchronization tasks retained and approximated by fixed remote latency), and does no validation-class post-processing |
| runner | Virtual runtime (task-by-task simulated execution of data plane) | Also consumes opTask/ccuInstr; light_performance reads only, does not write, does not advance isDone |
| insight | Visualization | light_performance's profiling json / chrome trace can be consumed by it (future) |

> **Plugin tag vs directory/module name**: This plugin's installation tag is `estimate` (the host's `@estimate` command channel
> and mode gate hard-code by tag, see Section 3.1), while the plugin directory and Python package name are both
> `light_performance` (`plugin/light_performance`) - the tag corresponds to the "host routing identity",
> and the directory/package name corresponds to the "code organization identity".

---

## 3. Plugin Engineering Design (Sub-Requirement 1)

### 3.1 Plugin Identification and manifest.json

**Plugin tag fixed as `estimate`** (rather than module name light_performance): the host CLI has
hard-coded gates and path conventions for this plugin -

- `hccl-vm plugin run @estimate`'s execution channel in `src/cmd/cmd_base_utils.cc:599-680`
  (`RunUserPlugin`) has a special case for `estimate`: only available in estimate mode (`sim::IsEstimateMode()`),
  plugin directory fixed as `<installRoot>/plugin/light_performance` (tag decoupled from directory name),
  directly reads its `manifest.json` and synchronously executes the `entry` command (details in Section 3.2);
- Mode exclusivity: `@checker` is unavailable in estimate mode (`cmd_base_utils.cc:618-624`),
  `@estimate` is unavailable in non-estimate mode - correctness validation and performance simulation belong to the two runtime modes' wrap-up actions;
- Plugin scan/resident model (`src/plugin/hccl_plugin_manager.cc:94-169`, `src/plugin/hccl_plugin.cc`)
  matches tag by manifest's `name` field (scans `<installRoot>/plugin`, depth <= 2).

manifest.json (source and installation both inside package directory: `src/plugin/light_performance/manifest.json`
-> `plugin/light_performance/manifest.json`):

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

- `name`: plugin tag, host routes by it (`hccl-vm plugin run @estimate`).
- `entry`: startup command inside plugin directory. @estimate takes the one-shot synchronous execution channel (Section 3.2);
  at execution time cwd is the plugin directory, relative paths resolve against the plugin directory.
- `setting`: plugin-private config key-values, read by main.py itself (following checker's `SettingManager`
  self-managed config approach, Python side directly json-parses). Path-type configs resolve relative to plugin directory (i.e., package directory):
  `db` default `<installRoot>/data/hccl_vm_data.db` (same source as checker's
  `Loader::LoadOpTaskFile`); `sim_config`/`bw_profile` are latency/bandwidth and utilization configs in Section 7.2 format,
  defaults to `sim_config.json`/`bw_profile.json` installed with package in same directory as manifest;
  `ranktable` defaults to `ranktable.json` in same directory as db;
  `device`/`iter` correspond to CLI's `--device`/`--iter`; `out_dir` is output root directory
  (subdirectories created by runtime timestamp underneath); `per_op_mode` currently only supports per-operator (when false, warns
  and executes per-operator); `enable_trace`/`dump_graph`/`max_trace_events` control additional artifacts.

Source and installation layout (plugin directory equals package directory, no nesting):

```
source = installation artifact structure (src/plugin/light_performance -> hccl_vm_install/plugin/light_performance):
plugin(light_performance)/            # plugin directory = light_performance package directory
+-- manifest.json                     # plugin manifest (name=estimate)
+-- main.py                           # @estimate one-shot execution entry (reuses pipeline.py)
+-- __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
+-- sim_config.json                   # default latency/bandwidth (Section 7.2)
+-- bw_profile.json                   # bandwidth utilization
+-- simulator/                        # simulation subpackage (Chapter 7)
\-- data/                             # runtime output (perf_profiling.json etc., Chapter 8;
                                      #   produced only inside install tree, source tree has no such directory)
```

> main.py is located within the package directory but runs as a script (`python3 main.py`): it adds the package's **parent directory**
> (`plugin/`, source tree is `src/plugin/`) to `sys.path` then does
> `from light_performance.pipeline import ...` - in-package relative imports unaffected,
> source tree and install tree layouts consistent.
> The repository root's `run_perf_sim.py` (manual entry) and `main.py` share the same pipeline implementation:
> main.py adds "install-tree path conventions + manifest setting reading" on top of it.
> Source tree debugging still uses run_perf_sim.py (Section 3.3); inside install tree, `hccl-vm plugin run @estimate`
> triggers main.py.

### 3.2 Call Chain and Process Model (@estimate One-Shot Execution)

Host side needs only one path adaptation - @estimate's one-shot execution channel is already implemented
(`cmd_base_utils.cc:599-680`); this iteration changes the special-case branch's plugin directory from `plugin/estimate`
to `plugin/light_performance` (tag still `estimate`, directory is package directory):

```
hccl-vm plugin run @estimate
    |
    v
RunUserPlugin("@estimate")                                  # subcmd_plugin.cc:55-57 routing
    +- gate: must be in estimate mode, otherwise error exit              # cmd_base_utils.cc:611-617
    +- read <installRoot>/plugin/light_performance/manifest.json  # missing file/entry field -> error
    \- system("cd <installRoot>/plugin/light_performance && python3 main.py")   # synchronous blocking execution
       \- non-0 exit code -> HCCL_VM_ERROR, command overall returns failure
```

**Long-duration command client wait strategy** (`cmd_base_utils.cc`'s `StartHostClient`/`ServerListen`):
In a session, `hccl-vm` sends to host via POSIX message queue, **host responds only after `ParseCommand` returns (i.e., command completes)**;
CCU MS mode microcode expansion can reach 10^5 tasks (empirically 192,005 tasks, software simulation about 54s),
far exceeding the normal command's 5s response limit. To this end:

- After the client recognizes `plugin run @estimate`, it uses a long wait cap (default 30 minutes, env var
  `HVM_CLIENT_RESP_TIMEOUT_MS` can override); during segmented wait, every 30s print
  "command still executing on host side" prompt to terminal (first at 5s), and print explicit warning on timeout; other commands keep the 5s cap.
- Response carries original command string (`ACK:<cmd>`); before sending a command, client drains the response queue and only accepts responses strictly matching this command -
  avoiding misreading a late response from a previous timed-out command and returning early (especially when retrying the same command after interrupt).
- host prints `[HVM] @estimate performance simulation start/end` markers at simulation start/end to terminal (stdout, along with report
  output); this mechanism ensures **the prompt returns only after simulation truly ends**, no longer producing the misimpression that "simulation already ended".

main.py (plugin side) execution flow:

```
RunEstimate():
    setting   <- parse manifest.json's setting section (paths resolve relative to plugin directory)
    db        <- setting.db (default <installRoot>/data/hccl_vm_data.db, derived as ../../data via __file__)
    config    <- setting.sim_config / bw_profile (use built-in defaults when missing, Section 7.2)
    ranktable <- ranktable.json in same directory as db (_find_ranktable default rule; consistent with host InitHvmCommEnv
                writing data/ranktable.json, cmd_base_utils.cc:437-450, no configuration needed)
    groups    <- db_reader.load_op_executions(db)            # Chapter 4
    topo      <- TopoResolver(ranktable)                     # Section 7.6
    for group in groups:                                    # per-operator (Section 5.1)
        config, dep_graph = graph_builder.build(group)      # Chapters 5, 6
        result = HcclSimulator(config, dep_graph, topo).run()   # Chapter 7
    profiling.dump(<plugin>/light_performance/data/<timestamp>/perf_profiling.json)   # Chapter 8
```

Comparison and selection basis against checker's resident process model (stdin single-line JSON command loop):

| Dimension | checker (resident model) | estimate (one-shot model) |
|------|--------------------|----------------------|
| Startup method | `plugin install` fork+pipe resident (`hccl_plugin.cc:74-156`) | `plugin run` `cd && entry` synchronous execution (`cmd_base_utils.cc:655-660`) |
| Trigger protocol | stdin JSON: `{"action":"status","payload":{"status":"finish","data_id":...}}` | No protocol: entry is start, input paths all by convention + manifest setting |
| data_id | Message carries (host `DumpData` generates) | Not passed, output directory uses runtime timestamp (Section 10.3 #13) |
| Lifetime | Reclaimed on stop command/pipe EOF/host exit | Process naturally exits, exit code is execution result |
| Adaptation reason | Validation triggered by host at flow end, may run multiple times | Simulation is offline batch: consumes already-landed db, runs once then exits |

> **Evolution direction** (not implemented this iteration): If "flow end host auto-triggers simulation" (resident model) is needed later,
> main.py adds stdin command loop reusing the same pipeline - host's message channel
> (`SendMessageToPlugin`) and resident startup (`RegisterPlugin`) are both ready; at that time, only
> switch @estimate in `RunUserPlugin` from the special-case branch back to the general message path.

Data contract (fixed conventions inside install tree, all aligned with existing mechanisms):

| Path | Source | Description |
|------|------|------|
| `data/hccl_vm_data.db` | Written during host flow execution | Simulation main input; estimate mode produces **single-rank task data** (phantom peer), naturally aligns with single-rank graph premise in Sections 4.6/5.1 |
| `data/ranktable.json` | Written by `InitHvmCommEnv` | eid->port resolution (Section 7.6); `_find_ranktable` default takes same directory as db, no explicit config needed |
| `plugin/light_performance/sim_config.json`, `bw_profile.json` | CMake installed with package directory (Section 3.4) | Section 7.2 format (units us/GBps) |
| `plugin/light_performance/data/<timestamp>/` | Created at main.py runtime | Output directory. **Must not be under `data/`**: host every start does `remove_all("data")` (`src/main.cc:32`), data directory content does not persist across sessions |

### 3.3 Command Line Design

Host integration mode (`plugin run` is simulation entry, replacing manual run_perf_sim.py execution):

```bash
# (1) Start environment in estimate mode (choose one)
hccl-vm start <topo> --estimate             # interactive mode, PS1 shows (estimate)$>
hccl-vm run <topo> --estimate <test command>     # one-shot mode
# (Inside an already started session, can also switch via hccl-vm mode estimate, subcmd_mode.cc)

# (2) Run business -> produce data/hccl_vm_data.db (single-rank task list)

# (3) One-click performance simulation (only available in estimate mode, Section 3.2 gate)
hccl-vm plugin run @estimate
# Artifacts: plugin/light_performance/data/<timestamp>/perf_profiling.json (+ trace / graph dump)
```

- The one-shot execution channel does not pass through command-line parameters; simulation parameters (target device, opIter filter, trace/dump switches, etc.)
  are configured via manifest `setting` section (Section 3.1); change config then rerun `plugin run @estimate` to take effect.
- `plugin install @estimate` is also executable (takes `RegisterPlugin` resident startup channel); main.py's one-shot
  run-and-exit is acceptable degraded behavior (Section 10.3 #14); normal use does not need install.

Independent debug mode (does not depend on host, convenient for offline db analysis; already implemented):

```bash
# Simulation + report (optional trace export, task graph dump)
python3 -m light_performance \
    --db hccl_vm_install/data/hccl_vm_data.db \
    [--config sim_config.json] [--bw-profile bw_profile.json] \
    [--ranktable ranktable.json] [--device 1] [--iter 0] \
    [--trace trace.json] [--dump-dir graph_dump] \
    [--max-trace-events 0] [-v] [-s]

# One-click simulation (run_perf_sim.py, outputs perf_profiling.json; shares pipeline with main.py)
python3 run_perf_sim.py hccl_vm_install/data/hccl_vm_data.db \
    [-o perf_profiling.json] [--config ...] [--bw-profile ...] [--trace-dir ...] [--dump-dir ...]

# One-click task graph dump (DOT + JSON, optional HTML; --no-sim skips simulation)
python3 dump_graph.py hccl_vm_install/data/hccl_vm_data.db [--html] [--no-sim]
```

| Parameter | Description |
|------|------|
| `--db` | hccl_vm_data.db path (required) |
| `--config` | Latency/bandwidth config (Section 7.2 format, units us/GBps; default uses built-in defaults) |
| `--bw-profile` | Bandwidth utilization config (default utilization is 100%) |
| `--jetty-num` | Override db-logged UB multi-channel (jetty) count, for multi-jetty utilization conversion (default per operator's `opDetails.ubMultiChannelNum`, see Section 7.2) |
| `--ranktable` | Topology file, for eid -> physical port resolution (default takes same directory as db's `ranktable.json`; when network tasks exist and no topology, **skips simulation, dumps only**) |
| `--device` | Target device (deviceId) for single-rank graph; default takes first device in opDetails; needed only when db contains multi-device data |
| `--iter` | Simulate only operators of specified opIter (default all) |
| `--trace` | Export Chrome tracing JSON (multiple operators split by `_<commId>_<opIter>`) |
| `--dump-dir` | Task graph dump directory (per-operator outputs .dot/.json/.html, including simulation timestamps) |
| `-v` / `-s` | Task-level detail / summary only |

### 3.4 Compilation and Installation Design (hccl_vm_install/plugin/light_performance)

Add install rule in `src/plugin/CMakeLists.txt` (following insight's directory installation precedent;
Python plugins need no build targets. Plugin directory is package directory, manifest.json/main.py/default configs all inside
package directory, single DIRECTORY rule carries all content):

```cmake
# estimate performance simulation plugin: plugin directory is light_performance package directory, install entire directory
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/light_performance"
    DESTINATION "plugin"
    FILE_PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
    PATTERN "__pycache__" EXCLUDE
    PATTERN "data" EXCLUDE)      # Runtime output directory does not enter install package
```

Comparison with existing plugin installation methods:

| Plugin | Installation Method | Artifact |
|------|---------|------|
| checker | `install(TARGETS checker DESTINATION plugin/checker)` + `install(FILES manifest.json)` (`src/plugin/checker/CMakeLists.txt:67-82`) | Build artifact + manifest |
| insight | `install(DIRECTORY insight DESTINATION plugin/visualization PATTERN frontend EXCLUDE)` (`src/plugin/CMakeLists.txt:26-39`) | Python directory + manifest |
| **estimate (this iteration)** | `install(DIRECTORY light_performance DESTINATION plugin)` (single rule; manifest/main.py/default configs all in package directory installed with package) | `plugin/light_performance/`: Python package + manifest (plugin directory and package directory unified, no nesting) |

Build and verify: After `bash build.sh` (internally `cmake && make && make install`, install prefix defaults to
in-repo `hccl_vm_install/`, `CMakeLists.txt:89-92`), check install artifacts:

```
hccl_vm_install/plugin/light_performance/
+-- manifest.json
+-- main.py
+-- __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
+-- sim_config.json / bw_profile.json
\-- simulator/ ...
```

- Packaging path: `build.sh --pkg` packages the entire `hccl_vm_install` as tar, the estimate plugin is distributed with it.
- Dependencies: python3 (>=3.8, with sqlite3 standard library) and **simpy** (discrete event simulation engine, current environment
  4.1.2, imported by `simulator/simulator.py`); deployment environment must be pre-installed (insight plugin also depends on python3,
  convention consistent), see Section 10.3 #16.

---

## 4. db File Reading and Parsing (Sub-Requirement 2 - Part 1, Focus of This Iteration)

### 4.1 db Location and Opening

- Path fixed as `<installRoot>/data/hccl_vm_data.db` (consistent with checker's `Loader::LoadOpTaskFile`).
- Read-only open: Python `sqlite3` standard library, `mode=ro` URI, to avoid conflicts with running hccl-vm writes:
  ```python
  conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
  ```
- Auxiliary library `hccl_sim.db` (same directory): only used to resolve `opDetails.commId` to communicator identity
  (`Communicator` table, `comm_id[128] + comm_hash` uniquely identifies), optional.

### 4.2 Table List and Association Model

hccl_vm_data.db is table-created by `OpDbOps::InitSchema()` (`src/runnerdb/modeldb/db_hccl_op_db_ops.cc:76-184`);
the tables consumed by light_performance are as follows:

| Table | Primary Key/Association | Content | Use |
|----|----------|------|------|
| `opDetails` | `id`; `pid` -> opTask table name | One row per operator per rank: pid/deviceId/rankId/commId/opIter/syncIter/streamId(main stream)/root/rankSize/srcRank/dstRank + `opDetail` BLOB (operator parameters) + `algorithmName` (algorithm selection result, DFX hijack/log dual-path intercepted and written back, see note below) + `ubMultiChannelNum` (UB multi-channel/jetty count, proxy level0 stub reads env var `HCCL_UB_MULTI_CHANNEL_NUM` and logs, unset/invalid becomes 1, see Section 7.2 multi-jetty conversion) | Operator grouping, main stream identification, operator parameters, algorithm selection result, bandwidth utilization conversion |
| `opMemInfo` | `opDetailId` -> opDetails.id | Addresses and sizes of input/output/ccl three memory segments | Direction determination (Section 5.5), data size validation |
| `opTask_P_<pid>` | `opDetailId` -> opDetails.id | SQE task queue: deviceId/streamId/taskType/isDone + `opTaskMeta` BLOB | **Main graph input** |
| `ccuChannels` | `(deviceId, dieId, channelId)` | CCU channels: src/dst die/device/rank + leid/reid(16B EID) + protocol + jettyNum + jettyId array | CCU channel -> peer/jetty mapping (Chapter 6) |
| `ccuInstrRes` | `(deviceId, dieId)` | Per-die microcode instruction space: instrCount + instrSpace BLOB (32K x 32B) | CCU microcode source (Chapter 6) |
| `ccuInstr` | `ccuInstrResId` -> ccuInstrRes.id | One row per microcode delivery (startId/instrInfoSize) | Only for sensing "new microcode delivered", decoding does not depend on it |
| `syncRecords` | - | host-side stream synchronization records | Not used this iteration |
| `JettyMaps` | `opDetailId` | eid mappings recorded per operator | Backup (not used when ccuChannels covers) |

> Table structure defined in `include/runnerdb/sim_op_db_types.h`; writer semantics in `src/runnerdb/tracedb/db_sim_op_db_ops.cc`.

> **algorithmName column source (algorithm selection result interception)**: dual-path interception, both last-wins writeback
> `sim::UpdateOpDetailAlgName`, mutually backup, write same value.
>
> 1. **DFX hijack stub (main path)**: After algorithm selection completes, business per-operator calls
>    `HcclDfxRegOpInfoByCommId(commName, &HcclDfxOpInfoCompat)` (hccl repository
>    op_common.cc); its `algTag` carries the selected algorithm name (format
>    `{opType_commName}_{algName}_{host|device}[_{dataType}...]`, graph mode
>    `Graph_{algName}_{...}`). Level=2 mode (default) only mounts level0+level2,
>    level1's dfx_stub is not in process; `src/proxy/level2/hcomm_api_stub.cc` with
>    level2 strong symbol preempts resolution (preload order precedes the real
>    libhcomm.so loaded by business via python plugin chain); after parsing the algorithm name from algTag and writing back, via
>    `DlsymRealWithFallback` (RTLD_NEXT + libhcomm.so SONAME fallback + dladdr
>    to prevent self-call) **forwards to real implementation, pass-through return value, does not change call semantics**. Structure layout replicated per
>    hccl_res_dl.h (`static_assert` locks algTag offset 68); content validation
>    (commId prefix + host/device suffix) filters op-begin and other registrations without algorithm names,
>    fallback for business header drift. In Level=1 mode, this stub is shadowed by level1 dfx_stub (behavior
>    same as old version), interception falls back to log path.
> 2. **Log interception (backup path)**: Tool is the implementer of business log lower interface (business
>    `DlogRecord` weak symbol resolves to `src/proxy/level2/log_stub.cc`). level0 stub
>    `RecordOpDbInfo` first inserts opDetails row (sets runnerdb global `g_currOpDetailId`) then calls
>    real business; inside business, selector-output INFO-level algorithm selection logs (`[HcclGetAlgRes] algName=` /
>    `SelectMinCost: selected algName=` / old selector `the selected algo type is ` three markers,
>    same operator multiple matches later writes overwrite) extracted via `DlogPrintStub` entry match. Interception executes before the message enters tool log output
>    (immune to `log_config.yaml` level); proxy layer also provides `acllogCheckDebugLevel`
>    always-release stub (immune to business-side log level threshold).
>
> Old library automatically has columns added by `OpDbOps::InitSchema()` via
> `ALTER TABLE`; plugin reading falls back to empty value for old library without columns.

**Operator grouping model**: One operator instance = rankSize opDetails rows under the same `(commId, opIter)` (one per rank),
each attached with one opMemInfo and a set of tasks (`WHERE opDetailId=?`, table name derived from that row's `pid` column as `opTask_P_<pid>`,
reference checker `QueryCompositeOpDetailByOpIter`).

### 4.3 opTask Dynamic Table and Multi-pid Handling

- Each write-task process creates its own table `"opTask_P_" + getpid()`; AICPU device-side process uses `getppid()` to share table with host.
- **Must enumerate all task tables** (multi-pid is the norm; early example db had `opTask_P_477776`/`opTask_P_477780` two tables, current example is single):
  ```sql
  SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'opTask_P_%'
  ```
- Consumption order: within the same operator by `id` ascending (= SQE delivery order; this is the basis for notify first-come-first-served pairing).

### 4.4 BLOB Decoding Specification (pack(1), little-endian)

All BLOBs are C structs as-is memcpy (no serialization framework); Python side uses `struct` module to decode.
The following layouts have all been empirically verified using `hccl_vm_install/data/hccl_vm_data.db` (from v5 collection onward, example
opTaskMeta length 190B; historical examples 189B/188B).

#### 4.4.1 opTaskMeta (190 bytes)

Structure definition shared with writer: `include/sim_profiling_defs.h` (profiling reconciliation info single-source definition)
and `src/plugin/checker/header/internal/task_meta_defs.h` (HcclTaskMetaData).

v5 collection structure (vs v4's 189B version): after `sqeGroupId` added `taskSrc` (1 byte) -
task source marker (`HcclTaskSrc`: 0=default SQE/aclrt native path, 1=DPU i.e. host CPU RoCE
`ibv_dpu_stub` collection path), for performance simulation DPU sequence merge pass to precisely split by source
(Section 5.1.2); taskData start offset 69 -> 70.

v4 collection structure (vs v3's 188B version): after `jettyId` added `sqeGroupId` (1 byte) - UDMA
SQE batch grouping marker (all WQEs in one doorbell share group), for performance simulation entry to batch-merge
(Section 5.1.1); taskData start offset 68 -> 69. Group number semantics: **circular increasing doorbell batch number 1..255**
(255 wraps, collection side `device_sqe_parse_stub.cc` maintains global `g_currSqeGroupId`,
`% 255 + 1`) - batch number unique within 255 window; other operators' doorbells are not visible in this operator's task list
but also consume batch numbers, so same-stream same-number must be same doorbell; **0 reserved for "unmarked"**
(HcclTaskMetaData construction default); counter never takes 0; merge entry treats all `sqeGroupId <= 0`
as not participating (Section 5.1.1).

v3 collection structure (vs v2's 185B version): after `taskType` added `HcclTaskProfInfo` substructure (3 contiguous bytes,
checker/runner pass through as opaque): `profTaskType` (aligned with hcomm TaskParamType value range 0~22) and
`linkType`/`transportType` (aligned with hcomm DfxLinkType value range 0~9 / DfxTransportType value range 0~4,
used for reconciliation with profiling data); taskData start offset 65 -> 68.

> **Implementation status**: `db_reader.py` auto-identifies layout by BLOB length - `>= 190B` per v5
> (`<BBBBQIQQIBB16s16s`, taskSrc decoded into `HcclTaskMetaData.src_type`; this branch
> **must precede v4 determination**, otherwise 190B would first hit the `>= 189` v4 branch causing eid/taskData
> offset misalignment), `>= 189B` per v4
> (`<BBBBQIQQIB16s16s`, sqeGroupId decoded into `RawTask.sqe_group_id`, src_type=0),
> `188B` per v3 (`<BBBBQIQQI16s16s`, sqe_group_id filled -1 not participating in merge),
> `185~187B` per old v2 (`<BQIQQI16s16s`, profInfo filled 0xFF=UNKNOWN) backward compatible,
> `< 185B` discard with warning. profInfo consumption (`task_type` display name + link/transport)
> **landed** (#18, Section 11.4).

v2 collection structure (vs earlier 169B version): header added `lcEid[16]` (explicit local EID),
taskData members once supplemented with `srcRankId/dstRankId` fields.

> **rank field rollback: srcRankId/dstRankId deprecated (current convention)**: taskData members' once-supplemented
> `srcRankId/dstRankId` fields have been rolled back and deleted, uniformly replaced with `srcDeviceId/dstDeviceId` - collection side
> (estimate mode, `device_sqe_parse_stub.cc` etc.) writes communicator rankId directly into
> `srcDeviceId/dstDeviceId` fields, i.e., **deviceId field carries rankId** (peer rank unresolved
> becomes `0xFFFFFFFF`). `db_reader.py` after unpack directly consumes as `src_rank_id/dst_rank_id` naming
> (reads `srcDeviceId` as `srcRankId` use). This is a hard switch: old
> BLOBs collected before field rollback (rank/device dual-field layout) are no longer supported.

```
Header 70B:  struct.unpack_from("<BBBBQIQQIBB16s16s", blob, 0)
+------+------+- -----------------+--------+------------+----------------------+
|Offset|Length|Field              |Type    |Sentinel    |Description           |
+------+------+- -----------------+--------+------------+----------------------+
|0     |1     |taskType           |u8 enum |-           |Redundant with table column taskType |
|1~3   |3     |profInfo           |substruct|0xFF       |HcclTaskProfInfo:      |
|      |  1   |.profTaskType      |u8 enum |0xFF        |aligned with hcomm TaskParamType |
|      |  1   |.linkType          |u8 enum |0xFF        |aligned with hcomm DfxLinkType |
|      |  1   |.transportType     |u8 enum |0xFF        |aligned with hcomm DfxTransportType |
|4     |8     |commId             |u64     |-           |Communicator ID        |
|12    |4     |rankId             |u32     |0xFFFFFFFF  |default via device->rank backfill |
|16    |8     |deviceId           |u64     |-           |Physical device ID     |
|24    |8     |streamId           |u64     |-           |Stream ID              |
|32    |4     |jettyId            |u32     |0xFFFFFFFF  |invalid = not network-bound |
|36    |1     |sqeGroupId         |u8      |0           |doorbell batch number 1..255 circular increasing (v4 added, Section 5.1.1); 0 = unmarked |
|37    |1     |taskSrc            |u8 enum |0           |task source (v5 added, Section 11.4.6): 0=default, 1=DPU(host ibv) |
|38    |16    |rmEid[16]          |bytes   |all 0       |Remote EID             |
|54    |16    |lcEid[16]          |bytes   |all 0       |Local EID (v2 added)   |
|70    |120   |taskData           |union   |interpreted per taskType, see below |
+------+------+- -----------------+--------+------------+----------------------+
```

taskData union internal layout (relative offset 69):

| taskType | Member | Format String | Fields (in order) |
|----------|------|--------|------|
| 0/1 (notify) | `NotifyTask` | `<IQIBB` | srcDeviceId(carrying rankId), notifyId, dstDeviceId(carrying rankId), notifyCount, protocol |
| 3 (transMem) | `TransMemTask` | `<IQIQQB` | srcDeviceId(carrying rankId), srcOffset, dstDeviceId(carrying rankId), dstOffset, len, protocol |
| 2 (reduce) | `ReduceTask` | `<IQIQQBBBB` | srcDeviceId(carrying rankId), srcOffset, dstDeviceId(carrying rankId), dstOffset, dataCount(actually byte size), dataType, reduceOp, protocol, resv |
| 4 (ccu) | `CcuTask` | `<BBHHHII` + `<13Q` | dieId, missionId, timeout, instStartId, instCnt, key, argSize, args[13] |
| 5 (aiv) | `AivTask` | `<Q` | launchIdx |
| 6 (sync) | `SyncStreamTask` | `<Q` | syncIdx |
| 7 (writeWithNotify) | `WriteWithNotifyTask` | `<IQIQQQBBBBB` | srcDeviceId(carrying rankId), srcOffset, dstDeviceId(carrying rankId), dstOffset, len, notifyId, isReduce, dataType, reduceOp, protocol, notifyCount |

> `len` and `dataCount` are both **byte sizes** (reduce's field name has historical reasons, actually used as size).
> `WriteWithNotifyTask.len` is also byte size; `isReduce=1` indicates write payload is reduction data
> (`dataType`/`reduceOp` are valid in this case), mapped to a single `Write_With_Notify` task (Section 5.2).

#### 4.4.2 taskType Enumeration Full Table

`HccLTaskMetaType` (`include/sim_common_defs.h:184-193`):

| Value | Enumeration | light_performance Handling |
|----|------|----------------------|
| 0 | `NOTIFY_WAIT` | -> local_wait / remote_wait (Section 5.2) |
| 1 | `NOTIFY_RECORD` | -> local_record / remote_record |
| 2 | `REDUCE` | -> memcpy / local_copy (reduction computation approximated, Section 5.7) |
| 3 | `MEM_CPY` | -> memcpy / local_copy (determined by jettyId) |
| 4 | `CCU_GRAPH` | -> CCU subgraph expansion (Chapter 6) |
| 5 | `AIV_GRAPH` | Not supported for now, skip with warning |
| 6 | `SYNC_STREAM` | -> skip, do not generate node (Section 5.6) |
| 7 | `WRITE_WITH_NOTIFY` | -> single-task mapping `Write_With_Notify` (network write + attached notify, not split, Section 5.2) |

#### 4.4.3 opDetails.opDetail (28 bytes)

`OpDetails` (`task_meta_defs.h:141-165`): `<HHH` header + union:

```
opType u16@0 | dataType u16@2 | reduceType u16@4
opV1: count u64@6 | dataType u16@14 | reduceOpType u32@16          (total 28B)
opV2: sendCount u64@6 | sendDataType u16@14 | recvCount u64@16
      | recvDataType u16@24 | extInfo u16@26                       (AllToallV class)
```

opType is `HcclCMDType` (ALLREDUCE=2, ALLGATHER=6, REDUCE_SCATTER=7, ALLTOALL=10, etc.,
complete enumeration in `src/plugin/checker/header/external/hccl_types.h:162-189`), used for profiling report's operator labeling.
`opExtInfo` BLOB (optional) stores V operator counts/displs, AllToallV send matrix, etc.; this iteration only passes through.

V1/V2 layouts distinguished by opType (level0 stub writer-side convention, `hccl_op_stub.cc`):
**V2 only AllToAll(10)/AllToAllV(8)/AllToAllVC(9)** (`BuildOpDetailsV2`, sendCount/recvCount
are this rank's send/receive total element counts); **AllGatherV(16)/ReduceScatterV(17) are V1 + opExtInfo**
(count = this rank side element count). db_reader decodes by opType choosing layout; V2 operators decode send/receive
two-side element counts and types (`send_count/recv_count/send_data_type/recv_data_type`);
`data_size/data_type` follows send side to keep old consumer compatibility; `reduce_op` set to 0.

#### 4.4.4 ccuChannels BLOB

| Field | Format | Description |
|------|------|------|
| `leid` / `reid` | Each 16B raw bytes | Source/destination endpoint EID (`URMA_EID_LEN=16`) |
| `jettyId` | `jettyNum x u32` little-endian | Contiguous jetty number array (C struct reserves `u32[64]`; example test jettyNum=1, jettyId=(1024,)) |
| `protocol` | int column | CommProtocol: HCCS=0/TCP=1/ROCE=2/UB_CTP=3/UB_TP=4/PCIE=5/SIO=6 |

Note: `srcRankId/dstRankId` two columns are actually filled with device_id (channel establishment precedes communicator, see writer side
`hccp_ccu_stub.cc:401-405` comment); rank semantics need correction via opDetails.

#### 4.4.5 ccuInstrRes instrSpace

- `instrSpace` = one die's **complete microcode instruction space snapshot**: `32K entries x 32B = 1 MiB`
  (`CCU_INSTRUCTION_NUM=32*1024`, `CCU_INSTRUCTION_SIZE=32`, `include/sim_common_defs.h:35-36`).
- The i-th instruction is at byte offset `i*32`; `instrCount = max(historical value, startId+instrCnt)`, **allows holes**.
- When reading, only take first `instrCount` entries; CCU task's `instStartId/instCnt` used directly as subscript range.
- `ccuInstr` table is only for runtime sensing (runner polls `COUNT(ccuInstr)+COUNT(ccuInstrRes)` changes);
  light_performance decoding does not depend on this table.

### 4.5 Intermediate Data Model (db_reader Output)

```python
@dataclass
class OpExecutionGroup:          # One operator instance (single-rank view: only target device's task data)
    comm_id: int
    op_iter: int
    op_type: int                 # HcclCMDType
    rank_size: int
    main_stream_ids: dict[int, int]      # deviceId -> main stream streamId (opDetails.streamId)
    tasks: list[RawTask]         # All SQEs of this operator, globally ordered
    mem_info: dict[int, MemInfo]         # deviceId -> input/output/ccl ranges
    channels: dict[tuple, ChannelInfo]   # (srcDev, srcDie, channelId) -> channel info
    instr_space: dict[tuple, list[bytes]]# (deviceId, dieId) -> 32B instruction list
    algorithm_name: str          # Algorithm selection result (opDetails.algorithmName, DFX/log dual-path intercepted writeback; old library empty string)
    ub_multi_channel_num: int    # UB multi-channel (jetty) count (opDetails.ubMultiChannelNum; old library/zero value per 1)
    send_count / recv_count: int # Send/receive two-side element counts (V2 layout operators; V1 same as count)
    send_data_type / recv_data_type: int # Send/receive two-side data types (same as above)

@dataclass
class RawTask:
    global_id: int               # Global sequence across opTask_P_* tables (by table pid numeric + in-table id order)
    op_detail_id: int
    device_id: int
    stream_id: int
    meta: HcclTaskMetaData       # 4.4.1 decode result (dataclass form)
```

`meta` contains v2 added fields: header `lc_eid` (explicit local EID, Section 5.5 local eid parsing prefers this);
NotifyTaskData/TransMemTaskData both carry `src_rank_id/dst_rank_id`
(under estimate collection convention unpacked from deviceId field carrying rankId value, see Section 4.4.1 rank field rollback).

### 4.6 Data Integrity Validation

For each `OpExecutionGroup`:

1. **Single-rank filter (design premise)**: Performance plugin only consumes **single-rank task data** (example db only device 1,
   this is design norm not data missing). If db contains multi-device data, filter opDetails and its tasks by target deviceId (`--rank`, default takes first device);
   `rankSize` only passed through as metadata; does not require `len(opDetails rows) == rankSize`.
2. **notify peer** (no backfill):
    - estimate collection convention (`device_sqe_parse_stub.cc`, two task classes consistent): notify's
      `srcDeviceId` (carrying rankId) = record side, `dstDeviceId` (carrying rankId) = wait side,
      both explicitly filled by emitter; peer rank unresolved becomes `0xFFFFFFFF`.
    - Earlier version's `dstDeviceId == 0` backfill fallback already deleted with rank field rollback (0 sentinel conflicts with legal
      rank 0, and emitter always explicitly fills, no backfill needed).
    - `WRITE_WITH_NOTIFY` (7)'s carried notify faces **write peer** (wait side not in this db, single-rank simulation premise); does not participate in local pairing; its `dstDeviceId` (carrying rankId) is the write destination.
    - Note: runner/checker consumer side (`DecodeAndExpandTaskMeta`) at decode boundary splits 7 into
      [write, NOTIFY_RECORD] two subtasks; light_performance **does not split** - single-task mapping
      `Write_With_Notify` (Section 5.2), ensuring simulation tasks correspond one-to-one with db tasks.
3. **Decode length validation**: `len(opTaskMeta) >= 185` (length-adaptive v4/v3/v2 layouts per Section 4.4.1),
   `len(opDetail) >= 28`,
   `len(instrSpace) >= instrCount*32`; if not met, discard the row and warn
   (checker does direct memcpy, here more lenient).

### 4.7 Decode Example (Empirical hccl_vm_data.db)

Example db (REDUCE_SCATTER, opType=7, dataType=4/FP32, count=8, rankSize=2, devType=6/950)
empirical decode result:

- `opTaskMeta` length constant 188B (v3 collection example; v4 collection onward is 189B, Section 4.4.1); taskType distribution: 4 (CCU_GRAPH) and 6 (SYNC_STREAM) alternating.
- opDetails only device 1 (rank 0) -> single-rank data (design norm); main stream stream 2.
- Task sequence (per opIter main stream):

```
stream 2 (main stream):  CCU_GRAPH(die=1, mission=0, instStartId=14, instCnt=100, argSize=13)
                   SYNC_STREAM(syncIdx=n)          # microcode delivery + stream sync
```

- CCU task's args match data directory `sqe_info_device_1_die_1_mission_0_startId_14.txt`
  item by item (args[0]=281475405602881, args[1]=281475405602945 ...);
  `args[0]/args[1]` empirically are opMemInfo's inputAddr/outputAddr.
- `ccuInstrRes` (dev 1, die 1) instrCount=114 = instStartId(14) + instCnt(100).
- In earlier 169B examples with MEM_CPY/NOTIFY empirically: MEM_CPY's srcOffset/dstOffset can both fall into
  opMemInfo's input/output/ccl ranges (`input contains output`, ReduceScatter's output is input's
  first 1/rankSize segment); network copy with jettyId, its offset also appears as "local range address"
  (communicator memory symmetric layout, peer buffer has same VA as local), so **direction determination cannot only look at address**
  (see Section 5.5). This conclusion applies to v2 data as well.

#### 4.7.1 v2 AICPU Mode Example (opTask_P_118800)

REDUCE_SCATTER, rankSize=8, count=32768 (FP32, output=128KB), 76 tasks,
stream 2 (main) / 6 (scheduling) / 7..13 (one stream per peer). Key differences from CCU mode example:

- taskType distribution: NOTIFY_WAIT x 30, NOTIFY_RECORD x 30, MEM_CPY x 8 (7 network write
  jet=1..7 + 1 local), REDUCE x 7 (local), SYNC_STREAM x 1; no CCU_GRAPH, `ccuChannels` empty.
- **Direction all determined by rank dimension** (Section 5.5 rule 1): 7 network MEM_CPY all are
  `srcRank=0/dstRank=1..7` (write), task offsets and opMemInfo ranges do not intersect (range inference fails),
  `direction_inferred=0`.
- **notifyId reused across ranks**: e.g., `RECORD(nid=30, dst=dev2)` (wait side at peer, skip) and
  local `WAIT(nid=30)` (record side at peer, not in this db) appear paired, two rounds each 7 x 2;
  local paired 16 pairs (nid 3/13 main <-> scheduling stream, 15..27 scheduling stream -> peer stream, 6..12 peer stream -> scheduling stream),
  unpaired local WAIT 14 -> billed by `remote_wait_delay` (Section 5.4).
- lcEid/rmEid match ranktable.json verbatim (`...0040...01010001` -> port `1/0` etc.),
  network task ports all resolvable.

### 4.8 eid->rank Index (`build_rank_index`, Peer Rank Resolution)

Peer rank resolution (Section 5.5 SQE fallback, Section 6.7 CCU main path) shares one eid(hex) -> rank_id
index, construction priority and basis:

1. **RankEidMap table (main path)**: runnerdb `hccl_sim.db` (same directory as hccl_vm_data.db)
   `RankEidMap` table. estimate mode Communicator table has only this rank one row; peer rank
   takes this table as authoritative (`sim_models.h` comment; host precedent `GetRmtRankIdByEid`,
   `device_sqe_parse_stub.cc:889`). Row BLOB layout consistent with `sim::RankEidMap`:
   `id(u64) + rankId(u32) + eid(16B) + pad(4B)`; eid is "hex string byte order",
   same source and order as `ccuChannels.leid/reid`; Python side by `bytes.hex()` direct equality match.
   This table written by ranktable.json parsing (`topo_ascend_cluster_parser.cc:229`);
   mock-comm scenario entirely overwritten along with communicator initialization, so authoritative.
2. **ranktable.json (fallback)**: When RankEidMap empty / `hccl_sim.db` missing / table missing, expand
   `rank_list[].rank_id <-> level_list[].rank_addr_list[].addr` (addr_type=EID)
   to build isomorphic index. Empirically the two paths' indexes are fully consistent (8 rank x 9 eid = 72 entries).

Read-only open (`file:...?mode=ro` URI); read failure (including table not existing) degrades to fallback rather than error.
Note **`ccuChannels.srcRankId/dstRankId` cannot be directly used as communicator rank**: writer side
(`hccp_ccu_stub.cc:401-405`) explicitly fills global endpoint deviceId (channel establishment precedes
communicator; same channel can be reused by multiple domains), with opDetails.rankId numbering system having offset;
checker deprecated this field for this reason (`g_allRankChannelInfo` only stores dstDeviceId/dstDieId).

---

## 5. SQE Task Graph Generation (Sub-Requirement 2 - Part 2, Focus of This Iteration)

### 5.1 Overall Pipeline

```
OpExecutionGroup
   |
   v
0 SQE batch merge (sqe_group_merge.py, Section 5.1.1): within same (stream, sqeGroupId) batch, UDMA data tasks merged to 1
   |
   v
0' DPU sequence merge (sqe_group_merge.py, Section 5.1.2): DPU large message N-1 leading blocks + tail block WRITE_WITH_NOTIFY merged to 1
   |
   v
1 Per SQE -> Task node (Section 5.2 mapping table; CCU_GRAPH handed to Chapter 6 for expansion)
   |     stream_id: int (SQE stream directly uses original streamId; CCU skeleton task also belongs to its SQE original stream, Section 6.5)
   v
2 stream serial chain: group by (deviceId, streamId), global_id ascending -> dep_graph.add_stream_chain
   v
3 notify pairing: cross-rank tasks do not participate in pairing (nodes retained); only local notify first-come-first-served -> dep_graph.add_match (Section 5.4)
   v
4 Network task parameter completion: direction inference / eid resolution (Section 5.5)
   v
5 CCU subgraph expansion (Chapter 6): CCU_GRAPH placeholder -> microcode instruction interpretation -> internal nodes + CKE pairing
   v
6 Output assembly: (SimConfig, DependencyGraph) -> HcclSimulator(config, dep, topo)
```

### 5.1.1 SQE Batch Merge (Pre-Graph Preprocessing)

Collection-side UDMA one doorbell (SQE) can carry multiple WQEs; tasks still logged per WQE, and
marked with `sqeGroupId` for batch membership (circular increasing doorbell batch number 1..255, 255 wraps, never takes 0; 0 = construction
default "unmarked", Section 4.4.1). Before graph, split batches by **(streamId, sqeGroupId)** - split key explicitly
includes stream; within batch, all data tasks (MEM_CPY/REDUCE/WRITE_WITH_NOTIFY) merged to one, aligned with
hcomm batch transfer profiling's `ExecProfilingAll` metric - three entries (`pipeline.py`/
`__main__.py`/`dump_graph.py`) uniformly call, ensuring simulation and dump metrics consistent:

- **Stream-grouped isolation guarantee**: Split key includes streamId, **cross-stream tasks never merged into same batch** (one doorbell
  originally belongs to only one stream; single doorbell batch not split by key); group number (doorbell batch number) unique within 255 window
  - other operators' doorbells are not visible in this operator's task list but also consume batch numbers, so **same-stream same-number must be
  same doorbell**, not mis-merge across doorbells (multi-operator db safe).
- **Description information** (type/address/endpoint/notifyId/reduction parameters/profInfo) takes batch's **last** data
  task; **length takes full batch sum** - consistent with `ExecProfilingAll(loc[insNum-1], rmt[insNum-1],
  totalSize, transferOp[insNum-1])` field by field; trace/dump display name similarly takes last task's
  op name (Section 11.4.1). Within batch mixed write/read direction or different dataTypes similarly follow last task
  (single SQE batch usually homogeneous; mix is rare scenario).
- **Do not participate in merge**: notify/SDMA/sync tasks (notifyId unique, pairing depends on order; local tasks
  retained as-is, do not affect batch boundary); unmarked/no-group-number tasks (`sqeGroupId <= 0`: 0 = unmarked,
  -1 = v3 and earlier BLOB has no such field) retained as-is and **cut current batch** (cannot determine batch membership, conservatively
  end current batch); UDMA non-data tasks (inline write notify) only participate in batch splitting; those interleaved within batch are sequenced after
  the merged task (notify causality).
- Merged task occupies the first data task's slot in batch (batch head); split key includes stream guaranteeing batch-internal tasks in same
  stream; stream serial chain (Section 5.3) and notify pairing (Section 5.4) relative order unaffected;
  local (non-UDMA) tasks do not participate in merge and do not reset batch boundary (retained in place, do not affect batch continuity).

### 5.1.2 DPU Sequence Merge (Pre-Graph Preprocessing)

DPU (host CPU RoCE) path's large message WriteWithNotify sequence merge, executes after 5.1.1
(`pipeline.py` unified call). Rules and semantics in Section 11.4.6: scan in DPU stream order
MEM_CPY chains with **src/dst dual addresses contiguous** (only `src_type==1` DPU-source data tasks participate;
other tasks retained as-is, do not cut chain); the address-contiguous WRITE_WITH_NOTIFY tail wraps up and merges to one -
**tail task is natural terminator**; two independent WriteWithNotify calls cannot cross-merge (each call's
tail block necessarily closes the current chain); mis-merge only has one theoretical scenario: "independent pure write happens to abut WriteWithNotify start address"
(semantic equivalent to a contiguous data + tail notify; simulation result no difference). Chains without tail not merged
(independent pure writes kept per-task `RDMASend`); merged task occupies tail task slot (notify causality); merged-in
leading blocks removed from original positions; unmerged tasks keep original stream order.

**Single-rank graph strategy (design premise, main difference from checker)**: db has only single-rank task data;
DependencyGraph only describes that rank's tasks and dependencies. **Cross-rank pairing edges (post-wait relations)
all not built (task nodes retained, approximated by fixed remote latency)**:

- SQE level: waitDev != this device's RECORD (cross-rank post), recordDev != this device's WAIT (cross-rank wait)
  **retain task nodes** - graph normally by stream order (src!=dst expresses remote, takes fixed remote latency), but **do not establish
  any pairing edges** - only care about intra-rank post-wait pairing (Section 5.4).
- CCU level: instructions that post to remote CKE, waits that depend on remote CKE set **also generate task nodes**
  (Notify_Record/Notify_Wait, src=local, dst=INVALID -> fixed remote latency), but do not establish
  cross-rank pairing edges (Sections 6.4/6.6).
- **Retain cross-rank data transfers** (direction=write/read MEMCPY): they are the main body of this rank's view of network
  bandwidth usage; only synchronization dependency edges are ignored; transfer itself not ignored.
- makespan semantics: single-rank view's operator end-to-end duration = this rank's computation/copy + this rank's initiated network transfer
  + this rank's sync wait; when peer is ready not modeled (equivalent to assuming peer instantly ready).

**Simulation granularity**: default per-operator (each `(commId, opIter)` one DependencyGraph, one simulation, one record),
consistent with checker's per-operator validation metric; `--all-in-one` optional full mode (on the same stream, multiple operators naturally serial).

### 5.2 taskType -> hccl_sim Task Mapping

Align with HCCLSIM new task model (`TaskType`: `NOTIFY_RECORD`/`NOTIFY_WAIT`/`REDUCE_INLINE`/
`MEMCPY`/`UB_INLINE_WRITE`/`UB_WRITE_OR_READ`/`WRITE_WITH_NOTIFY`, details in Section 7.3):

| db taskType | Determination Condition | hccl_sim TaskType | Task Parameters | Latency Model |
|-------------|---------|-------------------|-----------|---------|
| NOTIFY_RECORD (1) | waitRank == local rank | `NOTIFY_RECORD` | notify_id=int, src=dst=local_rank | fixed + trigger Event |
| NOTIFY_RECORD (1) | waitRank != local rank | `NOTIFY_RECORD` (cross-rank post: node retained, not paired) | notify_id=int, src=local_rank, dst=peer rank (unresolved falls back INVALID) | fixed remote_record_delay, does not trigger Event |
| NOTIFY_WAIT (0) | recordRank == local rank | `NOTIFY_WAIT` | same as above | wait for Event |
| NOTIFY_WAIT (0) | recordRank != local rank | `NOTIFY_WAIT` (cross-rank wait: node retained, not paired) | notify_id=int, src=peer rank (unresolved falls back INVALID), dst=local_rank | fixed remote_wait_delay |
| MEM_CPY (3) | jettyId == INVALID | `MEMCPY` | size=len, src/dst_rank=taskData endpoint rank (unfilled falls back local) | HBM dynamic shared bandwidth |
| MEM_CPY (3) | jettyId valid, write direction | `UB_WRITE_OR_READ` | size=len, src_rank=local, dst_rank=peer rank, src/dst_eid, jetty_id, protocol, is_write=True | port write direction bandwidth sharing |
| MEM_CPY (3) | jettyId valid, read direction | `UB_WRITE_OR_READ` | size=len, src_rank=peer rank, dst_rank=local, src/dst_eid, jetty_id, protocol, is_write=False | port read direction bandwidth sharing |
| REDUCE (2) | jettyId == INVALID | `REDUCE_INLINE` | size=dataCount, data_type=task-level dataType enum name, src/dst_rank same as MEMCPY local branch | HBM dynamic shared bandwidth |
| REDUCE (2) | jettyId valid | same as MEM_CPY network branch (Section 5.7 approximation) | same as above (including data_type) | same as above |
| WRITE_WITH_NOTIFY (7) | Always write direction (local src -> remote dst, no direction inference) | `WRITE_WITH_NOTIFY` (**single task, not split**) | size=len, notify_id=notifyId, src_rank=local, dst_rank=peer rank (unresolved via eid->rank index fallback), src/dst_eid, jetty_id, protocol, is_write=True; when isReduce=1, data_type=dataType enum name | port write direction bandwidth sharing + after write lands, notify fixed record latency (remote) |
| CCU_GRAPH (4) | - | (expand, Chapter 6) | - | - |
| AIV_GRAPH (5) | - | skip | - | warning |
| SYNC_STREAM (6) | - | **skip** (host-side stream sync point, Section 5.6) | - | - |

> waitRank/recordRank are taskData endpoint rankIds (deviceId field carries, src=record side,
> dst=wait side, Sections 4.4.1/4.6), used only for **determining local/cross-rank**: cross-rank notify tasks in
> mapping phase retain nodes but do not enter pairing (Section 5.1 strategy);
> pairing only occurs between local notifys. notify's local/remote expressed in new model as `src_rank == dst_rank`
> (`is_local_notify`).
> **WRITE_WITH_NOTIFY (7) single-task mapping** (one-to-one with db task, not split into write + NOTIFY_RECORD
> two subtasks): write direction always write (emitter convention src local -> dst peer, no Section 5.5 direction inference);
> takes port write direction bandwidth sharing; after write data lands, sends notify to peer, billed by fixed record latency
> (peer wait not in this db, single-rank simulation premise, same as cross-rank record semantics; self-write src==dst edge
> case per local record latency); `notifyId` retained with task (trace/dump display);
> when `isReduce=1`, only affects `data_type` display (same as REDUCE branch enum name mapping).
> **Network task rank directly consumes taskData** (deviceId field carries rankId; emitter convention:
> write=src local/dst peer, read=src peer/dst local): write fills `dst_rank=peer rank`,
> read fills `src_rank=peer rank`; when peer rank unresolved (`0xFFFFFFFF` or equals local) falls back INVALID
> (Section 10.1). Direction explicitly carried by `is_write`; simulator no longer relies on `src_rank==0` heuristic
> (Section 10.3 #1 fixed; task.txt old input path still takes heuristic).
> Local eid prefers SQE explicit lcEid (v2 collection added); fallback via jettyId -> ccuChannels channel
> leid derivation (Section 5.5, Section 10.1).
> This table's TaskType is internal latency model classification; trace/dump task display name already switched to
> profInfo.profTaskType's op name (Section 11.4.1 mapping table, Section 10.3 #18 landed);
> correspondence with this table's TaskType per Section 11.4.1 "Simulation TaskType" column.

### 5.3 Stream Serial Chain Construction

- Group by `(deviceId, streamId)`, within group by `global_id` ascending (= SQE delivery order) forming chain.
- Simulator starts one simpy process per chain executing serially (Chapter 7) - **does not need checker's mainStart virtual
  starting point and main stream special case**: simulation focuses on end-to-end makespan; each chain starts from time 0 in parallel;
  chain head task's WAIT naturally forms actual synchronization with main stream.

### 5.4 notify Pairing Algorithm (Local Only)

**Cross-rank post-wait edges all ignored** (Section 5.1 strategy): before pairing, first classify per Section 4.6 peer backfill result;
cross-rank notify tasks (waitDev != this device's RECORD, recordDev != this device's WAIT) in Section 5.2
mapping phase retain nodes (graph by stream order), but do not enter pairing. Therefore pairing key simplifies from checker's triple to single notifyId:

```
checker semantics:   pairing key = (recordDeviceId, waitDeviceId, notifyId)   <- multi-rank
this plugin:         first filter cross-rank -> pairing key = notifyId                    <- single-rank
```

Consistent with checker's `AddLocalNotifyEdges` (delivery order simulation + first-come-first-served consumption,
`task_graph_generator_v3.cc:847-946`) in principle; light_performance simplifies to
**two-pass scan + first-come-first-served**:

```
Input: all local notify tasks of one operator (cross-rank nodes retained but not participating in pairing)
First pass: scan by global_id ascending
    RECORD -> pending_records[notifyId].append(task_id)                        # queue
    WAIT   -> if pending_records[notifyId] non-empty:
                  rec = queue head pop; dep_graph.add_match(wait_id, rec_id)          # first-come-first-served
              else:
                  unmatched_waits.append(wait_id)                               # wait for "later" record
Second pass: handle unmatched_waits (WAIT appears before RECORD scenario)
    Replay by global_id order: when WAIT unmatched, suspend; when RECORD arrives, pair with earliest suspended WAIT
    (pairing-successful RECORD synchronously removed from pending_records, ensuring diagnostic stats accurate)
Wrap-up: still-residual WAITs -> normal (see below); still-residual RECORDs -> warning, normal data should not have
```

Notes:

- db id order is delivery order, and stream serial chain already guarantees "tasks after WAIT on same stream do not execute first",
  so complete queue simulation and deadlock detection of checker not needed (deadlock in simulation exposed as makespan anomaly / task not executed).
- Same notifyId multiple RECORD/WAIT (e.g., in early 169B example n11/n15 repeatedly appear on stream 6, main stream two iterations)
  naturally paired by first-come-first-served rule; **notifyId reuse allowed**.
- **Unpaired local WAIT is normal semantics**: in v2 AICPU example (Section 4.7), notifyId has cross-rank reuse -
  local `RECORD(nid=30, dst=dev2)` (wait side at peer, retained but not paired) and local `WAIT(nid=30)`
  (record side at peer, not in this db) appear paired; this WAIT in simulation billed by `remote_wait_delay` fixed latency
  (Section 7.6); warning only for prompt.
- In early 169B example cross-rank pair: `RECORD(n18, s1->d2)` (notify dev2) and `WAIT(n18)` (backfill recordDev=2)
  both retain nodes, do not build cross-rank pairing edges: graph by stream order, `src!=dst` takes fixed remote latency;
  data transfer tasks on stream 8 (MEMCPY(jet=3) etc.) unaffected.

### 5.5 Network Task Direction and eid Resolution

**Direction determination** - network data tasks uniformly mapped to `UB_WRITE_OR_READ`; direction determines port billing
direction (is_write) and rank endpoint values. taskData endpoint rankId carried by deviceId field
(Section 4.4.1 rank field rollback); emitter convention (`device_sqe_parse_stub.cc`):
write = src local/dst peer, read = src peer/dst local:

| Priority | Rule | Basis |
|--------|------|------|
| 1 | srcRank == local and dstRank != local -> write; dstRank == local and srcRank != local -> read | Collection endpoint rankId (deviceId field carries, Section 4.4.1); empirical example (Section 4.7) all determined by this, no inference |
| 2 | When rank not distinguished (equal or both unfilled), by memory range: dst falls into local output range -> read; src falls into local input range -> write | Empirical example summary (Section 4.7): when task offsets and opMemInfo ranges do not intersect, fails |
| 3 | Fallback: default write; diagnostic stats `direction_inferred` count | - |

> Network tasks on Task explicitly carry `is_write` (Section 10.3 #1 fixed; CCU path Section 6.7 synchronously fixed);
> simulator accordingly chooses write/read billing direction and port to query eid (`src_eid`/`dst_eid`); no longer relies on
> `src_rank==0` heuristic; task.txt old input path lacks this field, falls back to heuristic.
> `WRITE_WITH_NOTIFY` (7) **does not participate in direction determination**: semantics always write (local src -> remote dst);
> at graph time directly sets `is_write=True`.

**eid resolution** - network tasks query ranktable via eid for physical port (`TopoResolver.get_ports_by_eid`):

```
SQE:        local eid prefers SQE explicit lcEid(v2 added); fallback via jettyId
            --ccuChannels(this device outbound channel)--> leid(local eid)
            peer prefers channel reid, fallback back to SQE rmEid; when channel missing, local eid degrades to any local channel leid
CCU microcode: channelId -> channels[(dev,die,channelId)] -> leid/reid + jettyId list
            When channel missing, local eid prefers that mission SQE's lcEid, degrades to any same-die channel leid
write direction: simulator queries task.src_eid -> ports; read direction queries task.dst_eid -> ports
```

- `protocol`: channel protocol column -> HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO (CommProtocol name,
  display only); `link_type`/`transport_type`: task profInfo (DfxLinkType/
  ProfTransportType name, Sections 11.4.2/11.4.3, `_add_task` uniform fill, reconciles with msprof L1).
- When local eid unresolvable, warn + `eid_unresolved` count; when network tasks exist but no ranktable,
  simulation overall skipped (TopoResolver directly throws for missing eid, see Section 10.3).

**Peer rank resolution** - SQE network task's peer rank (`_peer_rank`): taskData endpoint rankId
(deviceId field carries, emitter already resolved per `GetRmtRankIdByEid`) preferred; when unresolved
(UINT32_MAX/local value), via eid->rank index (Section 4.8) by peer eid fallback; still none then
INVALID + `rank_unresolved` count.

### 5.6 SYNC_STREAM Handling

host-side stream sync point (example main stream once per iteration), indicates host waiting for this stream to complete,
**does not represent device-side task**: does not generate task node, takes no time (makespan is stream chain last task end time).

- Diagnostic stats `skipped_sync` count;
- Previously approximated by "unpaired local `NOTIFY_RECORD` fixed latency placeholder"; removed because it would introduce spurious
  `local_record_delay` at stream tail (and appears in trace as pseudo notify task with notify_id=invalid).

### 5.7 REDUCE Approximation

Reduction = network transfer + computation. Current model approximates by data transfer:

- Network-bound (jettyId valid): billed as `UB_WRITE_OR_READ` port bandwidth latency
  (direction per Section 5.5 determination).
- Not network-bound (jettyId INVALID): mapped to `REDUCE_INLINE`, billed by HBM bandwidth dynamic sharing
  (`size / hbm_bandwidth`); computation latency not yet separately listed (future can calibrate by `dataType x dataCount`,
  see Section 10.1 data_type field gap).

### 5.8 Comparison and Simplification vs checker Graph

| checker Stage | light_performance | Reason |
|--------------|-------------------|------|
| mainStart virtual node + main stream in-edge special case | Removed | Simulation only needs makespan; chain head WAIT already expresses synchronization |
| AddLocalNotifyEdges queue simulation + deadlock detection (GRAPH_DEADLOCK) | Two-pass scan first-come-first-served, no deadlock detection | No validation requirement; anomaly exposed as simulation result |
| AddInterRankNotifyEdges cross-rank pairing (full graph BFS simulation) | **Completely removed**: cross-rank post-wait pairing edges all ignored (notify task nodes retained, graphed by stream order) | Single-rank graph (core difference) |
| MemSlice address resolution + memory layout table | Only range attribution for direction determination (optional) | Simulation only needs len, not precise memory model |
| CheckSlaveTaskQueue secondary stream structure validation | Removed | Not a validation tool |
| Memory conflict / semantic / sync conflict / reachability / single-task checks | All removed | Not a validation tool |
| CompactSyncNodes (SYNC_STREAM merge) | Skip, do not generate node (host-side sync point, not device-side task) | makespan is stream chain last task end time (Section 5.6) |
| AIV subgraph expansion | Not supported (warning skip) | Requirement explicitly not considered for now |
| CCU subgraph: queue parallelization transformation, loop merge, resource conflict validation | Queue parallelization simplified implementation (Section 7.4.1: assembly-phase skeleton/unit split + fork/completion-set join and attach extra edges + `#v` virtual stream); loop merge equivalently implemented (Section 6.5); resource conflict validation removed | Parallelization eliminates queue serialization distortion (Section 7.4.1); no validation, node count controllable |

### 5.9 Output Assembly

```python
def build(group: OpExecutionGroup, sim_config: SimConfig | None = None
          ) -> tuple[SimConfig, DependencyGraph]:
    dep = DependencyGraph()
    # (1) SQE -> Task (CCU_GRAPH temporarily placeholder)
    # (2) stream chain (int stream id)
    # (3) notify pairing (add_match)
    # (4) direction/eid completion
    # (5) CCU expansion (ccu_graph_builder.expand_ccu_graph(..., local_rank, alloc_stream))
    return sim_config, dep
```

- `SimConfig` source: JSON specified by `--config` / manifest setting (Section 7.2 format, units
  us/GBps), plus bandwidth utilization config.
- Topology: `TopoResolver` constructed by CLI layer (ranktable location per Section 3.3) and passed to simulator.
- Output immediately sendable to `HcclSimulator(config, dep_graph, topo).run()` (returns dict, see Section 7.1).

---

## 6. CCU Mode Task Graph Generation (Sub-Requirement 3, Focus of This Iteration)

### 6.1 CCU Data Organization in db

```
SQE with taskType=4 (CCU_GRAPH) in opTask_P_<pid>
    \- taskData.ccu: {dieId, missionId, timeout, instStartId, instCnt, key, argSize, args[13]}
            |  instStartId/instCnt define instruction execution window [instStartId, instStartId+instCnt)
            |  args[13] consumed one by one by LOAD-class instructions (every 13 args read advances sqeIndex)
            v
ccuInstrRes (deviceId, dieId) -> instrSpace: 32K x 32B microcode instruction space
ccuChannels (deviceId, dieId, channelId) -> {dstDeviceId, dstDieId, leid/reid, jettyId[]}
    \- channel register values -> peer location and jetty
```

- SQEs grouped by `(dieId, missionId, queue)` (single-device graph, deviceId constant; queue is SQE stream)
  into dispatch context (checker's `CcuMissionKey` + latest-node-append semantics,
  `task_meta_translator_v3.h:47-62`): same-key SQEs only merged into same context **when contiguous with previous** (previous
  `instCnt==13` and `instStartId` end-to-end, `IsContiguousCcuSqe`,
  `task_meta_translator_v3.cc:213-231`); otherwise **new context** -
  multiple dispatches of same mission (instruction window wrap re-send, typical like "head SQE (13 Load
  SqeArgs) + body SQE" sent twice in a row) each independently expanded: each context fully replays its own instruction window
  (including Load SqeArgs header, re-initializing loop count registers with **this dispatch's args**),
  avoiding second dispatch using residual register state from previous causing data transfer loss (Section 10.3 #20).
  - **args consumption** (checker `CcuGraphStateV3::GetSqe` semantics): within context, SQEs arranged by dispatch order;
    args consumed slot by slot - every 13 args read (read to last slot `sqeArgsId==12`) cursor advances
    to next SQE; typical layout head SQE's 13 Load SqeArgs consume head SQE's args[0..12];
    body SQE window's subsequent Load instructions consume body SQE's args.
  - **Register/CKE bitmap globally shared across contexts** (checker `AllRankParamRecorder::Global()`
    semantics): later dispatch's wait can be resolved by previous dispatch's leftover CKE bits / residual post accounting,
    consistent with checker multi-graph shared recorder behavior.
  - **Same queue's dispatch contexts strictly serially advance by dispatch order** (checker queue semantics: single
    queue strictly by instruction order; same-stream later TaskCcuGraph must wait previous to complete); cross-queue polling
    joint advance (Section 6.6).
- Subgraph attachment point: CCU_GRAPH SQE's position in stream chain is subgraph entry; after expansion, the placeholder is removed
  (task graph does not introduce any placeholder nodes); subgraph attached via extra edges with SQE predecessor/successor tasks (Section 7.4.1).

### 6.2 Microcode Instruction Format (32B Fixed Length)

Same source as `${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode`
(this repository reference copy: `ai_skills/skills/convert-ccu-microcode/references/ccu_microcode_v1.h`):

```
CcuInstr = CcuInstrHeader (2B) + union { V1(A5) | V2(A6) | V3 } (30B)

CcuInstrHeader bit field (little-endian u16):
    code     : 11 bit   (header & 0x7FF)      instruction code
    type     :  4 bit   ((header >> 11) & 0xF) instruction major class
    reserved :  1 bit

Instruction major classes:  LOAD_TYPE=0x0 | CTRL_TYPE=0x1 | TRANS_TYPE=0x2 | REDUCE_TYPE=0x3
```

Chip generation -> instruction set version mapping (checker `GetCcuInstrVersion`, `ccu_task_transform_v3.cc:175-184`):
`Ascend950 -> A5/V1`, `Ascend960 -> A6/V2`, other devType falls back A5; derived from Device's soc_version
(hccl_sim.db) or `opDetails.devType` (`expand_ccu_graph` selects by `dev_type`
`_MissionContext`/`_MissionContextA6` interpreter). **Both instruction sets implemented** (A6 ported from
checker `ccu_task_transform_instruct_a6_v3.cc`, Section 11.3.3); A5/A6 instruction set differences overview in Section 6.8.

### 6.3 Instruction Type -> Task Mapping

#### 6.3.1 Summary Table

| Instruction Major Class | Representative Instructions | light_performance Behavior | Output Task |
|----------|---------|----------------------|-----------|
| LOAD (register write) | LoadSqeArgsToXn/GSA, LoadImd, LoadX, arithmetic (A6 Add/Sub/...) | Updates register model; data plane **does not generate node**; A6 LOAD class generally carries `setCKE` tail field (mask non-zero -> local `NOTIFY_RECORD`, checker `ProcessSetMask` semantics); A6 Load/Store (HBM<->Xn data plane) does not model HBM content, register not backfilled (only retains setCKE sync semantics, warning leaves trace) | - (when A6 setCKE non-zero, Notify_Record) |
| CTRL: SetCKE/SetCKBit | Sets CKE bit | Local -> `NOTIFY_RECORD` (registers CKE accounting, with mask); remote target (cross-rank post, SyncCKE etc.) -> remote `NOTIFY_RECORD` (not paired, Section 6.6) | Notify_Record |
| CTRL: Wait/Jmp/Fence | Waits CKE bit / jumps / fence | CKE wait (A5 embedded waitCKE field): when bits insufficient, suspends that queue pointer; when bits complete, generates `NOTIFY_WAIT` (split by contributing bits, with mask); wait depending on remote bits -> remote `NOTIFY_WAIT` (not paired, fixed latency, Section 6.6). A6 independent Wait instruction (Xn value compare spin, checker `TransformWaitInstr`): satisfied passes through no node; unsatisfied = depends on peer remote register write (SyncWtX/TransMem notify) -> top-level by stall resolution generates remote `NOTIFY_WAIT` (fixed `remote_wait_delay`); within loop body directly generates node and continues. Jmp: A5 evaluates by `Xn[conditionXnId] != expectData` (checker `TransformJmpInstrA5` semantics); A6 supports 6 condition codes (==/!=/>/>=/</<=, >=6 unconditional) + relative/absolute two modes (jumpMode, target `Xn[relTarInstrXnId]`); target limited to current instruction window, out-of-range warning pass-through; within loop body pass-through; with step budget defense against backward jump infinite loop. Fence (A6 only): checker does not support, warning pass-through (queue serial model order naturally guaranteed) | Notify_Wait |
| CTRL: ClearCKE | wait + clear bit | wait semantics; A6 per checker `ClearWaitMask` semantics clears this consumption's wait bits (`clearMask != 0` is checker unsupported scenario, warning + tolerant clear target bits) | Notify_Wait |
| CTRL: Loop/LoopGroup | loop | loopCnt from register; **merge-expand** (Section 6.5, on failure falls back full expand); A6 Loop `mode==1` is conditional expansion (`wishCKEBit == Xp.loopCtxId + 1024` hit to execute; always takes full expand path) | None (body instructions mapped per this table; under merge path, data task size x total iteration count) |
| TRANS (transfer) | LocMem<->LocMS/RmtMem/RmtMS (A5), local 4-code + unified TransMem dmaOpCode (A6) | Mapped by role (Section 6.3.2) | `Memcpy`/`Ub_Write_Or_Read` |
| TRANS: Sync (cross-rank sync) | SyncCKE/GSA/Xn (A5), SyncXnWtx/SyncAtx (A6) | Remote set/notify -> remote `NOTIFY_RECORD` (not paired, Section 6.6); value sync as remote register write, does not build node | Notify_Record (remote) |
| REDUCE | Add/Max/Min | Same as SQE REDUCE approximation (Section 5.7); MS is rank local buffer, always local HBM reduce | `Reduce_Inline` |

#### 6.3.2 TRANS Transfer Role Determination (Consistent with checker `ccu_loop_merge_v3.cc:73-144`)

By src/dst device membership:

| src | dst | Role | hccl_sim Task |
|-----|-----|------|---------------|
| Local | Local | LOCAL_COPY | `Memcpy` (HBM dynamic sharing) |
| Local | Remote | WRITE | `Ub_Write_Or_Read` (src_rank=local, dst_rank=peer, is_write=True) |
| Remote | Local | READ | `Ub_Write_Or_Read` (src_rank=peer, dst_rank=local, is_write=False) |

> Cross-die/device read-write two directions uniformly mapped to `Ub_Write_Or_Read`; direction expressed via `is_write`/trace
> `direction` field; port billed independently by write/read direction; peer rank via channel
> reid -> eid->rank index resolution (unresolved is INVALID). To preserve microcode-distinguishable buffer
> information, CCU-converted transfer tasks carry `src_buf_type`/`dst_buf_type` (mem/ms, determined by
> instruction source/destination operand); SQE/CLI path same-name tasks lack this information; output invalid value N/A.

> Cross-rank **data transfers** (WRITE/READ roles) retained - they are the main body of network bandwidth usage;
> only cross-rank **synchronization edges** (post-wait, see Sections 6.4/6.6) are ignored.

Key field sources:

- `size`: length register (A5 `lengthXnId` direct read / A6 `xlId` via `GetXnId` - when bit15=1,
  register number offset by loop `xnIdOffset`) -> query register model.
- Peer location: A6 channel via **register value** (`Xn[GetXnId(xcId)]`'s value is channelId) ->
  `channels[(dev,die,channelId)]` -> `{dstDeviceId, dstDieId}`; A5 channel is instruction
  immediate field. A5 MS-class instruction `msId / 0x8000` = rank, `msId % 0x8000` = MS offset (MS page 4KB) - **only
  addressing encoding**: MS is rank's directly accessible local buffer (checker `GenSliceFromMs` +
  `AddLocalBatchReduce` for all msId always local); msId high bits **not used as** local/remote task type
  basis; cross-rank data movement expressed by Rmt* role instructions (via channel). A6 TransMem's
  `msIdMode==1` marks MS end (write source / read destination is local MS,
  `src/dst_buf_type=ms`).
- eid: channel leid/reid (hex string); local eid prefers that mission SQE's `lcEid`
  (v2 added); fallback takes same-die channel leid approximation.
- A6 `TransMem`'s `dmaOpCode` (implemented, checker `TransformTransMemInstr`):
  3=write, 5=write with notify (notify value writes remote Xn/CKE, target type single-rank unresolvable ->
  approximated by remote record), 6=read, 0x1A=remote atomic count (no data plane, approximated by remote record),
  0x19 (compressed write+atomic) and other unsupported (warning skip data plane, setCKE local
  post still executes preserving sync skeleton); `udfEnable` (network reduce) billed as normal UB transfer
  (Section 5.7 approximation; peer reduce not modeled); `udfType != 0` warning skip data plane.

### 6.4 Execution Model: Register + CKE Accounting (register_model.py)

Python simplification of checker `AllRankParamRecorder` (`ccu_all_rank_param_recorder_v3.h:27-110`):

```python
class CcuRegisterModel:
    xn:    dict[(dev, die, id)] -> int      # General registers (length/loop count etc.)
    cke:   dict[(dev, die, ckeId)] -> int   # CKE bitmap (set/clear)
    gsa:   dict[(dev, die, id)] -> int      # A5 GSA registers (A6 has no GSA, only xn/cke)
    seen_post: dict[key, list[PostRecord]]  # CKE post accounting (for pairing)

    key = (waitDeviceId, dieId, ckeId)
    PostRecord = {record_device_id, cke_mask, remaining_mask, task_id, is_local}
```

- LOAD/arithmetic instructions "execute" on this model (read/write registers); uninitialized read -> warning and treat as 0
  (checker errors and exits; here tolerant handling to adapt to incomplete data). A6 arithmetic instructions
  (Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt) execute per 64-bit wrap semantics (Mul takes
  low 32-bit product, shift amount truncated by 0x3F, arithmetic shift per signed interpretation); LoadX/StoreX's
  indirect addressing target register number out-of-range (>16383) warning skip write (checker errors and aborts).
- **A6 in-loop register number offset** (checker `GetXnId`/`UpdateXnId`): when payload register number
  field bit15=1, within expansion region (`curLoopIdx >= expandOffset`) offset register number by
  `xnIdOffset x curExpandCnt`; when bit15=0, direct use original number. CKE/MS id's
  loop offset two generations same rule (`UpdateId`).
- **CKE pairing rules** (consistent with checker `GenWaitNode`, `ccu_task_transform_v3.cc:56-121`):
  ```
  post matches wait <=> post.waitDeviceId == wait.waitDeviceId
                    && post.dieId == wait.dieId
                    && post.ckeId == wait.ckeId
                    && (post.ckeMask & wait.ckeMask) != 0        # bits intersect
  After match: one wait instruction aggregates to at most 1 local wait node (mask = all contributing
          posts' bits OR) + 1 remote wait node (bits depending on remote set)
          post.remaining_mask &= ~wait.ckeMask                  # one post's bits can be consumed by multiple waits in parts
          emptied -> removed from seen_post
  Node dependency: aggregated wait node builds edges with all contributing posts - first contributing post takes
          notify pairing (add_match, Event semantics); rest via extra edges
          (child=wait, parent=post, "wait for all parents to complete" pure timing constraint)
  ```
- **CKE accounting only registers this rank's post** (single-rank graph): cross-rank post generates remote
  `Notify_Record` node (src=local, dst=INVALID -> fixed remote latency) but does not register
  `seen_post` (set action happens at peer rank's CKE); wait bits depending on remote set generate
  remote `Notify_Wait` node (fixed latency, not paired), see Section 6.6.
- When bits not satisfied, that queue pointer suspends (waiting for other queues in this rank's post to unlock) - multi-queue joint advance
  see Section 6.6 algorithm; when same queue has multiple dispatch contexts (same mission multiple dispatches, Section 6.1),
  later context must wait previous context to complete before advancing.
- Wrap-up validation: `seen_post` residual -> warning and count in diagnostic stats (checker reports GRAPH_UNMATCHED, here downgraded).

### 6.5 queue Model and loop Handling

- **queue**: `CcuSqeParam` groups by SQE's owning stream (checker is ccuParams[queueId]);
  each queue one serial chain. New task model `stream_id` is **int**:
  SQE stream directly uses original streamId; **CCU queue chain also belongs to its SQE original stream** - CCU-class SQE
  on whichever stream, the expanded skeleton task is on that stream (`queue` is original stream number,
  `alloc_stream("ccu_d{die}_q{queue}")` resolves back to original stream number via `ccu_queue_origin_stream`,
  same-name reuse to merge same-queue multi-mission); only newly created virtual streams for queue parallelization from base
  `1<<20` sequentially allocated; logical names retained in `stream_labels` for dump/report display.
  Exception: when same original stream is **occupied by another die**'s queue, the latter falls back to base allocation of synthetic stream with warning
  (chain merge would serialize cross-die parallel execution); at this time, in `stream_meta` that stream kind=ccu.
  Within same queue, instructions strictly in order (checker semantics: `ccu_task_transform_v3.cc:189`,
  "single queue strictly advances by instruction order; cross-queue coordination only via generated wait/post edges").
- **Assembly-phase queue parallelization split** (Section 7.4.1): after interpretation completes, each CCU queue chain further split into
  skeleton (wait / stray set / remote sync nodes - stray set refers to record nodes not immediately following data tasks)
  and asynchronous data transfer unit two classes:
  skeleton stays in queue serial chain (= SQE original stream chain, re-assembled back to placeholder location by dispatch order) expressing timing framework;
  async units re-attached to virtual stream
  `ccu_d{die}_q{queue}#v{n}` (sequentially allocated from base `1<<20`), reused across missions,
  not reused within mission, connected with skeleton via fork extra edges - serial chain only constrains skeleton and unit internal;
  eliminating "entire queue task-by-task serial" distortion.
- **loop expansion** (merge preferred, aligned with checker `ccu_loop_merge_v3.cc`):
  - `Loop`/`LoopGroup` instruction iteration count from register, known at interpretation time:
    - A5: LoopGroup parameters in Xn (LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41])
      and Xm (gsaOffset/MSOffset/ckeOffset); Loop iteration count in Xn[xmId] low 13 bits;
    - A6: LoopGroup parameters changed to be carried by **Xn/Xm/Xp three registers** (checker
      `LoopGroupXnA6/XmA6/XpA6`) - Xn: `loopInsCnt:10 | expandOffset:9 |
      expandCnt:9`, Xm: `ckOffset:10 | msOffset:11 | gsaOffset:32`,
      Xp: `xnIdOffset:32` (GetXnId offset, Section 6.4); Loop iteration count in
      `Xn[GetXnId(xmId)]` low 13 bits, GSA step in `Xn[xnId]` low 32 bits, context id
      in `Xn[xpId]` low 9 bits; `mode==1` is **conditional Loop** - only when instruction
      `wishCKEBit == loopCtxId + 1024` (`A6_LOOP_BOARD_ENTRY`) hit to expand; otherwise
      this expansion skipped (checker `ProcessLoopIns`); conditional Loop always takes full expand path
      (checker `CollectLoopExpandA6` rejects merge).
    Each Loop within LoopGroup first tries **merge path** (checker `TryProcessMergedLoopA5/A6`
    -> `LoopIterationMerge` / `LoopExpandMerge` semantics); on validation failure falls back per-iteration full expand.
  - **Merge pre-validation** (any not satisfied -> WARN fallback, same as checker "Skip merged-loop
    optimization" semantics):
    1. Loop body contains only mergeable instructions:
       - A5: simple TRANS / SYNC TRANS (SyncCKE/GSA/Xn) / REDUCE / ClearCKE
         (`CollectLoopCkeOpsA5` rule; SetCKE, LOAD, LOOP, JMP,
         `setRmtCKEMask != 0` TransMem, `clearMask != 0` ClearCKE all rejected);
       - A6 (`CollectLoopCkeOpsA6` + `Collect*LoopInstrA6` rules): local TRANS
         4-code / TransMem (`udfType==0` and `dmaOpCode in {3,6}` - 5 write with notify
         per iteration triggers remote post, rejected; channel must be resolvable, `IsExistRemoteDieInfo`) /
         REDUCE (`udfEnable` TransMem network reduce also folded per trans slot) /
         ClearCKE (`clearMask==0`); SetCKBit, LOAD, LOOP, JMP, Wait all rejected;
    2. wait/set masks **balanced** within loop body (`ConsumeProducedMask`/`ProduceMask`:
       each wait's bits must be produced by earlier set; at sequence end all produced bits exhausted);
    3. Cross-expand CKE/MS resources no conflict (`CheckResourceConflict`: different expands via
       ckOffset/msOffset offset CKE/MS numbers must not reuse; A6's MS conflict check covers
       local transfer / TransMem `msIdMode`'s MS end and reduce msId list);
    4. Transfer/reduce length > 0 (`ValidateLoopLen`) and expand count x iteration count > 0;
    5. **A6 additional**: within loop body wait CKE ID must not reuse (`CheckA6LoopWaitCkeIdUnique`
       hardware constraint, checker errors and aborts, here downgraded fallback full expand); each expand's iteration count /
       length / channel value consistent (checker each expand `loopCnt` equal + `InstrMerge`/
       `MemMerge` merge-ability performance-side approximation - register numbers after xnIdOffset offset may
       point to different registers).
  - **Merge launch** (`EmitMergedLoopInstrsV3` semantics): loop body expanded by sample
    (curExpandCnt=0) executed **once**, slot by slot wait -> data task -> set -> clear wait bits;
    data task size = single length x (expand count x iteration count) (timing-equivalent folding, same as checker
    AddBatchTransMem/AddBatchReduce batch tasks); other expands' CKE register
    bitmap by or/clear idempotence replayed, ensuring post-loop register state consistent with full expand (only
    PostRecord task accounting not duplicated). All tasks within launch window (data tasks and folded
    notify tasks) carry `loop_cnt` (post-expand loop count = N+1 after N expansions) and
    `loop_iter_cnt` (single loop self-loop count); total iteration count = `loop_cnt` x
    `loop_iter_cnt`, retaining pre-merge loop details for future simulation use; straight-line instructions, fallback full expand, and
    SQE/CLI path tasks have both fields 0 (invalid, trace uniform output N/A).
  - **Quantitative effect** (REDUCE_SCATTER empirical db, 128 expands x 50 iterations x 13 instructions):
    task count 192,005 -> 66 (data tasks 10, local post/wait several, remote sync nodes
    35 - remote post 21 + remote wait 14, Section 6.6), graph + simulation end-to-end
    ~55 s -> ~6 ms, makespan 70,113.650 us -> 2,753.711 us (this section merge eliminates
    128K task-level notify overhead and 64K small transfer per-task latency; queue parallelization
    and remote sync node inclusion final value, Section 7.4.1).
  - A6 LoopGroup config validation (`loopInsCnt<=512`, `expandCnt<=511`,
    `expandOffset<=511` and `expandOffset<=loopInsCnt`, checker
    `TransformLoopGroupInstr`) when not satisfied, warning skip entire LoopGroup (defends against abnormal
    register value causing runaway expansion); "within loop waitCKE must not duplicate CKE ID" validation see above
    merge pre-validation #5 (`ccu_task_transform_instruct_a6_v3.cc:1126-1146`).

### 6.5.1 Merged loop Pipeline Concurrency Recovery (loop Tag + Signature Chain Resolution)

**Problem - folded pseudo-serial (merge is not timing-equivalent for notify structure)**: Section 6.5 merge for **data tasks** is
timing-equivalent folding (size x total iteration count, in bandwidth-limited scenario total duration unchanged), but for loop body's CKE
wait/set handshake **is not**. During real microcode execution, LoopGroup's N expansions emitted concurrently, each expansion
internally further advanced by M inner iterations pipelined (typical each iteration 4KB magnitude); loop body's
`TransMemToMs -> setCKE -> waitCKE -> TransLocMsToRmtMem / TransLocMsToLocMem`
is **per-iteration fine-grained handshake**: iteration k+1's HBM->MS prefetch (memcpy engine) and iteration k's
network write + local copy-out (channel / memcpy engine) overlap; steady-state throughput ~= max(each phase total).
After merge folding, per-iteration handshake degrades to **single pair** wait/record node; semantics becomes "first-phase all data
(x N x M iterations) complete -> second-phase can start" **full barrier** - pipeline concurrency folded to serial.

Taking `hccl_vm_install/data/hccl_vm_data.db` first loop template as example (microcode 0-10:
`TransMemToLocMs(4KB, set 0:1) -> waitCKE(0:1) -> 7x TransLocMsToRmtMem(each 4KB,
set 0:1..0x40) -> TransLocMsToLocMem(4KB, set 0x80) -> waitCKE(0:ff)`; LoopGroup
128 expands x 256 inner iterations = 32768 times x 4KB = 128MB/task):

```
Current graph (after 7.4.1 parallelization;  AA = loopgroup predecessor task representative, i.e., queue skeleton in
                            last node before loop launch window):
  AA -> memcpy(mem->ms) -> record(0:1)                     <- prefetch unit (128MB, 95.95us)
  record(0:1) ->[paired]-> wait(0:1) -> 7x ub_write + memcpy(ms->mem)
                                    ^ full serial: second-phase 8 units fork from wait(0:1),
                                      via paired edge wait for prefetch 128MB all complete
  multiple { record(0:xx) / record(0x80) -> wait(0:ff) }       <- loop end barrier (retained)
```

- Real execution: steady state ~= max(prefetch 95.95, 7 port writes 2526.48) us;
- Current graph: serial 95.95 + 2526.48 us.
- **Quantitative** (example db ALLGATHER, before/after this section fix empirical): makespan **2657.52 ->
  2561.54 us**. Eliminates 95.95 us serial prefetch phase; after prefetch and local copy-out concurrent, through HBM dynamic sharing
  each 191.79 us (pipeline steady-state correct expression, Section 7.5); 7 port network writes 2526.48 us become only
  critical path. Error upper bound: when two phases belong to different engines (HBM prefetch vs network write) and durations comparable,
  loop body duration overestimated by nearly **2x** (serial t1+t2 vs steady-state max(t1,t2)); same-engine phases
  (such as pure HBM prefetch+copy-out) serialization error absorbed by HBM dynamic sharing (total unchanged);
  error accumulates linearly with merged loop count.

**Correctness premise - merge validation invariant**: Section 6.5 merge pre-validation #2 (wait/set masks within body
balanced) ensures merged loop body's **all wait bits produced by earlier set in body and exhausted at end** - i.e.,
within merge launch window all CKE dependencies are intra-loop per-iteration pipeline handshake; no cross-loop
real data dependency (otherwise validation fails, fallback full expand). Therefore resolving "in-body intermediate wait and its consumed
set" does not break loop-external synchronization: **loop-end wait** (last wait in launch window, no same-loop data task after it) and **mission completion set**
(skeleton tail + all unit tails, Section 7.4.1) continue to carry post-loop instruction / SQE successor / same-queue next mission's queue-level serial semantics.

#### Solution (loop Tag + Signature Chain Identification + Deletion and Reconnection, Iterate to Convergence)

1. **loop tag**: `_MissionContext` maintains current launch window's `_loop_tag`, named
   `loopgroup_{i}_loop_{j}` (i = LoopGroup instruction execution sequence within dispatch context, 0-based;
   j = within-group cur_loop_idx, 0-based; when loop body embeds LoopGroup, each takes value by launch window,
   does not form hierarchy). Within `_emit_merged_loop` window set bits, `add_task` default inherits to new field
   `Task.loop_tag` (same pattern as `loop_cnt`). **Only merge path tags**; full expand fallback /
   SQE / CLI / straight-line instructions empty string (trace output N/A). Tag also serves resolution range limitation
   (cross-loop / cross-mission no mis-delete) and trace/dump provenance.
2. **Signature chain identification** (recorded during launch + determined before assembly, avoiding graph result reverse lookup): when `_gen_waits`
   pairing succeeds, records (wait_task_id -> all contributing post task_ids) into
   `ctx.wait_posts` (aggregated wait's match + extra all contributors, Section 6.4). Assembly phase
   (`expand_ccu_graph` assembly loop beginning, before skeleton/unit split) scans ctx.nodes sequence;
   signature chain hit requires all of:
   - W is local paired `Notify_Wait` and `loop_tag == T` (within merge window no remote
     wait - validation #2 ensures body wait bits all locally produced);
   - All W's contributing posts R* are `loop_tag == T`'s `Notify_Record`, and each in
     ctx.nodes **immediately follows same-tag data task M\*** (unit-tail record, i.e., the folded producer
     task, such as TransMemToMs folded memcpy; data tasks subset of `_QUEUE_DATA_TYPES`,
     not limited to Ub_Write_Or_Read - Memcpy/Reduce_Inline also have this pseudo-serial);
   - **After W (ctx.nodes order) exists same-tag data task** (mid-body determination; loop-end
     wait has no same-loop data task after, naturally exempt; later loop's data task tag different,
     also not mis-judged);
   - R* all do not immediately follow non-data nodes (stray set / sync slot's local set **does not hit** - its
     wait is for real dependency like remote sync, resolution prohibited).
3. **Deletion and reconnection** (prune ctx.nodes, not post-assembly graph surgery):
   - **W always deleted**: removed from ctx.nodes and dep task table, revokes its pairing edge (`remove_match`)
     and aggregated extra edge (`extra_edges[W]`) - this is the essential operation breaking the barrier;
   - **R\* depends on consumer**: R does not appear in other wait's `wait_posts` values -> deleted together
     (example db's set24 only consumed by wt25, directly deleted); otherwise **retain R only delete
     W** (its retained edges unaffected, loop-end wait waits as usual - timing-equivalent, semantics-lossless
     conservative branch; when post bits partially consumed by multiple waits takes this branch);
   - **Successor unit reconnection automatic**: after pruning, assembly logic re-scans ctx.nodes; original units forked from W
     automatically fork to W's previous skeleton node (i.e., loopgroup predecessor task AA);
     when W is skeleton head (no preceding skeleton), that unit becomes entry unit; entry dependency expressed by SQE attachment edge
     (Section 7.4.1). **No edge-reconnection graph surgery needed** - fork parent / skeleton chain /
     completion set naturally generated by existing assembly rules based on post-pruning sequence.
   - Register / CKE accounting **zero change**: pairing consumption completed during launch phase; pruning is only graph structure
     operation; post-loop instruction's wait match uses unchanged accounting result (deleted post's bits
     already consumed by W, no residual waiter).
4. **Iteration and statistics**: repeat 2-3 until no new match (handles in-body multi-level chains, such as
   prefetch->wait->mid-segment->wait->tail-segment: after first layer deleted, next layer exposed as new chain head);
   `CcuBuildStats` adds `pipeline_chains_broken`; summary line shown as `pb=N`.

Post-fix example task graph (compare with this section beginning; deleted nodes are record(0:1)/wait(0:1); original
"wait(0:1) -> ub_write" gate taken over by AA, timing-equivalent):

```
AA -> wait(0:ff)                                        <- loop-end barrier (skeleton)
AA -> memcpy(mem->ms)                                   <- prefetch unit (record deleted)
AA -> multiple { ub_write_or_read -> record(0:xx) }          <- 7-way write, concurrent with prefetch
AA -> memcpy(ms->mem) -> record(0x80)                   <- copy-out unit, concurrent with prefetch
multiple { record(0:xx) / record(0x80) -> wait(0:ff) }      <- loop-end waits for all data to complete
```

#### Implementation Details (Code Landing Points)

- `simulator/task_model.py`: `Task` adds `loop_tag` slot (default empty string);
  `to_trace_args` outputs `loop_tag` (empty string -> `N/A`, same metric as `loop_cnt`);
- `ccu_graph_builder.py`:
  - `_MissionContext.__init__`: `_loop_tag = ""`, `_lg_seq = 0`, `wait_posts = {}`;
  - `_exec_loop_group`: on entry records `lg_tag = f"loopgroup_{self._lg_seq}_loop_{
    cur_loop_idx}"` (passed through to `_emit_merged_loop`); after LoopGroup processing completes
    `self._lg_seq += 1`;
  - `_emit_merged_loop`: within window sets `self._loop_tag = lg_tag` (`finally` resets,
    same pattern as `_loop_cnt`);
  - `_gen_waits`: after pairing success `self.wait_posts[wait_task.task_id] =
    [p.task_id for p in posts]`;
  - `expand_ccu_graph` assembly loop beginning calls `_break_pipeline_chains(ctx, dep)`
    (returns deleted chain count, accumulated into `stats.pipeline_chains_broken`);
- `graph_dumper.py` / `simulator/report.py`: dump `nodes[]` and trace args
  add `loop_tag`; DOT/HTML legend synchronized (optional, presented along with loop_cnt).

trace presentation (`export_trace`, display layer only, does not affect simulation): virtual secondary stream carrying
`loop_tag` task, its display line re-attached to this stream's loop tag line (line name is tag, such as
`loopgroup_0_loop_0`), indicating it is converted from in-loop instruction; other tasks on same secondary stream
keep original display line - cross-mission reused mixed secondary stream thus split into original line + tag line.
Display lines partitioned by `(stream_id, loop_tag)`; tasks of different virtual streams not merged (retain
inter-secondary-stream task partition); main stream tasks (including loop-end wait) all keep original line; task
`args.stream_id` retains real stream number.


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
                                                                 # stray set/sync
            if no same-tag data task after W:                   continue   # loop-end barrier
            for R in posts:                                        # --- hit ---
                if R in other wait's wait_posts values: continue        # has retained consumer
                dep deletes R; ctx.nodes removes R                       # unit tail now taken by M
            dep.remove_match(W); dep.extra_edges pops W
            dep deletes W; ctx.nodes removes W
            broken += 1; changed = True; break                     # re-scan
    return broken
```

Acyclic guarantee: pruning only deletes nodes and edges, does not add edges; deleted W/R are all launch-window nodes;
fork/join edges recomputed by assembly phase based on post-pruning sequence (Section 7.4.1 acyclic argument continues to hold);
`_drop_cyclic_matches` (Section 5.4) and simulator runtime deadlock detection fallback unchanged.

#### Solution Evaluation (Pros and Cons) and Alternatives

**Advantages**:

1. Does not touch Section 6.5 merge semantics and checker alignment logic - post-hoc resolution, risk isolated;
2. Prunes ctx.nodes before assembly (rather than graph surgery on assembled result): fork parent / skeleton chain /
   completion set naturally generated by existing assembly rules, avoiding "after delete node, re-connect edges" graph surgery correctness burden
   (post-assembly surgery requires manual fork edge / skeleton chain / completion set reconnection, error-prone);
3. During launch records wait->posts pairing; identification does not depend on assembled result's dependency shape - insensitive to Section 7.4.1
   assembly rule evolution (fork parent selection, unit split adjustment);
4. loop tag limits resolution range (same loop same mission), preventing cross-loop / cross-mission
   mis-delete; tag also feeds back trace/dump provenance;
5. Deletes nodes rather than only delete pairing edge: avoids retaining a semantically dangling unpaired local wait (though its
   empirical takes `local_wait_delay` = 30ns, timing no difference, but trace statistical metric clean).

**Disadvantages / Risks**:

1. Signature chain is structural heuristic; identification conditions coupled with launch / assembly logic, needs to evolve with both
   (already compressed coupling surface to ctx.nodes sequence layer by "launch-phase record + pre-assembly prune");
2. `Task` model extends field; trace/dump/JSON presentation needs sync (Chapter 11 metric);
3. "Fully concurrent + bandwidth manager arbitration" is pipeline **steady-state** approximation: ignores pipeline fill/drain
   latency (single iteration 4KB magnitude, sub-microsecond, negligible) and per-iteration handshake **rate** overhead
   (if CKE engine throughput is bottleneck, needs dedicated rate model; current folding / deletion both counted once,
   same metric as pre-fix, registered Section 10.3 #21);
4. Full-expand fallback path (`mf > 0`) does not go through this resolution - its per-iteration wait/record is real 4KB
   granularity handshake, but inter-iteration overlap also serialized by queue chain; that path node count large, current example not
   triggered, as known residual (Section 10.3 #21).

**Alternative comparison**:

| Alternative | Approach | Advantages | Disadvantages | Conclusion |
|------|------|------|------|------|
| B. Launch-phase direct judgment (no tag / no identification) | `_emit_merged_loop` by slot structure directly judges mid-body wait (after it still has data slots) and skips emitting its wait/pairing | Implementation simplest; no Task field extension; judgment info (slot order) on hand, no heuristic | Loses trace handshake node and loop provenance; skipping emit changes register pairing accounting (loop-end wait absorbs prefetch set, timing harmless but behavior difference vs checker widens); stray set / sync scenarios need additional exclusion conditions | Feasible simplified variant; if trace presentation not needed, can choose |
| C. Segmented pipeline modeling | Data tasks folded into K segments (8~16); between segments retain wait/record; explicitly models fill/drain | Includes pipeline fill effect, more refined | Node count x K; K value no physical basis; under steady state differs from full concurrent by only first/last one segment | Over-design, rejected |
| D. Simulator soft dependency edge | New edge type "pipeline edge" (child starts after parent completes X%) | Graph structure unchanged | Simulator semantics complicated; X has no physical basis; and serial chain structure (unit forks from wait) still needs graph-layer modification | Rejected |

Conclusion: Adopt original solution (tag + signature chain + deletion), per above step 2/3 implementation locations and safety
conditions strengthened (launch-phase record + pre-assembly prune + mid-body determination + consumer conservative branch).

**Implementation status (landed)**: `Task.loop_tag` field (task_model / trace args /
dump `nodes[]` / input_builder loopback) + `DependencyGraph.remove_task()`
(together with cleaning extra edges and notify pairing) + `_break_pipeline_chains()` (called at assembly loop
beginning, statistics `pipeline_chains_broken`, summary presented as `pb=N`) + trace
loop secondary stream rename presentation (`export_trace`, see above). Verification:
example ALLGATHER db cross-engine scenario makespan 2657.521 -> 2561.541 us (56 -> 54
tasks, pb=1, 9 units all fork re-attached to previous skeleton node rwt22, item-by-item consistent with hand-crafted prototype);
archive library regression - same-engine phase scenario (mv->mv both HBM) makespan unchanged (zero-error scenario,
theoretical expectation); SQE-only library pb=0 unaffected. `loop_tag` retained as persistent task attribute
for future performance simulation (such as per-loop bandwidth/rate modeling, Section 10.3 #21) consumption.

### 6.6 Graph Algorithm (Pseudocode)

```
expand_ccu(device_id, die_id, sqe_params, instr_space, dep_graph, reg_model):
    SQEs grouped into dispatch contexts by contiguous segment (Section 6.1: non-contiguous = new dispatch, independent window)
    Same-queue contexts serial by dispatch order; each queue takes first context into active set
    while active set not empty:
        progress = False
        for q in active set's each queue's current context:        # cross-queue polling
            while ptr[q] not at window end:
                instr = decode(instr_space[ptr[q]])
                if instr is WAIT and reg_model.cke bits insufficient:
                    break                              # this queue suspends,轮到 other queues
                nodes = transform(instr, reg_model)   # Section 6.3 mapping; LOAD only modifies registers
                for node in nodes: attach to queue q's chain tail
                if instr is SET_CKE: reg_model.seen_post registers
                if instr is WAIT:     pair with seen_post -> aggregated wait node (Section 6.4)
                if instr is LOOP:     merge path (Section 6.5 validation) or repeat-interpret loop body by iteration
                ptr[q] += 1; progress = True
            if ptr[q] at window end: current context completes; same-queue next context enters active set
        if not progress:
            # Stall = exists wait depending on remote CKE (cross-rank, invisible in single-rank graph)
            For stalled queue: generate remote Notify_Wait node (notify_id+mask, fixed latency),
                then replay rest of instruction (local set / data transfer), continue advancing
    SQE stream attachment (Section 7.4.1): placeholder removed from SQE stream chain (task graph does not introduce placeholder nodes);
        entry edge (SQE predecessor -> queue skeleton head) / exit edge (mission completion set -> SQE successor)
        expressed via extra edges
    Assembly split (Section 7.4.1): each mission queue sequence -> skeleton (stays in queue chain) + async units
        (attached to #v virtual stream, stream_id re-points to virtual stream), fork (predecessor skeleton -> unit head) /
        completion set join (skeleton tail + all unit tails -> SQE successor / same-queue next mission entry)
        extra edges connect
```

- **Cross-rank post-wait does not build edges, but retains sync task nodes** (single-rank graph, main difference from checker): checker's `GenerateRawCcuGraphsV3` (`ccu_task_transform_v3.cc:1275-1406`)
  lets multi-rank CCU graphs advance alternately, cross-rank post mutually unlocks; this plugin has only single-rank instruction data:
  - Instructions that post to remote CKE / notify remote register -> generate remote `Notify_Record` node
    (src=local, dst=INVALID -> fixed `remote_record_delay`, does not register local CKE accounting,
    does not participate in pairing):
    - A5: SyncCKE's `rmtCKEId/locCKEMask`, SyncGSA/SyncXn's
      `setRmtCKEId/setRmtCKEMask`, TransLocMSToRmtMS's `setRmtCKE` (consistent with checker
      `AddPost`'s trigger condition);
    - A6: TransMem `dmaOpCode=5`'s write-then-notify and `0x1A` remote atomic count,
      SyncXnWtx's `notifyValid=1` notify, SyncAtx's atomic sync - remote target type
      (Xn/CKE) resolved by peer address mapping (checker `GetXnAndTypeIdByAddr`), single-rank
      unavailable, all approximated as remote record (mask takes notify value low 16 bits for trace display only).
    SyncGSA/SyncXn (A5) and SyncXnWtx main write (A6) value sync is only remote register write
    (checker builds node/writes state on peer graph); local does not generate task separately.
   - wait's required bits cannot be satisfied by this rank's post (stall detection finds) -> that wait depends on remote -> generate
     remote `Notify_Wait` node (fixed `remote_wait_delay`, not paired), and replay rest of instruction;
     unsatisfied waits within loop body also generate remote wait node then continue. Partially satisfied: satisfied
     bits paired with local post normally (aggregated to 1 local wait node, Section 6.4); unsatisfied bits generate one
     remote wait node. A6 independent Wait instruction (Xn value compare) when unsatisfied same mechanism: top-level stall ->
     stall resolution generates remote `Notify_Wait` (task id label `xwt`) then passes through.
    - Task graph does not凭空 generate any node: all nodes from original db data or microcode instruction conversion
      (attachment dependencies expressed via extra edges, see Section 7.4.1).
- Intra-rank post-wait paired normally: multiple queues of same device, multiple dies' CKE
  all in local graph, builds edges per Section 6.4 rules.

### 6.7 Comparison with checker CCU Graph Simplification

**CCU task rank/eid/direction semantics**:

- **Channel-class network tasks** (write/read): local rank takes opDetails.rankId; peer rank via
  channel peer eid (reid) queries eid->rank index (Section 4.8); not found fills INVALID +
  `rank_unresolved` count. Equivalent to checker `GetRemoteRankByChannel`
  (`g_allRankChannelInfo`: channelId -> peer), but uses RankEidMap's communicator rank
  instead of checker's physical dstDeviceId.
- **Reduce tasks** (REDUCE microcode): MS is rank local buffer (checker for all msId always
  `GenSliceFromMs` + `AddLocalBatchReduce`); reduce always local HBM operation; src/dst both fill
  local rank; cross-rank data already billed in preceding Rmt* role transfer; reduce does not repeat network.
- **Local tasks** (local transfer Memcpy, local reduce Reduce_Inline): src/dst both fill local rank.
- **is_write**: CCU-produced UB tasks all explicitly carry (write=True / read=False);
  does not rely on simulator rank heuristic.
- **Local rank self-validation**: if channel leid <-> indexed rank inconsistent with opDetails.rankId,
  takes opDetails as authoritative and warns (eid numbering system offset protection, such as deviceId and rankId having offset).

| checker Stage | light_performance | Reason |
|--------------|-------------------|------|
| Two-phase (raw graph -> queue parallelization -> re-attach) | Single-phase interpret + assembly-phase queue parallelization split (Section 7.4.1: `ApplyCcuQueueParallelizationV3` simplified port, skeleton/unit split + fork/join and attach extra edges + `#v` virtual stream) | No validation; no need to retain transformable raw intermediate form; parallelization completed in one shot at assembly (reuse granularity and other differences see Section 7.4.1) |
| Loop merge (BatchTransMem/BatchReduce, resource conflict check, producedMasks balance) | **Equivalently implemented** (Section 6.5: pre-validation + folded launch + fallback full expand; memory slice merge not needed - simulation only cares about data size and timing) | Aligns checker semantics, eliminates node blowup |
| SQE merged to TaskCcuGraph (`IsContiguousCcuSqe` contiguous to merge + `CcuMissionKey` latest-node-append) | **Equivalently implemented** (Section 6.1: contiguous segment split dispatch contexts; each context independent window replay + sequential args consumption; register/CKE shared across contexts; same-queue contexts serial by dispatch order) | Aligns checker multi-dispatch (window wrap re-send) semantics; once merged to single context lost second dispatch data transfer (Section 10.3 #20 fixed) |
| GetRemoteRankByChannel (channelId -> dstDeviceId/dstDieId, via Communicator table maps rank) | Peer eid (reid) -> RankEidMap/ranktable index -> communicator rank (Sections 4.8/6.7) | estimate mode Communicator table only this rank one row; ccuChannels.rankId field is endpoint deviceId unusable |
| AllRankParamRecorder complete address space mapping (XN/CKE/PFE/CHANNEL/JETTY/MISSION/MS ...) + HBM data content + remote address -> register reverse lookup (`GetXnAndTypeIdByAddr`) | Retains only xn/cke/gsa/channel four classes; A6 Load/Store's HBM data plane and remote target reverse lookup not modeled (Section 6.4, `_MissionContextA6` deviation list) | Simulation only needs length and peer; data content / remote register number does not affect timing |
| A6 checker error-abort scenarios (unsupported dmaOpCode/dstType, clearType!=1, clearMask!=0, LoopGroup config illegal, LoadX/StoreX indirect number out-of-range, etc.) | Downgraded to warning + skip data plane (sync semantics still execute) or skip instruction | Performance tool tolerates incomplete/abnormal data, keeps queue skeleton intact |
| CcuTraceInfo provenance, ccu_convert_config.json generation | Removed | Validation/rewrite tool exclusive |
| CheckAllPostMatch error | Downgraded to warning + diagnostic stats | Adapts to single-rank data |
| Cross-rank post unlocks other's wait (multi-rank joint advance) | **Does not build cross-rank edges**: cross-rank post/wait generates remote sync node (fixed latency); peer when ready approximated by fixed latency | Single-rank graph (core difference) |

### 6.8 A5 vs A6 Instruction Set Differences Overview

| Dimension | A5 (V1/Ascend950) | A6 (V2/Ascend960) |
|------|--------------------|--------------------|
| LOAD class | 7 instructions: includes GSA family | 18 instructions: unified X register + LoadX/StoreX/ClearX/Nop + arithmetic (Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt) |
| CTRL class | 5 instructions: Loop/LoopGroup/SetCKE/ClearCKE/Jmp | 7 instructions: + SetCKBit/Wait/Fence |
| TRANS class | 12 instructions: Mem<->MS all combinations + SyncCKE/GSA/Xn | 7 instructions: local 4 types + unified TransMem (dmaOpCode distinguishes) + SyncWtX/SyncAtX |
| REDUCE class | Add/Max/Min (msId list addressing, <=8 MS) | ReduceAdd/Max/Min (same structure, no embedded waitCKE) |
| Loop register | v1.loop{xnId, startInstrId, endInstrId}, loopCnt in Xn | v2.loop{xpId(context), xnId, xmId(iteration count), mode}, mode==1 conditional expansion (Section 6.5) |
| LoopGroup parameters | Xn: LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41]; Xm: gsaOffset/MSOffset/ckeOffset | Xn: loopInsCnt:10/expandOffset:9/expandCnt:9; Xm: ckOffset:10/msOffset:11/gsaOffset:32; Xp: xnIdOffset:32 |
| Register number addressing | Payload field direct use | `GetXnId`: when bit15=1, offset by loop xnIdOffset (Section 6.4) |
| Embedded sync fields | TRANS/REDUCE generally carry waitCKE + setCKE + clearType | Local TRANS/REDUCE only setCKE (wait taken by independent SetCKBit/ClearCKE/Wait instructions) |
| Channel/length value | Channel is instruction immediate field | Channel/length both Xn register values (`Xn[xcId]`/`Xn[xlId]`) |

Both instruction sets already landed in graph (A6 interpreter `_MissionContextA6` reuses A5's task launch/
CKE pairing/advance/merge-launch infrastructure, only overrides instruction decode and loop organization); A6 instruction ->
task type summary table see Section 11.3.3; established deviations with checker see `_MissionContextA6`
class docstring and Section 6.7 comparison table.

---

## 7. Performance Simulation Module (Sub-Requirement 4, Aligned per HCCLSIM Baseline)

> This chapter is the design description of `simulator/` subpackage (ported from `docs/light_performance/HCCLSIM/hccl_sim/`);
> interface takes HCCLSIM as authoritative baseline. Differences from old version and leftovers in Chapter 10.

### 7.1 Overview

```python
sim = HcclSimulator(config, dep_graph, topo_resolver)
result = sim.run(trace_filepath=None, max_trace_events=0)   # -> dict
```

Simulator receives three parameters:

| Parameter | Type | Content | Builder |
|------|------|------|--------|
| `config` | `SimConfig` | Latency, bandwidth, utilization config | Caller |
| `dep_graph` | `DependencyGraph` | Task nodes + dependency relations | `graph_builder` (db path) / `input_builder` (JSON path) |
| `topo_resolver` | `TopoResolver` | eid -> physical port resolution (ranktable.json) | Caller |

`run()` returns result dict (no longer returns raw makespan):

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

Simulator does not accept task sequence list, does not infer dependencies - all from `dep_graph`.

### 7.2 Parameter 1: `config: SimConfig`

**Unit convention**: internal latency = seconds, bandwidth = B/s; **JSON config units latency = us, bandwidth = GBps**
(`from_dict`/`to_dict` auto-convert). Port identifier is `"die_id/port_number"` string
(such as `"1/0"`), consistent with ranktable.json.

#### Field Definition

| Field | Type | Default | Description |
|------|------|--------|------|
| `local_record_delay` | float | `1e-6` | Local record fixed duration; JSON value unit us |
| `remote_record_delay` | float | `1e-6` | Remote record fixed duration; JSON value unit us |
| `remote_wait_delay` | float | `1e-6` | Remote wait / unmatched local wait fixed duration; JSON value unit us |
| `port_bandwidth` | dict | `{}` | `"die/port"` -> bandwidth (B/s). One eid can correspond to multiple ports, configured per port, total bandwidth summed |
| `default_port_bandwidth` | float | `53.125e9` | Unconfigured port default bandwidth (425 Gbps = 53.125 GBps); JSON value unit GBps |
| `port_type` | str | `"x4"` | Port type (`x2`/`x4`), authoritative source db `Server.port_type` (server/pod topology config `port_type` field written into db via servers_info.json). Configured bandwidth is x4 baseline; when `x2`, non-d2h port bandwidth halved (see below "port type scaling") |
| `hbm_bandwidth` | float | `1.4e12` | HBM bandwidth, used for Reduce_Inline/Memcpy; JSON value unit GBps |
| `bw_profile_set` | BandwidthProfileSet? | `None` | Bandwidth utilization config set; when None, utilization is 100% (including multi-jetty conversion step `jetty_utilization_step`, see below) |

#### Port Bandwidth Config Format

```json
{"port_bandwidth": {"1/0": 53.125, "0/7": 53.125}}
```

#### Port Type Scaling (`port_type`, x2/x4)

Port bandwidth in `sim_config.json` (`<topo_type>_port_bandwidth` and
`default_port_bandwidth`) are all **x4 baseline**. server/pod topology config file's
`port_type` field (`x2`/`x4`, default `x4`, invalid values error) written by
`generate_cluster_topo.sh` into cluster-level `servers_info.json`'s each server entry
(tool-exclusive file, does not enter topo.json/rootinfo/ranktable etc. hccl business native format files),
then by C++ cluster parser (`ParseServersInfoJson`) into `Server.port_type` column
(`sim_models.h`/`db_sim_runner_ops.h`, RunnerDB v4).

At simulation time, `pipeline`/`__main__`/`dump_graph` entry calls
`db_reader.load_port_type(db_path, device_id)` to read port type from `hccl_sim.db`'s `Server`
table (locates target device's server by `Device.id -> server_id -> Server.id`;
when not located, falls back to full-table unique value; when old library has no `port_type` column, defaults `x4`), then via
`SimConfig.with_port_type()` lands in config. `get_port_bandwidth()` multiplies **non-d2h**
ports by `port_type_scale` (`x4=1.0`, `x2=0.5`), i.e., x2 port bandwidth halved;
`d2h_bandwidth` and HBM bandwidth unaffected. Command line `--port-type` can override db value.

#### Bandwidth Utilization Config (`bw_profile_set`)

```
BandwidthProfileSet
+-- profile_low:   BandwidthProfile  <- used when shared port concurrency < 8
+-- profile_high:  BandwidthProfile  <- used when shared port concurrency >= 8
\-- THRESHOLD: 8
```

Each `BandwidthProfile` is data-size (32B~128MB) -> utilization (0~1) mapping table; when querying, floor to
largest tier not greater than that data size (floor match).

JSON format:
```json
{
  "profile_low":  {"32": 0.30, "64": 0.35, "...": 0.90, "134217728": 0.95},
  "profile_high": {"32": 0.20, "64": 0.25, "...": 0.84, "134217728": 0.85},
  "single_port_utilization": 1.0,
  "jetty_utilization_step": 0.02745
}
```

> File actual key names are `multi_port_utilization_low` / `multi_port_utilization_high`
> (`from_dict` also backward-compatible with `profile_low` / `profile_high` old key names).

#### Multi-jetty Conversion (`jetty_utilization_step`)

`multi_port_utilization_low` calibrated per **one channel one jetty** regular config.
When `HCCL_UB_MULTI_CHANNEL_NUM=n`, one channel attaches n jettys; concurrent transfer capacity approximately linear
improvement; lightweight simulation does not build jetty-level model (jetty only used for eid/protocol resolution, see Section 5.5); instead
converts jetty count to `profile_low` table overall lift:

```
util'(size) = min(1.0, util(size) + (n - 1) x jetty_utilization_step)
```

- `n` taken from `opDetails.ubMultiChannelNum` (proxy level0 stub reads env var and logs, Section 4.2);
  per-operator effective; when `n <= 1` (unset/invalid falls back 1) directly uses original table, no lift;
- `jetty_utilization_step` value range `[0.0109, 0.044]`, default takes mid value `0.02745`,
  only configured by `bw_profile.json` (out-of-range prints WARNING then clamps to interval boundary);
- Only acts on `profile_low`: `profile_high` is high-concurrency tier (port already saturated by multi-path transfer; multi-jetty
  no additional gain); single-port/`d2h` takes constant utilization (`single_port_utilization` /
  `d2h_utilization`); none converted;
- Entry: `SimConfig.for_jetty_num(n)` -> `BandwidthProfileSet.for_jetty_num(n)`
  (`bandwidth_profile.py`); pipeline per-operator generates one converted config copy;
  debug can use `--jetty-num` to override db value (`run_perf_sim.py` / `python -m light_performance`).

### 7.3 Parameter 2: `dep_graph: DependencyGraph`

#### Four-Layer Content

```
DependencyGraph
+-- tasks: dict[str, Task]                       <- task nodes (with all parameters)
+-- stream_chains: dict[int, list[str]]          <- serial dependencies (int stream id)
+-- notify_matches: dict[str, str]               <- notify dependencies (wait_task_id -> record_task_id)
\-- extra_edges: dict[str, list[str]]            <- additional dependency edges (child_task_id -> [parent_task_id, ...])
                                                   <- sources: CCU queue parallelization fork/join edges and SQE attach edges (Section 7.4.1)
```

#### First Layer: Task Nodes (`Task`)

Each Task carries all parameters needed for execution (invalid values uniformly use `INVALID = 0xFFFFFFFF`,
eid empty string, string attributes `""`):

| Field | Type | Applicable Task Types | Simulator Purpose |
|------|------|------------|-----------|
| `task_id` | str | All | Dependency graph index |
| `task_type` | TaskType | All | Determines latency model (internal classification, Section 7.3) |
| `task_name` | str | For display | trace/dump display name: profInfo.profTaskType's op name (`_add_task` uniform fill, Section 11.4.1); when empty, `display_name` falls back `task_type.value` (CCU subtask/CLI input) |
| `stream_id` | **int** | All | Serial chain membership |
| `notify_id` | int? | Notify_Record/Wait/Write_With_Notify | Dependency pairing identifier (CCU path = cke id); Write_With_Notify display only (its notify faces peer, does not participate in pairing) |
| `mask` | int | Notify_Record/Wait | notify bit mask (CCU path): notify_id + mask uniquely identifies one notify action; wait node carries aggregated mask (one wait instruction all contributing posts' consumption bits OR, Section 6.4); SQE path with 0 = unfilled |
| `src_rank` / `dst_rank` | int? | notify/transfer | Local determination (src==dst); read direction local determination |
| `size` | int | Memcpy/Reduce_Inline/Ub_*/Write_With_Notify | Computes latency + looks up utilization |
| `port` | str/list? | (reserved field) | Port resolved real-time by eid; db path not filled |
| `src_eid` / `dst_eid` | str | Ub_*/Write_With_Notify | eid -> ranktable -> physical port |
| `transport_type` / `link_type` | str | For display | msprof reconciliation dimension (profInfo transportType/linkType name, Sections 11.4.2/11.4.3, `_add_task` uniform fill; CCU subtasks not filled) |
| `data_type` | str | For display | trace/report label |
| `protocol` | str | For display | Channel CommProtocol name (Section 11.4.2 note; when no channel info, empty) |
| `jetty_id` | int? | Ub_*/Write_With_Notify | trace label |
| `is_write` | bool? | Network tasks | Billing direction and port query eid selection (Section 10.3 #1) |
| `start_time` / `end_time` | float | All | Simulator fills (-1 = not executed) |

7 TaskTypes:

| TaskType | Latency Model | Description |
|----------|---------|------|
| `NOTIFY_RECORD` | Fixed latency (local also triggers Event) | Local = `is_local_notify` (src==dst); unpaired local record only fixed latency |
| `NOTIFY_WAIT` | Wait Event / fixed latency | matched -> wait for record's Event; unmatched/remote -> fixed `remote_wait_delay` |
| `REDUCE_INLINE` | HBM dynamic sharing | `size / hbm_bandwidth`, shares bandwidth with other HBM tasks |
| `MEMCPY` | HBM dynamic sharing | Same as above (HBM copy, not network transfer) |
| `UB_INLINE_WRITE` | Fixed latency (same as Notify_Record: local/remote record_delay) | notify inline write; SQE graph path does not produce; only old input path (task.txt) uses |
| `UB_WRITE_OR_READ` | Port bandwidth sharing (write/read two directions billed independently) | Network data send/receive; `is_write` explicitly carried by graph (SQE network tasks two directions uniformly mapped to this type) |
| `WRITE_WITH_NOTIFY` | Port write direction sharing + post-write notify fixed latency | Network write + attached notify single task, one-to-one with db `WRITE_WITH_NOTIFY` (7); notify per remote record fixed latency (peer wait not in this db) |

#### Second Layer: Serial Dependencies (`stream_chains`)

```
int stream id -> [task_id_0, task_id_1, ..., task_id_n]
```

- List order = execution order: task[i] must complete before executing task[i+1]
- Each stream_id starts one simpy process, executes serially by chain order with `yield`
- Different streams naturally parallel
- db path stream id addressing: SQE stream directly uses original streamId; **CCU queue skeleton chain belongs to its SQE original stream**
  (Section 6.5, whichever stream CCU-class SQE is on, expanded task is on that stream; only when original stream is occupied by another die's
  queue, then allocate synthetic stream); virtual streams from base `1<<20`
  allocated (Section 6.5, virtual stream logical name with `#v{n}` suffix); logical names retained via `stream_labels`;
  `GraphBuilder.stream_meta` on top gives real/ccu/virtual three-class classification and virtual stream's
  parent_sid (Section 7.4.1, points to its owning queue's stream id, normally SQE original stream), for report/trace/dump labeling

#### Third Layer: Notify Dependencies (`notify_matches`)

```
wait_task_id -> record_task_id
```

Explicit 1:1 pairing: which wait waits for which record. When same notify_id has multiple records, only one
pairs with wait; unpaired records take fixed latency.

#### Fourth Layer: Additional Dependency Edges (`extra_edges`)

```
child_task_id -> [parent_task_id, ...]
```

- Semantics: before child starts, must wait for **all** parents to complete (pure timing constraint, does not carry
  record/wait event semantics - event pairing still expressed by `notify_matches`).
- Current sources: CCU queue parallelization's fork edges (predecessor skeleton -> unit head), mission completion set
  join edges (skeleton tail + unit tail -> SQE successor / same-queue next mission entry), CCU subgraph
  attach edges on SQE stream (entry/exit, see Section 7.4.1), and aggregated wait node's multi-parent edges
  (wait -> all contributing posts, Section 6.4 - first contributing post takes `notify_matches`, rest take this layer).
- Writer side `add_edge` auto-deduplicates (same child-parent pair only recorded once); simulator via
  `get_extra_parents` / `extra_parent_ids` queries.
- Acyclic: above edges always along original instruction order "forward" (skeleton before unit, unit before join point,
  attach at placeholder location before/after, post registered before consuming wait instruction); graph direction ensures acyclic,
  and counted into `_drop_cyclic_matches`'s cycle detection fallback (Section 5.4).

### 7.4 Simulator Consumption Flow

```python
def run(self, trace_filepath=None, max_trace_events=0):
    # 1. Take all task nodes
    tasks = list(dep_graph.tasks.values())

    # 2. Create sync primitives: only matched wait creates Event
    for wait_task_id in dep_graph.matched_wait_ids:
        notify_events[wait_task_id] = env.event()
    for parent_id in dep_graph.extra_parent_ids:  # extra edges' all parent tasks
        task_done_events[parent_id] = env.event() # must pre-build before stream process starts:
                                                  # if parent completes first, child later builds Event will wait forever (Section 7.4.1)

    # 3. HBM bandwidth manager (Reduce_Inline + Memcpy shared, utilization always 1.0)
    hbm_mgr = _BandwidthManager(env, config.hbm_bandwidth, empty utilization table)

    # 4. Each stream starts one simpy process, executes serially by chain order
    for task in chain:
        for pid in dep_graph.get_extra_parents(task.task_id):
            yield task_done_events[pid]           # extra edges (Section 7.4.1), start_time does not include wait
        if NOTIFY_RECORD:
            yield timeout(calc_notify_record_delay(task, config))
            if local and matched: trigger matched wait's Event
        elif NOTIFY_WAIT:
            if local and matched: yield notify_events[task_id]   # wait Event
            else:              yield timeout(remote_wait_delay)
        elif REDUCE_INLINE / MEMCPY:
            yield hbm_mgr.send(task_id, size)                 # HBM dynamic sharing
        elif UB_INLINE_WRITE / UB_WRITE_OR_READ / WRITE_WITH_NOTIFY:
            is_write = task.is_write  # explicit direction; old input path None falls back heuristic
            eid = src_eid if is_write else dst_eid
            ports = topo_resolver.get_ports_by_eid(eid)       # missing throws error
            port = lookup/create NetworkPort(ports, total bandwidth)
            yield port.send(task_id, size, is_write)          # port bandwidth sharing
            if WRITE_WITH_NOTIFY:
                yield timeout(calc_write_with_notify_delay(task, config))
                                                           # notify after write lands:
                                                           # fixed record latency
                                                           # (peer wait not in this db)
        task.end_time = env.now
        if task_id in task_done_events: task_done_events[task_id].succeed(env.now)
    # 5. makespan = max(end_time); assemble result dict (including trace export)
```

Latency formulas unified and maintained single-point in `task_timing.py`.

#### 7.4.1 CCU Queue Parallelization and Virtual Stream (extra edges)

**Problem** - The CCU queue chains produced by Sections 6.5/6.6 are task-by-task serial flows: all
data transfers, wait, set in same queue execute in order. On real hardware, queue's async transfer units execute concurrently;
serialization causes three distortions:

| Distortion | Mechanism |
|------|------|
| makespan overestimated | Async transfers separated by skeleton (wait/set) one by one, cannot overlap |
| Utilization tier distortion | Same queue's concurrent transfers degrade to one-by-one execution; `_BandwidthManager` observed port concurrency always 1; always queries profile_low tier (Section 7.5) |
| Port peak concurrency stats distortion | `peak_concurrency` always 1, cannot reflect concurrent shared bandwidth behavior (Section 8.2 port-level metrics) |

**checker reference** - `ApplyCcuQueueParallelizationV3` (`ccu_task_transform_v3.cc:1052`):
after raw graph, second-pass transformation; `CollectAsyncBoundRecordNodesV3` collects async transfer nodes;
`QueueIdReuseAllocatorV3` by `[startNodeId, afterNode]` interval reuses virtual queue id;
fork/join edges re-attached to skeleton.

**Simplified port** (`ccu_graph_builder.py` assembly loop, assembly phase completes in one shot, no raw
intermediate form) - splits each mission's queue node sequence:

1. **Unit identification**: `_QUEUE_DATA_TYPES` (`Memcpy`/`Ub_Inline_Write`/
   `Ub_Write_Or_Read`/`Reduce_Inline`) data tasks + immediately following contiguous
   `NOTIFY_RECORD` chain = one **async unit**; other nodes (wait / stray set /
   remote sync nodes) are **skeleton**, stay in queue serial chain - queue chain is skeleton itself,
   no longer introduces any head/tail placeholder nodes.
2. **fork edge**: `add_edge(unit head, previous skeleton node)` - before unit starts, must wait for queue to advance
   to its predecessor skeleton (adjacent units share same fork parent); units appearing before skeleton
   (mission first node is data task) has no skeleton predecessor; its entry dependency instead expressed by attach edge.
3. **Completion set join**: mission completion set = skeleton tail + all unit tails, via extra edges joins to
   - same-queue next mission's entry (skeleton head + pre-skeleton unit head, queue serial semantics;
   skeleton tail-to-tail order already covered by queue chain serial) and SQE stream successor tasks (exit edge,
   graph_builder attach phase adds).
4. **Virtual stream re-attach**: per queue one counter/reuse pool set (`vstream_seq`/
   `vstream_free`/`vstream_busy`); unit task's `stream_id` re-points to
   `ccu_d{die}_q{queue}#v{n}` (same `1<<20` base address space); unit chain registered into
   `result.virtual_chains` and `add_stream_chain` - serial chain only constrains skeleton and
   unit internal task order.
5. **SQE stream attach** (`graph_builder._expand_ccu` consumes `result.attach`):
   CCU placeholder removed from SQE stream chain (task graph does not produce any placeholder/fictitious node);
   entry edge = SQE predecessor task (including adjacent previous CCU mission completion set) -> skeleton head and
   pre-skeleton unit head; exit edge = mission completion set -> SQE successor task. All via extra edges
   expressed; timing semantics: SQE predecessor (and previous mission completion set) completes before subgraph can start,
   subgraph mission completion set all complete before SQE successor task can start.
   All-placeholder tasks' SQE stream removes empty chain.

**CKE timing preservation**: split does not touch `add_match` pairing - whichever unit record falls into, wait
stays in skeleton both do not affect event pairing (wait via notify_events waits across streams). In empirical example, tail memcpy
unit's fork parent wt28 happens to be on previous read unit tail set's accounting wait;
cross-stream event + fork edge together guarantee "read complete -> tail copy" original timing. **Exception**: merged loop
launch window's mid-body wait/record pairing is folded pseudo-serial barrier (per-iteration pipeline handshake
folded into full barrier); resolved by Section 6.5.1 signature chain before assembly pruned - this section split based on post-pruning
sequence; skeleton/unit/fork structure unaffected.

**Reuse rule differences vs checker**:

| Dimension | checker (V3) | light_performance |
|------|--------------|-------------------|
| Reuse granularity | afterNode interval (join successor node) level | mission level (when entering new mission, all previous released) |
| Intra-mission reuse | Allowed (DAG explicit edges, no pseudo-dependency) | Not reused (same-mission units concurrent with each other; under chain model, reuse same stream would introduce pseudo-serial) |
| Residual strictness | GRAPH_UNMATCHED error exit | Tolerant (performance tool, Section 6.4 downgraded warning) |

Cross-mission reuse safety: units reusing same virtual stream serial within that stream chain (later mission's
unit chain after previous mission unit chain); time axis no overlap.

**Acyclic guarantee**: fork (skeleton -> unit head) and completion set join edge (unit tail -> SQE successor /
next mission entry) always along original instruction order "forward"; split does not change original order;
`_drop_cyclic_matches`'s cycle detection also counts extra edges (Section 5.4) as fallback; simulator
also has runtime deadlock detection (when task `end_time<0`, errors and exits).

**Statistics and marking**: `CcuBuildStats` carries parallelization (`parallel_units`/`virtual_streams`/
`fork_edges`/`join_edges` - join_edges now stats completion-set join and SQE attach edges), remote sync
(`local_posts`/`remote_posts`/`wait_matched`/`remote_waits`) etc. counts;
summary line shown as `pu/vs/lp/rp/wm/rw`;
`GraphBuilder.stream_meta` classifies by logical stream name - `s{n}` -> real, contains `#` ->
virtual (with parent_sid), rest -> ccu, for three consumptions: report (`Stream count:
N (including virtual M)`, virtual stream lines marked `vir_stream {n} (label)`, n by allocation order from 0
numbered, over 64 streams folds details), trace (virtual stream thread_name `vir_stream
{n} ({label})` + task args `virtual_stream: true`), dump (streams[] with
kind/virtual/parent_sid,
extra_edges list; DOT purple dashed cluster + blue dotted extra edges, HTML badge
+ dotted arc). When stream_meta default (simulator standalone CLI old input), not marked; backward compatible.

**Quantitative effect** (example db REDUCE_SCATTER iter0, 66 tasks): queue parallelization makes makespan
20,404.20 -> 5,149.36 us (~3.96x); subsequent two corrections further reduced to **2,753.71 us**:
(1) remote MS reduce correction (Section 6.3.2, reduce always local HBM) - original second phase 128MB reduce billed as
network read ~2526 us, corrected to HBM ~96 us; (2) cross-rank sync node completion (Section 6.6) -
14 waits depending on remote set previously skipped (read unit launched early at t~=0), corrected to fixed
remote latency simulation. 10 async units re-attached to 10 virtual streams (fork edges 10, attach/join edges by
SQE neighbors and queue serial conditions generated) - first phase 8 units concurrent (7 network read units each
~2526 us, belong to 7 independent ports `1/0`..`1/6` + 1 local memcpy ~96 us), then
reduce ~96 us and tail memcpy ~96 us serial connect. Note single-port read peak concurrency still
1 - benefit comes from **cross-port overlap**, not single-port concurrency improvement.

### 7.5 Bandwidth Sharing Mechanism (`_BandwidthManager`)

#### General Manager (Port and HBM Reuse Same Model)

- **Single manager process model**: when new transfer joins, `interrupt` manager process; manager by current concurrency N
  equally divides bandwidth: `bw_per = total bandwidth / N`.
- **Utilization query basis is `min_remaining`**: same group concurrent tasks by "minimum remaining data size" queries
  utilization table (concurrency <8 uses profile_low, >=8 uses profile_high); within group same basis.
- **profile_low table already converted by jetty count**: before lookup, pipeline already used
  `SimConfig.for_jetty_num(opDetails.ubMultiChannelNum)` to generate config copy;
  multi-jetty (`HCCL_UB_MULTI_CHANNEL_NUM>1`) table values overall lifted; manager itself does not perceive
  jetty (Section 7.2 "Multi-jetty conversion").
- Effective bandwidth = `(total bandwidth / N) x utilization(min_remaining, N)`; computes earliest completion time,
  timeout settles progress; on join/completion recomputes.
- **HBM dynamic sharing**: `Reduce_Inline` and `Memcpy` share one HBM manager
  (utilization always 1.0) - when multiple HBM tasks concurrent, equally shares HBM bandwidth (different from old version's per-task independent
  `size / hbm_bandwidth`, see Section 10.3).

#### Two-Direction Independence

Each `NetworkPort` contains write and read two independent managers, bandwidth independent, no interference:
when configured port bandwidth B, write direction available B; simultaneously read direction also available B.

#### Multi-Port Aggregation

When one eid corresponds to multiple physical ports, `port_key = tuple(port_ids)` as logical port;
total bandwidth = sum of each port's bandwidth (`get_ports_total_bandwidth`).

### 7.6 Parameter 3: `topo_resolver: TopoResolver`

- Input `ranktable.json` (`TopoResolver(ranktable_path=...)`); provides
  **eid (16B hex string) -> physical port list (`["die/port", ...]`)** resolution.
- Transfer task routing: write direction queries `task.src_eid`; read direction queries `task.dst_eid`.
- **When eid cannot find port, directly throws error** (no jettyId fallback, no default port) - therefore db path
  when "network tasks exist but no ranktable" overall skips simulation (Section 3.3); db graph counts tasks with unresolvable
  eid as warning (Section 5.5).
- Port bandwidth by resolved each port one by one queries `SimConfig.port_bandwidth` and sums.

### 7.7 Build Method and CLI

All input methods uniformly build `(config, dep_graph)`:

| Input Source | Build Function | Returns | Status |
|--------|---------|------|------|
| **hccl-vm db (new this iteration)** | `graph_builder.build(group)` | `(config, dep_graph)` | Chapters 5, 6; main input |
| JSON task file | `load_dep_graph(filepath)` + `load_config(...)` | `(config, dep_graph)` | Retained (hccl_sim compatible) |
| task.txt / algorithm generation etc. old input sources | - | - | **Removed along with HCCLSIM version** (see Section 10.3) |

Build flow unified as:
1. Create `Task` node, call `dep_graph.add_task(task)`
2. By intra-stream task order call `dep_graph.add_stream_chain(stream_id, task_ids)`
3. By notify pairing result call `dep_graph.add_match(wait_task_id, record_task_id)`

CLI entry:

```bash
# db input (light_performance main entry, details see Section 3.3)
python3 -m light_performance --db data/hccl_vm_data.db \
    --config sim_config.json --bw-profile bw_profile.json --ranktable ranktable.json -v

# One-click task graph dump (repository root)
python3 dump_graph.py data/hccl_vm_data.db --html

# hccl_sim compatible entry (JSON task file)
python -m simulator --tasks tasks.json --config sim_config.json \
    --bw-profile bw_profile.json --ranktable ranktable.json -v --trace trace.json
```

`--config` overrides each mode's default latency/bandwidth (units us/GBps); `--bw-profile` independently loads
utilization config; report and trace generated by `report.py`'s `generate_report` / `export_trace`
from `run()` result dict.

trace file is Chrome Trace Event format: `ts`/`dur` values unit **us** (format spec convention);
file top-level carries `"displayTimeUnit": "ns"` specifying viewer (chrome://tracing etc.)'s
Start/Wall Duration **display** unit as ns (default ms).

`export_trace` display adaptation conventions:

- **tid sequential numbering**: thread rows by stream id ascending mapped to tid `0,1,2,...` (original
  streamId 2/6..13 direct as tid, string-sorting viewers would sort 10..13 before 2);
  real stream id retained in thread_name label and `args.stream_id`.
- **Endpoint rounding**: `ts`/`end` each rounded then subtract to get `dur` - `ts` and `dur` each independently rounded
  would produce 0.001us pseudo-overlap; same-tid overlap slices viewer would nest as sub-rows (wrap display).
- **Virtual stream label** (Section 7.4.1): db path via `export_trace(..., stream_meta=...)`
  passes stream meta info; virtual stream thread_name changed to `vir_stream {n} ({label})` (n by
  sid ascending i.e. allocation order from 0,摆脱 2^20 base large stream numbers; real stream id see
  in-line task `args.stream_id`); its task args appends `virtual_stream: true`;
  real/ccu stream names unchanged (ccu queue chain logical name only appears in text report line). When stream_meta
  default (simulator standalone CLI old input), not labeled; backward compatible.
- **notify task args carry mask**: CCU path's `Notify_Record`/`Notify_Wait`
  task args appends `mask: "0x..."` (notify_id + mask uniquely identifies one notify action,
  Section 7.3); SQE path with mask=0 does not output this field.
- **Operator summary info first line (op_summary)**: pipeline per-operator calls
  `export_trace(..., op_summary=...)`; each operator (when multi-operator merged export, per
  pid process group) with tid 1 + `thread_sort_index` 0 reserves `op_summary` first line; task line
  tid and sort_index extended (presentation layer only; tid avoids 0 - some viewers like Perfetto treat
  tid 0 as idle thread; `thread_sort_index` is Chrome Trace standard line order control);
  summary event placed at event array front; triple guarantee op_summary always first line. First line places one
  `X` summary slice spanning operator start-to-end (start time and duration are slice's own ts/dur, taking
  task earliest start to latest end); line label   `"<op_name> | <algorithm_name>"`;
  args carries: `algorithm_name` (algorithm selection result, opDetails.algorithmName log
  interception writeback (DFX main path, Section 4.2 note)), `data_type`/`data_count` (element count, V2 layout operators as
  send/receive two-side element count and type; rest as count and single-side type), `task_count` (post-graph
  total task count, CCU mode is microcode instruction translated to task graph count), and op_type/
  comm_id/op_iter/rank/rank_size/input_bytes/output_bytes (opMemInfo's
  inputSize/outputSize, at collection time computed by business API parameters) context; summary event not counted into
  `max_events`.

### 7.8 Alignment Completion Status

Old version (this repository pre-modification) differences with HCCLSIM baseline digested in this alignment:

| # | Item | Handling |
|---|----|------|
| 1 | Task model 6 types -> new 6 types (local/remote record/wait merged to Notify_*, LOCAL_COPY split Memcpy/Reduce_Inline, network transfer changed to Ub_*; later per db added WRITE_WITH_NOTIFY expanded to 7 types, Section 10.3 #19) | `task_model.py` per HCCLSIM replaced |
| 2 | stream_id str -> **int**; notify_id str -> **int** (INVALID=0xFFFFFFFF) | `dependency_graph.py`/graph side sync |
| 3 | Port specified by `port` int field -> **eid real-time resolution** (write queries src_eid / read queries dst_eid) | `topo_resolver.py` + graph side fills eid |
| 4 | Units: JSON latency s -> **us**, bandwidth B/s -> **GBps**, port key int -> **"die/port"** | `config.py` from_dict converts |
| 5 | HBM latency `size/bw` independent -> **dynamic sharing** (Reduce_Inline+Memcpy share manager) | `_BandwidthManager` reused |
| 6 | `run()` returns makespan -> **result dict**; simulator two-params -> **three-params** (+topo_resolver) | `simulator.py` |
| 7 | Latency formulas scattered -> `task_timing.py` single-point | Incorporated |

Leftover unmatched items (including `src_rank==0` hard-code, eid no fallback, broadcast pairing only triggers first wait etc.)
uniformly moved to Chapter 10 tracking.

---

## 8. Performance Profiling Output (Sub-Requirement 5, Overview)

### 8.1 Output Location and Files

```
<installRoot>/plugin/light_performance/data/<timestamp>/
+-- perf_profiling.json      # Main artifact: performance metrics + task graph summary
+-- trace_<commId>_<opIter>.json   # (optional) Chrome tracing, chrome://tracing visualization
\-- light_performance.log    # Runtime log
```

- Directory naming: `hccl-vm plugin run @estimate` one-shot execution channel does not pass data_id (Section 3.2);
  uses runtime timestamp; standalone debug mode (run_perf_sim.py) per `-o` specifies output path.
- Output fixed in plugin data directory not `data/`: host every start clears `data/` (`src/main.cc:32`);
  artifacts under plugin directory persist across sessions; `archive.sh` archive only covers logs/ and data/; plugin artifacts
  not participating in auto-archive (Section 10.3 #15).

### 8.2 Metric Definition (Draft)

| Level | Metric | Source |
|------|------|------|
| Operator-level | opType/opIter/rankSize/dataSize, makespan, software simulation duration | opDetails + `sim.run()` result dict |
| Operator-level | Operator summary info first line (algorithm_name/data_type/data_count, start/end and duration carried by slice ts/dur, Section 7.5 op_summary) | opDetails.algorithmName (DFX/log dual-path interception) + simulation task timeline |
| Operator-level | Total transfer size, effective bandwidth utilization (total transfer / (makespan x port bandwidth)) | Aggregates result["tasks"] |
| Stream-level | Each stream start/end time, duration, task count + virtual stream label (Section 7.4.1) | result["tasks_by_stream"] + stream_meta |
| Graph-level | Async unit count, virtual stream count, fork edge count, completion-set/attach edge count (Section 7.4.1); local/remote post count (lp/rp), local wait paired count (wm), remote wait count (rw) | CcuBuildStats (parallel_units / virtual_streams / fork_edges / join_edges / local_posts / remote_posts / wait_matched / remote_waits, summary line `lp/rp/wm/rw`) |
| Port-level | Each port peak concurrency, write/read transfer size and average bandwidth | result["ports"] (NetworkPort) |
| Task-level | Per-task start/end/duration, transfer task average bandwidth | result["tasks"] (existing trace logic) |

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

> `cross_rank_notify` is retained cross-rank notify node count (diagnostic info): these nodes graphed normally by stream order,
> take fixed remote latency, but do not participate in local pairing (no cross-rank record-wait edges); this count for troubleshooting
> data/classification anomalies (such as local notify mis-classified as cross-rank). `kind`/`virtual_stream_count`/
> `extra_edge_count` are CCU queue parallelization (Section 7.4.1) new fields: stream classification real/ccu/virtual,
> virtual stream count and extra edge (fork/join/attach) count.

---

## 9. Appendix

### 9.1 checker Graph Module -> light_performance Graph Module Full Comparison

| Aspect | checker (C++) | light_performance (Python) |
|------|---------------|---------------------------|
| Goal | Correctness validation DAG (mainStart reachable, supports various checks) | Simulation DependencyGraph (three layers: tasks / stream_chains / notify_matches) |
| Input | db (via StorageManager/sim::Query) | Same db (sqlite3 direct read + struct decode) |
| Graph scope | **Multi-rank joint graph** (all rank tasks enter same DAG) | **Single-rank graph** (cross-rank does not build pairing edges, remote sync tasks retained) |
| SQE translation | TaskMetaTranslatorV3 (including MemSlice address resolution) | Section 5.2 mapping table (only takes len/jettyId/notify fields) |
| WRITE_WITH_NOTIFY consumption | Decode boundary split: `DecodeAndExpandTaskMeta` expands to [write, NOTIFY_RECORD] two subtasks | **Does not split**: single-task mapping `Write_With_Notify` (Section 5.2), one-to-one with db task, trace/dump/stats granularity consistent |
| Dependency edges | stream chain + local/cross-rank notify edges (queue simulation) | stream chain + only local notify first-come-first-served (two-pass scan) + extra edges (CCU fork/join/attach, Section 7.4.1) |
| CCU | Two-phase expansion + queue parallelization + loop merge + CKE accounting | Single-phase interpret + assembly-phase queue parallelization (Section 7.4.1) + loop merge (Section 6.5) + CKE accounting (Chapter 6) |
| Post-validation | Memory conflict / semantic / sync conflict / reachability / single-task check | None |
| Output | Validation conclusion + dump | (SimConfig, DependencyGraph) -> makespan -> profiling json |

### 9.2 Terminology

| Term | Description |
|------|------|
| SQE | Submit Queue Element, AICPU/SDMA-side submitted task unit, i.e., opTask table row |
| WQE | Work Queue Element (UB/DB mode), not involved this iteration |
| notify | AICPU-side event sync primitive, record sets / wait waits, paired by (recordDev, waitDev, notifyId) |
| CKE | CCU-side event sync register (bitmap), post sets / wait waits, paired by (waitDev, dieId, ckeId, mask) |
| jetty | Network card (communication engine) port number, determines memcpy's bandwidth ownership |
| die | Chip die, CCU microcode space and CKE both organized by (device, die) |
| mission | CCU one task delivery (instStartId/instCnt defines instruction window, args[13] passes params) |
| makespan | Task graph end-to-end duration (latest task end time) |
| Main stream / secondary stream | opDetails.streamId marked operator main stream; rest are secondary (head WAIT tail RECORD structure) |

---

## 10. HCCLSIM Alignment Leftovers and To-Match List

> This chapter records **field gaps,
> task type mismatches, and other conflicts** confirmed during this iteration's "db graph path + design document" alignment with HCCLSIM baseline.
> Current implementations all landed per "default fill invalid value / approximate mapping";
> retained here for future joint digestion with collection side and simulation module.

### 10.1 Task Field Availability Table (db Graph Path)

New model Task fields' source and default value in db graph path (invalid value convention: int fields `INVALID=0xFFFFFFFF`,
eid empty string, other strings `""`):

| Task Field | db Source | Availability | Default/Approximate | Future Match Path |
|-----------|--------|--------|------------|-------------|
| `task_id` / `stream_id` | Global sequence / streamId + synthetic base | Available | - | - |
| `notify_id` | opTaskMeta `NotifyTask.notifyId` | Available | - | - |
| `size` | MEM_CPY `len` / REDUCE `dataCount` / CCU length register | Available | - | - |
| `jetty_id` | SQE table column / ccuChannels.jettyId array | Available | INVALID (non-network task) | - |
| `link_type` | opTaskMeta `HcclTaskProfInfo.linkType` (v3, enum name mapping Section 11.4.2) | Available (v3) | "" (v2 old BLOB has no profInfo, shows `Unknown`) | - (Section 11.4 already consumed) |
| `protocol` | ccuChannels.protocol (CommProtocol enum name mapping) | Available | "" (when no channel info, trace shows `N/A`) | - (migrated from `link_type`, Section 11.1 solution a) |
| `src_eid` (local, write) | SQE `lcEid` (v2 collection added, Section 4.4.1) preferred; fallback jettyId -> local device outbound channel `leid` derived | Available (v2)/approximate (fallback) | empty -> any local channel leid fallback | - (v2 already supplies explicit field) |
| `dst_eid` (peer) | Channel `reid` preferred; falls back to SQE `rmEid` | Approximate | "" (tasks without channel info) | Collection side to supply peer explicit eid |
| `src_rank` / `dst_rank` | taskData endpoint rankId (deviceId field carries, Section 4.4.1 rank field rollback) directly consumed: write=src local/dst peer, read=src peer/dst local; CCU/unresolved via eid->rank index fallback (Section 4.8) | Available/index fallback | When both explicit value and index unresolved, INVALID + `rank_unresolved` count | - (Sections 5.2/5.5/6.7) |
| `transport_type` | opTaskMeta `HcclTaskProfInfo.transportType` (v3, enum name mapping Section 11.4.3) | Available (v3) | "" (v2 old BLOB has no profInfo, shows `Unknown`) | - (Section 11.4 already consumed) |
| `data_type` | REDUCE task-level `dataType` (v2 added, enum name mapping int8..bfp16); MEM_CPY task-level no source | Partial available (v2) | REDUCE fills enum name; MEM_CPY "" | MEM_CPY collection side to supply |
| `port` | (reserved field) | Not filled | None | Simulator real-time resolves port by eid (Section 7.6) |

**Instruction-level gaps (CCU microcode expansion)**: channel-class network task's peer rank already resolved by eid->rank index
(Sections 4.8/6.7); reduce tasks always local (MS is rank local buffer, Section 6.3.2), no peer eid
needed. CCU-produced UB tasks all explicitly carry `is_write` (Section 6.7); direction determination does not rely on rank
value.

### 10.2 Task Type Mismatch Table

| db taskType / Old Type | New Model TaskType | Mismatch Handling |
|---------------------|----------------|-----------|
| SYNC_STREAM (taskType=6) | **No corresponding type** | Skip, do not generate node (host-side stream sync point, not device-side task, Section 5.6); stats `skipped_sync` |
| AIV_GRAPH (taskType=5) | None | Skip + warning (old and new model consistent, both unsupported) |
| WRITE_WITH_NOTIFY (taskType=7, db new) | `WRITE_WITH_NOTIFY` (new, 7th type) | Single-task mapping (**does not split** to write + NOTIFY_RECORD): write always write direction takes port bandwidth sharing; attached notify per fixed record latency (Section 5.2, Section 10.3 #19 supported) |
| Old `REMOTE_RECORD` / `REMOTE_WAIT` | Already merged into `NOTIFY_RECORD` / `NOTIFY_WAIT` (src!=dst expresses remote) | db single-rank path's cross-rank notify as `src!=dst` generates remote node (Section 5.2, retains node not paired) |
| Old `MEMCPY` (network transfer) | Semantic change: new `Memcpy` = HBM copy | Network transfer now by `Ub_Write_Or_Read` (write/read direction distinguished); `Ub_Inline_Write` only retains notify inline write semantics (fixed latency) |
| Old `LOCAL_COPY` | Split to `Memcpy` (pure copy) / `Reduce_Inline` (reduce) | REDUCE not network -> Reduce_Inline |
| Old `MemcpyDirection` enum (direction field) | Deleted | Direction expressed by task type + rank/eid value (Section 5.5) |
| Old `sync_stream_delay` config item | Canceled along with type gap | Merged into local_record_delay reuse |

### 10.3 Other Conflicts (Future Match)

1. ~~**`src_rank == 0` local hard-code** (`simulator.py` TRANSFER branch): if read direction task's
   peer happens to be rank 0 and explicitly filled, would be mis-judged as write.~~ **Fixed**: Task adds explicit
   `is_write` field (db graph path fills; task.txt old path None falls back heuristic); simulator
   consumes first; taskData's peer rankId (deviceId field carries) simultaneously directly fills src/dst_rank (Sections 5.2/5.5).
2. **TopoResolver no fallback**: when eid cannot find in ranktable, directly throws error (no jettyId->port mapping,
   no default port). db path avoids by "network tasks exist and no ranktable -> skip simulation, dump only"
   (Section 3.3); future can consider connecting ccuChannels/JettyMaps to build fallback mapping.
3. **Broadcast pairing only triggers first wait** (matched record's `break`): when multiple waits pair same
   record, later waits' Event not triggered, end_time stays -1 (task silently lost).
4. **`is_matched_record` linear scan**: `dependency_graph.py` does O(R x M)
   traversal on matches; large-scale db (tens of thousands of tasks) has performance risk; recommend maintaining record->wait reverse index (also
   solves #3).
5. **SimConfig unit system and port key change**: JSON latency s -> **us**, bandwidth B/s -> **GBps**,
   port key int -> **"die/port"** string; old config file incompatible (this iteration already synchronizes example
   `src/plugin/light_performance/sim_config.json`).
6. **HBM dynamic sharing semantic change**: old version per-task independent `size / hbm_bandwidth`; new version
   when Reduce_Inline+Memcpy concurrent, equally shares HBM bandwidth (manager with utilization always 1.0); result metric
   has difference.
7. **Utilization query basis `min_remaining`**: same group concurrent tasks by minimum remaining data size queries utilization tier;
   different from old version's "each task by own data_size queries" - small tasks within group would drag down large tasks' utilization tier.
8. **Old input sources removed**: task.txt parser, AllToAll generator, built-in demo removed from input_builder
   along with HCCLSIM version; if log replay debug needed, must restore separately.
9. **TopoResolver no net_type filter**: when same eid has multiple network type entries, no priority rule.
10. **profiling json not implemented** (sub-req 5): current output is task graph dump (DOT/JSON/HTML)
    + text report + Chrome trace; perf_profiling.json format see Chapter 8 draft.
11. ~~**local_rank fallback convention**: when opDetails.rankId missing, falls back by `deviceId-1`
    (follows task.txt path convention); deviceId and rank numbering have no necessary relation; ranktable can
    precise.~~ Peer rank already resolved by eid->rank index (RankEidMap main path + ranktable
    fallback, Section 4.8); no longer relies on numbering convention; local still per opDetails.rankId as authoritative
    (channel leid cross-validation, Section 6.7).
12. ~~**CCU queue parallelism not modeled**: CCU queue chains as independent synthetic streams execute in parallel; real hardware's
    queue concurrent resource constraints not modeled (checker side has queue parallelization transformation; this plugin Section 6.7 simplified).~~
    **Queue parallelization implemented** (Section 7.4.1: assembly-phase skeleton/unit split + fork/join/attach extra edges +
    `#v` virtual stream cross-mission reuse). Still does not model: real hardware queue concurrent resource upper limit -
    virtual stream count unlimited (checker V3 also unlimited); future can cap by chip generation.
13. **@estimate one-shot channel does not pass data_id**: `plugin run @estimate` special-case branch directly executes
    entry (`cmd_base_utils.cc:626-663`); does not go through general message path's `DumpData` (its generated
    `data_id` currently fixed value "1234"); so output directory named by runtime timestamp (Section 8.1); future if
    switching to resident model, can align data_id semantics.
14. **`plugin install @estimate` semantic degradation**: install takes `RegisterPlugin` resident startup
    (fork+pipe); main.py one-shot execution completes then exits on stdin EOF; MonitorThread reclaims -
    behavior harmless but inconsistent with install's resident semantics; if truly resident needed, must implement stdin command loop
    (Section 3.2 evolution direction).
15. **Plugin artifacts not participating in auto-archive**: `archive.sh` only archives `logs/` and `data/`;
    `plugin/light_performance/data/`'s perf_profiling.json etc. artifacts do not enter archive directory
    (benefit is unaffected by start's `remove_all("data")`); if archive needed, must extend archive.sh.
16. **simpy deployment dependency**: target environment must pre-install simpy (not distributed with hccl_vm_install offline wheel);
    main.py startup import failure should give clear error and install prompt (Section 3.4).
17. ~~**db_reader.py not synced v3 collection layout**: C-side `HcclTaskMetaData` already expanded to 188B (after taskType
    added `HcclTaskProfInfo` 3 bytes, Section 4.4.1); `db_reader.py` still per 185B
    (`TASK_META_LEN=185`, `<BQIQQI16s16s`) decodes - length gate `len(blob) >= 185` cannot stop
    188B BLOB; reading v3-collected db would **fields overall mis-align** (from commId on). Need sync to
    188B/`<BBBBQIQQI16s16s` and decode profInfo three fields (value range/name see Section 11.4).~~
    **Synced**: `db_reader.py` by BLOB length auto-identifies v3 (188B/`<BBBBQIQQI16s16s`,
    profInfo three fields into `HcclTaskMetaData`) and old v2 (185B) layouts; see Section 4.4.1 implementation status.
18. ~~**JSON task type not consuming profInfo** (remaining narrowed): trace args'
    `transport_type`/`link_type` already switched profInfo-driven; CommProtocol migrated to new field
    `protocol` (Sections 11.4.2/11.4.3, Section 11.1 solution a landed); **remainder** is `task_type`
    field - currently still by TaskType enum + jettyId/direction conditional derivation (Section 11.2), with collection-side
    profInfo's hcomm op name has **semantic misalignment** (such as `Ub_Inline_Write` bidirectionally refers to different
    tasks, Section 11.4.4); after switching to `profInfo.profTaskType` driven, can correspond one-to-one with msprof timeline
    op name; target state comparison (including each value and simulation TaskType correspondence) see Section 11.4.1.~~
    **Landed (all remainder wrap-up)**: `Task` adds `task_name` display name (`task_model.py`,
    `display_name` = `task_name or task_type.value`); `graph_builder._add_task`
    uniformly injects `prof_task_type_name(profInfo.profTaskType)`; trace event name/`cat`
    (`report.py export_trace`), `args.task_type` (`to_trace_args`), verbose
    details and dump (DOT/JSON/HTML legend) all switched op name. Internal `TaskType` retired as
    latency model classification (Section 11.4.5 #3); latency/pairing/billing zero change. v2 old BLOB
    (profInfo=0xFF) shows `Unknown` (same metric as link/transport); CCU microcode subtasks
    have no profInfo, falls back internal name (Section 11.3). Misalignment elimination comparison see Section 11.4.4 implementation record.
19. ~~**WRITE_WITH_NOTIFY (taskType=7) decode missing**: `db_reader.py`'s
    `HccLTaskMetaType` IntEnum only 0~6; `decode_op_task_meta` on 7 directly raises and discards
    (counted into warning).~~ **Supported (single-task mapping)**: `db_reader.py` decodes
    `WriteWithNotifyTask` (Section 4.4.1) and single-task passes through; `graph_builder` maps to
    `WRITE_WITH_NOTIFY` TaskType (trace display name `Write_With_Notify`, Section 5.2);
    one-to-one with db task; latency = port write direction bandwidth sharing + post-write notify
    fixed record latency (Sections 7.3/7.4). **Does not adopt split strategy** (checker/runner side's
    "split to [write, NOTIFY_RECORD] two subtasks", and this plugin's early decode-boundary split both already
    deprecated): split would disperse task granularity, making trace/dump/stats unable to correspond one-to-one with db tasks.
    Collection side profInfo=TASK_WRITE_(REDUCE_)WITH_NOTIFY (op name `Write_With_Notify` /
    `Reduce_With_Notify`, hcomm source the latter is misspelled `Reuce_With_Notify`, see Section 11.4.5);
    both profInfo values mapped to same `Write_With_Notify` task (isReduce only affects
    data_type display).
20. ~~**Same mission multiple dispatches SQE merged to single context**: CCU expansion takes all SQEs of same
    `(dieId, missionId, queue)` into one dispatch context (single instruction window
    [first instStartId, last), single SQE args consumption) - does not satisfy checker
    `IsContiguousCcuSqe`'s "contiguous to merge" constraint (`task_meta_translator_v3.cc:213-231`).
    Typical victim scenario: instruction window wrap re-send (head SQE 13 Load SqeArgs + body SQE sent twice,
    second time uses new args); old implementation second dispatch: (1) window start per head SQE; tail segment
    JMP jumps over entire data transfer instruction segment; (2) SQE args already consumed by first dispatch; later Loads take 0 value
    (GSA addresses all 0); (3) loop count/CKE residual first state - second dispatch's
    Ub/Memcpy/local post overall lost; network data size only 3/10 of checker's.~~
    **Fixed (contiguous segment split)**: by `instStartId`/`instCnt` continuity split to multiple dispatch
    contexts (Section 6.1); each context independent window replay (including Load SqeArgs header) + independent sequential
    args consumption (`GetSqe` semantics); register/CKE bitmap shared across contexts (`AllRankParamRecorder`
    semantics); same-queue contexts serial by dispatch order; multi-segment task id with `_s{sequence}` suffix disambiguated
    (`ccu_graph_builder.expand_ccu_graph`). Post-fix with checker single-pass graph stream-by-stream aligned:
    record local/remote 54/54, wait local/remote 27/36, Ub 39/39, Memcpy 6/6,
    network+local data size 1088MB (2333 verification environment ALLGATHER dev1).
    Same-root-cause **wait node launch deviation** also fixed: old implementation per "each contributing post one wait node"
    split (one wait instruction can produce N nodes, inconsistent with checker `GenWaitNode`'s "per instruction at most
    1 local + 1 remote node"; in this environment manifests as local wait 48 vs 27); now aggregated to single node
    (mask = all contributing posts' bits OR; first contributing post takes notify pairing; rest via extra edges,
    Section 6.4); post-fix local wait 27/27 aligned.
21. **Merged loop folded pseudo-serial residual gap**: Section 6.5.1 signature chain resolution covers merge path's
    mid-body wait/record (folded pseudo-serial barrier), but two residuals: (1) full-expand fallback path
    (`mf > 0`)'s per-iteration wait/record is real 4KB granularity handshake; its inter-iteration overlap also
    serialized by queue chain (node count large, current example not triggered, not handled for now); (2) per-iteration handshake
    **rate** overhead not modeled - folding and resolution both counted once (if CKE engine throughput is bottleneck, needs dedicated
    rate model, such as ns/time x 32768 times magnitude evaluation).

---

## 11. db Task Type and JSON Artifact Task Type Mapping Summary Table

> This chapter is a type mapping quick reference, answering three questions:
> (1) Which JSON artifact task types do tasks in the db database (`opTask_P_<pid>` table taskType column) ultimately convert to (Sections 11.1, 11.2);
> (2) CCU mode microcode instruction and JSON task type correspondence (Section 11.3);
> (3) Collection-side profiling reconciliation info (`HcclTaskProfInfo`) and JSON
> display type correspondence and target state (Section 11.4). Implementation basis: `src/plugin/light_performance/
> graph_builder.py` (SQE mapping, `_map_tasks`/`_map_transfer`),
> `src/plugin/light_performance/ccu_graph_builder.py` (CCU instruction mapping, `transform`
> family), `src/plugin/light_performance/simulator/task_model.py` (TaskType enum).
> Same source as Section 5.2 (SQE mapping + latency model), Section 6.3 (instruction major class mapping); here reorganized by
> "db -> JSON" correspondence, with reverse index supplemented.
>
> **profInfo background**: From v3 collection on, db task's opTaskMeta BLOB carries `HcclTaskProfInfo`
> substructure (Section 4.4.1, single-source defined in `include/sim_profiling_defs.h`; checker/runner as
> opaque pass-through); three fields `profTaskType`/`linkType`/`transportType` value ranges and
> names aligned one-to-one with hcomm `TaskParamType`/`DfxLinkType`/`DfxTransportType` - i.e.,
> msprof timeline op name and L1 detail linkType/transportType fields. This chapter 11.1~11.3 describes
> **current TaskType implementation** (not yet consuming profInfo); 11.4 gives three-field full comparison and switch
> target state (todo see Section 10.3 #17~#18).

### 11.1 JSON Artifact Task Type Carriers

Current implementation's JSON artifacts total three types; task types in all taken from `TaskType` enum values
(string form, `task_model.py:20-26`):

| JSON Artifact | Format | Task Type Field | Generation Module |
|-----------|------|----------------|---------|
| `perf_profiling.json` (main artifact, Section 8.1) | Chrome Trace Event | `traceEvents[].name` / `.cat` / `.args.task_type` | `simulator/report.py` (`export_trace`; event name is task display name: profInfo op name, Section 11.4) |
| `trace_comm<commId>_iter<opIter>.json` (optional per-operator trace) | Chrome Trace Event | same as above | same as above |
| Task graph dump `graph_comm<commId>_iter<opIter>_dev<dev>.json` (optional, `--dump-dir`) | `light_perf.graph_dump/2` | `nodes[].type` | `graph_dumper.py` |

> dump JSON's notify nodes when mask non-zero carry `nodes[].mask` (Section 7.3 Task.mask);
> HTML view tooltip synchronized display; `perf_profiling.json`/trace task args also contains
> `mask` (Section 7.7).

Full set of task types appearing in JSON (7 types):

| TaskType Value (JSON Presentation) | Latency Model | Semantics |
|--------------------------|---------|------|
| `Notify_Record` | Fixed latency (local also triggers Event) | notify set (record / CKE post) |
| `Notify_Wait` | Wait Event / fixed latency | notify wait (wait / CKE wait) |
| `Reduce_Inline` | HBM dynamic sharing | Local reduce (computation approximated as transfer, Section 5.7) |
| `Memcpy` | HBM dynamic sharing | Local copy (HBM, not network) |
| `Ub_Inline_Write` | Fixed latency (same as Notify_Record) | notify inline write (old input path only) |
| `Ub_Write_Or_Read` | Port bandwidth sharing (write/read direction billed independently) | Network data send/receive (SQE network tasks two directions uniformly mapped) |
| `Write_With_Notify` | Port write direction bandwidth sharing + post-write notify fixed latency | Network write + attached notify (one-to-one with db `WRITE_WITH_NOTIFY`, Section 5.2) |

> Section 8.3's `perf_profiling.json` structured format (meta/ops array) is draft; current main artifact
> actually Chrome tracing format (Section 10.3 #10); regardless of format, task types are all above
> 7 TaskType values; this chapter's mapping table applies uniformly.

**trace args type-related fields (current state and profInfo consumption)**: Task's `to_trace_args`
(`task_model.py`) outputs four type-related fields; all four already switched profInfo-driven
(Section 10.3 #18 all landed, Section 11.4):

| trace args Field | Current Source | Status |
|----------------|---------|--------------------------|
| `task_type` | `profInfo.profTaskType`'s op name (`prof_task_type_name`, `graph_builder._add_task` uniform fill as `Task.task_name`; when `display_name` empty, falls back internal TaskType name) | **Landed** (one-to-one with msprof timeline op name, Section 11.4.1; v2 old BLOB is `Unknown`) |
| `transport_type` | `profInfo.transportType` name (`ProfTransportTypeName`, Section 11.4.3, `graph_builder._add_task` uniform fill) | **Landed** |
| `link_type` | `profInfo.linkType` name (`ProfLinkTypeName`, Section 11.4.2, `graph_builder._add_task` uniform fill) - **note this is different concept from `protocol` (CommProtocol)** | **Landed** (solution a: overridden as DfxLinkType name) |
| `protocol` (new field) | Channel `CommProtocol` enum name (HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO, via `graph_builder._resolve_eids` / CCU `_channel_eids` filled) | **Landed** (CommProtocol migrated from `link_type`; when no channel info, `N/A`) |

> `link_type` override (solution a) implemented: semantics aligned with msprof L1 detail field name, avoiding same-name different-meaning;
> existing CommProtocol info not lost; moved into `protocol` field. CCU microcode subtasks are simulator
> semantic artifacts, no profInfo (Section 11.3); their `link_type`/`transport_type` is `N/A`;
> `protocol` retains channel CommProtocol name.

### 11.2 SQE Path: db taskType -> JSON Task Type Mapping Summary Table

Forward mapping (db task -> JSON task display name). "JSON task type" column is trace/dump display
op name (already switched `profInfo.profTaskType` driven, Section 10.3 #18 landed); "internal TaskType"
(latency model classification, determines billing/pairing) keeps Section 11.2 original determination conditions (jettyId/direction/recordDev);
correspondence with op name see Section 11.4.1 "Simulation TaskType" column:

| db taskType (opTask.taskType column) | Determination Condition | JSON Task Type (op name) | Collection-side profInfo.profTaskType (op name) | Description |
|----------------------------------|---------|--------------|--------------------------------------|------|
| `NOTIFY_WAIT` (0) | recordDev == local device | `Notify_Wait` | `TASK_NOTIFY_WAIT` -> `Notify_Wait` OK | Local wait: participates in first-come-first-served pairing (Section 5.4); waits Event |
| `NOTIFY_WAIT` (0) | recordDev != local device | `Notify_Wait` | `TASK_NOTIFY_WAIT` -> `Notify_Wait` OK | Cross-rank wait: node retained not paired; src!=dst -> fixed `remote_wait_delay` |
| `NOTIFY_RECORD` (1) | waitDev == local device (SDMA/local notify) | `Notify_Record` | `TASK_NOTIFY_RECORD` -> `Notify_Record` OK | Local record: participates in pairing; triggers Event |
| `NOTIFY_RECORD` (1) | waitDev != local device (SDMA/local notify) | `Notify_Record` | `TASK_NOTIFY_RECORD` -> `Notify_Record` OK | Cross-rank post: node retained not paired; src!=dst -> fixed `remote_record_delay` |
| `NOTIFY_RECORD` (1) | UBDMA WQE `inlineEn==1` (notify inline write) | `Ub_Inline_Write` | `TASK_UB_INLINE_WRITE` -> `Ub_Inline_Write` OK | notify inline write (name returned, Section 11.4.4); internal TaskType still Notify_Record (fixed record latency) |
| `REDUCE` (2) | jettyId == INVALID | `Reduce_Inline` | `TASK_REDUCE_INLINE` -> `Reduce_Inline` OK | Local reduce: HBM dynamic sharing billing |
| `REDUCE` (2) | jettyId valid + write direction | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` -> `Ub_Reduce_Inline` OK | Network reduce approximation (Section 5.7); internal TaskType=Ub_Inline_Write; port write direction billing |
| `REDUCE` (2) | jettyId valid + read direction | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` -> `Ub_Reduce_Inline` OK | same as above; internal TaskType=Ub_Write_Or_Read; read direction billing |
| `MEM_CPY` (3) | jettyId == INVALID (same device) | `Memcpy` | `TASK_SDMA` -> `Memcpy` OK | Local copy: HBM dynamic sharing billing; profInfo.linkType=ONCHIP |
| `MEM_CPY` (3) | jettyId == INVALID (cross-device SDMA SQE) | `Memcpy` | `TASK_SDMA` -> `Memcpy` OK | Cross-die SDMA transfer; also Memcpy; difference expressed by profInfo.linkType=PCIE / transportType=SDMA (Section 11.4) |
| `MEM_CPY` (3) | jettyId valid + write direction | `Ub_Write_Or_Read` | `TASK_UB` -> `Ub_Write_Or_Read` OK | Network send (direction not encoded into type name; expressed by direction/is_write); internal TaskType=Ub_Inline_Write |
| `MEM_CPY` (3) | jettyId valid + read direction | `Ub_Write_Or_Read` | `TASK_UB` -> `Ub_Write_Or_Read` OK | Network receive; internal TaskType=Ub_Write_Or_Read |
| `WRITE_WITH_NOTIFY` (7) | Always write direction (no direction inference, Section 5.5) | `Write_With_Notify` (single task, not split); when `udfFlag==1`, `Reduce_With_Notify` | `TASK_WRITE_WITH_NOTIFY` -> `Write_With_Notify`; when `udfFlag==1`, `TASK_WRITE_REDUCE_WITH_NOTIFY` -> `Reduce_With_Notify` (hcomm source is misspelled `Reuce_With_Notify`, Section 11.4.5) | Write+notify combined task (`WriteWithNotifyTask`, Section 4.4.1): port write direction bandwidth sharing + post-write notify fixed record latency; both profInfo values mapped to same internal WRITE_WITH_NOTIFY (latency model consistent; display name and isReduce distinguished) |
| `CCU_GRAPH` (4) | - | (placeholder, does not directly produce) | `TASK_CCU` -> `Ccu` | Expands to CCU subgraph; final artifact see Section 11.3 |
| `AIV_GRAPH` (5) | - | None (skip) | `TASK_AIV` -> `AivKernel` | Warning + `skipped_aiv` count (not supported for now) |
| `SYNC_STREAM` (6) | - | None (skip) | `UNKNOWN` -> `Unknown` | Host-side stream sync point, not device-side task (Section 5.6); hcomm profiling has no corresponding type |

> Cross-rank determination (waitDev/recordDev) see Section 4.6; network direction determination (write/read) see Section 5.5.

Reverse index (JSON task display name <- all sources, including CCU path and profInfo dimension; SQE path
display name = profInfo op name; CCU path display name = internal TaskType name, Section 11.3):

| JSON Task Display Name | SQE Source (db taskType) | Collection-side profInfo Source (op name) | CCU Source (microcode instruction, Section 11.3) |
|--------------|------------------------|------------------------------|------------------------------|
| `Notify_Record` | `NOTIFY_RECORD` (1), local/cross-rank (SDMA/local notify) | `Notify_Record` (TASK_NOTIFY_RECORD OK) | SetCKE/SetCKBit local post; TRANS/REDUCE/LOAD class instruction's embedded `setCKE` field (A6 LOAD class also carries); cross-rank set/notify (A5 Sync family setRmtCKE; A6 TransMem notify/0x1A atomic, SyncWtX notify, SyncAtx - remote approximation); subgraph attach launch-record / completion-record |
| `Notify_Wait` | `NOTIFY_WAIT` (0), local/cross-rank | `Notify_Wait` OK | Various instructions' embedded `waitCKE` field; ClearCKE; A6 independent Wait instruction (Xn value compare, unsatisfied -> remote wait); subgraph attach completion-wait / launch-wait |
| `Reduce_Inline` | `REDUCE` (2), jettyId invalid | `Reduce_Inline` OK | REDUCE class instructions (msId list all local) |
| `Memcpy` | `MEM_CPY` (3), jettyId invalid (including cross-die SDMA) | `Memcpy` OK | TRANS local transfer instructions (A5 4 codes: 0x0/0x2/0x5/0xA; A6 4 codes: 0x0/0x2/0x5/0x6) |
| `Ub_Inline_Write` | `NOTIFY_RECORD` (1) + `inlineEn==1` (notify inline write) | `Ub_Inline_Write` (TASK_UB_INLINE_WRITE OK, name returned) | - (CCU network transfer uniformly mapped `Ub_Write_Or_Read`, Section 11.3.1) |
| `Ub_Write_Or_Read` | `MEM_CPY` (3) / `REDUCE` (2), jettyId valid (write/read direction) | `Ub_Write_Or_Read` (TASK_UB OK; direction expressed by direction/is_write) | TRANS network transfer instructions (A5 write role 3 codes: 0x3/0x7/0x9, read role 4 codes: 0x1/0x4/0x6/0x8; A6 unified TransMem 0x10: dmaOpCode 3/5=write, 6=read, including udfEnable network reduce approximation); REDUCE class containing remote MS |
| `Ub_Reduce_Inline` | `REDUCE` (2), jettyId valid (network reduce) | `Ub_Reduce_Inline` (TASK_UB_REDUCE_INLINE OK) | - |
| `Write_With_Notify` | `WRITE_WITH_NOTIFY` (7), isReduce=0 | `Write_With_Notify` (TASK_WRITE_WITH_NOTIFY OK) | - |
| `Reduce_With_Notify` | `WRITE_WITH_NOTIFY` (7), udfFlag==1 | `Reduce_With_Notify` (TASK_WRITE_REDUCE_WITH_NOTIFY OK) | - |

### 11.3 CCU Path: Microcode Instruction -> JSON Task Type Mapping

db's `CCU_GRAPH` (4) SQE itself is only placeholder (`CcuPlaceholder`, Section 6.1); final
JSON task produced from `ccuInstrRes` instruction space's microcode interpretation (Chapter 6). **A5 (V1,
Ascend950, devType=6) and A6 (V2, Ascend960, devType=8) both instruction sets implemented**
(selected by `opDetails.devType`; other devType falls back A5): A5 summary table in Section 11.3.1;
A6 summary table in Section 11.3.3.

> profInfo correspondence: CCU placeholder SQE's `HcclTaskProfInfo` overall is
> `TASK_CCU` -> `Ccu` (linkType/transportType is UNKNOWN; microcode has no link dimension). Expanded
> microcode subtasks (`Memcpy`/`Ub_Write_Or_Read`/`Notify_*` etc.) are
> simulator's semantic artifacts, **no corresponding ProfTaskParamType**; display name falls back internal TaskType name
> (Section 10.3 #18 implementation's established trade-off, Section 11.4.4); msprof side CCU tasks via
> microcode-level detail (`CcuProfilingInfo`: CCU_TASK/WAITCKE/LOOPGROUP three classes, reported along with TASK_CCU)
> expresses equivalent granularity info; for reconciliation, uses whole SQE's `Ccu` as alignment anchor.

#### 11.3.1 A5 Instruction Summary Table (Instruction -> JSON Task Type)

Microcode instruction = 2B header (type:4bit major class + code:11bit instruction code, Section 6.2) + 30B payload:

| Instruction Major Class (type) | Instruction (code) | JSON Task Type | Description |
|----------------|------------|--------------|------|
| LOAD (0x0) | LoadSqeArgsToGSA (0x0), LoadSqeArgsToXn (0x1), LoadImdToGSA (0x2), LoadImdToXn (0x3), LoadGsaXn (0x4), LoadGsaGsa (0x5), LoadXx (0x6) | **None** (does not generate node) | Only reads/writes register model (xn/gsa), for later instructions to fetch length, loop count etc. |
| CTRL (0x1) | SetCKE (0x2) | `Notify_Record` | Local CKE post (ckeId as notify_id, ckeMask as mask, src=dst=local); embedded `waitCKE` field can additionally produce `Notify_Wait` |
| CTRL (0x1) | ClearCKE (0x4) | `Notify_Wait` | wait semantics (embedded `waitCKE` field); clear action itself does not produce node |
| CTRL (0x1) | Loop (0x0) | **None** | Must be triggered by LoopGroup; standalone appearance is abnormal, warning skip |
| CTRL (0x1) | LoopGroup (0x1) | (no direct node) | Loop control: merge/expand loop body (Section 6.5); body instructions mapped per this table; under merge path, data task size x total iteration count; launched tasks carry loop_cnt (post-expand loop count) / loop_iter_cnt (single loop self-loop count); total iteration count = product of two fields |
| CTRL (0x1) | Jmp (0x5) | **None** | Conditional jump: when `Xn[conditionXnId] != expectData`, pc jumps to instruction pointed by `Xn[dstInstrXnId]` (checker `TransformJmpInstrA5` semantics; target limited to current instruction window, out-of-range warning pass-through); within loop body pass-through (merge pre-validation already rejected loop body containing JMP); with step budget defense against backward jump infinite loop |
| TRANS (0x2) | LocMemToLocMs (0x0), LocMsToLocMem (0x2), LocMsToLocMs (0x5), LocMemToLocMem (0xA) | `Memcpy` (src/dst_buf_type labels mem/ms) | Local transfer (LOCAL_COPY role, Section 6.3.2), HBM dynamic sharing |
| TRANS (0x2) | LocMsToRmtMem (0x3), LocMemToRmtMem (0x9), LocMsToRmtMs (0x7), RmtMemToLocMs (0x1), RmtMsToLocMem (0x4), RmtMsToLocMs (0x6), RmtMemToLocMem (0x8) | `Ub_Write_Or_Read` (is_write distinguishes direction; src/dst_buf_type labels mem/ms) | Cross-die/device transfer (write/read roles unified type, Section 6.3.2); peer rank via channel reid -> eid->rank index resolution (Sections 4.8/6.7) |
| TRANS (0x2) | SyncCKE (0xB), SyncGSA (0xC), SyncXN (0xD) | Local post part -> `Notify_Record`; remote post part -> `Notify_Record` (remote, not paired) | Remote set field (SyncCKE's `rmtCKEId/locCKEMask`, SyncGSA/SyncXN's `setRmtCKEId/setRmtCKEMask`) generates remote record node (src=local, dst=INVALID -> fixed `remote_record_delay`, does not register local accounting); counted into `remote_posts` (Section 6.6); SyncGSA/SyncXN carrying second remote post also generates; embedded `waitCKE` can produce `Notify_Wait` |
| REDUCE (0x3) | Add (0x0), Max (0x1), Min (0x2) | `Reduce_Inline` | Reduce computation approximated as transfer (Section 5.7); MS always rank local buffer (msId high bits only addressing encoding, Section 6.3.2); local HBM billing |
| (Subgraph attach) | - (synthesized at graph phase, not microcode) | None (does not produce task node) | Attach dependencies all via extra edges expressed (Section 7.4.1): entry edge SQE predecessor -> skeleton head / pre-skeleton unit head; exit edge mission completion set -> SQE successor; task graph does not凭空 produce any node |

#### 11.3.2 Embedded Sync Fields (Cross-Instruction Universal)

TRANS / REDUCE / SetCKE / Sync class instructions' payload generally carries sync fields, with main semantics
**additionally producing** JSON tasks:

| Payload Field | JSON Task Type | Behavior |
|---------|--------------|------|
| `waitCKEId` / `waitCKEMask` | `Notify_Wait` | When bits already satisfied, pairs with local post accounting (aggregated to single node: mask = all contributing posts' consumption bits OR; first post add_match; rest extra edges, Section 10.3 #20); when bits insufficient: top-level instruction blocks that queue (stall detection finds then generates remote wait node and replays rest of instruction, `remote_waits`); within loop body cannot block, directly generates remote wait node and continues |
| `setCKEId` / `setCKEMask` | `Notify_Record` | Local CKE post, registers seen_post accounting (Section 6.4) |
| `clearType` bits / SetCKE's clear flag | None (only register clear) | Clears bits after wait completes; does not produce node |

> Therefore one TRANS instruction can produce 1 data transfer task (`Memcpy`/`Ub_Write_Or_Read`),
> and by embedded fields additionally produce several `Notify_Wait` (paired by bits,
> 0..n) and 1 `Notify_Record`. Under merged loop (Section 6.5), each slot similarly retains
> wait/set tasks; only data tasks folded to size x total iteration count.
>
> **A6 difference**: embedded `waitCKE` only exists in SetCKBit/ClearCKE (CTRL class); local
> TRANS/REDUCE and TransMem/Sync class only carry tail `setCKEId/setCKEMask` (@26/28,
> additionally producing `Notify_Record`); wait semantics taken by independent Wait instruction (Xn value compare).

#### 11.3.3 A6 (Ascend960) Instruction Summary Table (Instruction -> JSON Task Type)

devType=960 (`DEV_TYPE_960`) corresponds to V2/A6 instruction set, **implemented** (ported from checker
`ccu_task_transform_instruct_a6_v3.cc`, interpreter `_MissionContextA6`). Microcode instruction
= 2B header (isomorphic with A5) + 30B V2 payload (`ccu_microcode_v1.h` CcuV2 struct,
pack(1)); register number generally via `GetXnId` (bit15 addressing mode, Section 6.4).

| Instruction Major Class (type) | Instruction (code) | JSON Task Type | Description |
|----------------|------------|--------------|------|
| LOAD (0x0) | LoadSqeArgsToX (0x1), LoadImdToX (0x2), LoadX (0x6), StoreX (0x7), ClearX (0x8), Nop (0x9), arithmetic Add (0xD)/Sub (0xE)/Mul (0xF)/And (0x10)/Or (0x11)/Not (0x12)/Xor (0x13)/Shl (0x14)/Shr (0x15)/Popcnt (0x16) | Data plane **none**; when `setCKEMask != 0` additionally `Notify_Record` | Unified Xn register read/write (no GSA): SQE args sequentially consumed (Section 6.1), immediate load, indirect read/write (LoadX/StoreX, oMode 0=immediate offset/1=register offset), range clear (ClearX), 64-bit wrap arithmetic (Mul takes low 32-bit product, shift amount 0x3F truncated, Shl/Shr supports logical/arithmetic/circular three types); all except Nop carry tail setCKE field (checker `ProcessSetMask`; mask non-zero -> local post node) |
| LOAD (0x0) | Load (0xA), Store (0xB) | Data plane **none**; setCKE additionally `Notify_Record` | HBM <-> Xn data plane: HBM content not modeled (checker via AllRankParamRecorder reads/writes data); register not backfilled; warning leaves trace; only retains sync semantics |
| CTRL (0x1) | SetCKBit (0x2) | `Notify_Record` (+ embedded `waitCKE` additionally `Notify_Wait`) | wait -> local post -> when clearType==1 clears wait bits (checker only supports clearType=1, rest warning skip clear) |
| CTRL (0x1) | ClearCKE (0x4) | `Notify_Wait` (embedded `waitCKE`) | After wait, per checker `ClearWaitMask` semantics clears consumed wait bits; `clearMask != 0` is checker unsupported scenario (warning + tolerant clear target bits) |
| CTRL (0x1) | Loop (0x0) | **None** | Must be triggered by LoopGroup; standalone appearance warning skip. `mode==1` conditional Loop: `wishCKEBit == Xp.loopCtxId + 1024` hit to expand (Section 6.5) |
| CTRL (0x1) | LoopGroup (0x1) | (no direct node) | Loop control: parameters from Xn/Xm/Xp three registers (loopInsCnt:10/expandOffset:9/expandCnt:9 etc., Section 6.5); merge/expand same as A5 (merge validation includes A6 additional rules: wait CKE ID no reuse, cross-expand iteration count/length/channel consistent); config illegal (loopInsCnt>512 etc.) warning skip |
| CTRL (0x1) | Jmp (0x5) | **None** | Conditional jump: conditionType 0~5 (==/!=/>/>=/</<=), >=6 unconditional; jumpMode 0=relative pc (mod 0x10000)/1=absolute instruction number; target takes `Xn[relTarInstrXnId]`; limited to current instruction window, out-of-range warning pass-through; within loop body pass-through; step budget defense against infinite loop |
| CTRL (0x1) | Wait (0x7) | When unsatisfied, remote `Notify_Wait` | Xn value compare spin (conditionType same as Jmp, default per equal); satisfied passes through no node; unsatisfied = depends on peer remote register write -> top-level stall resolution / within loop body directly generates remote wait node (fixed `remote_wait_delay`; task id label `xwt`; counted into `remote_waits`) |
| CTRL (0x1) | Fence (0x8) | **None** | checker does not support (errors and aborts); here warning pass-through (queue serial model order naturally guaranteed) |
| TRANS (0x2) | TransLocMemToLocMS (0x0), TransLocMSToLocMem (0x2), TransLocMSToLocMS (0x5), TransLocMemToLocMem (0x6) | `Memcpy` (src/dst_buf_type labels mem/ms) | Local transfer (length `Xn[GetXnId(xlId)]`); no embedded waitCKE; tail setCKE additionally `Notify_Record` |
| TRANS (0x2) | TransMem (0x10) | `Ub_Write_Or_Read` (dmaOpCode 3/5=write, 6=read); notify/atomic -> remote `Notify_Record` | Unified network transfer: channel = `Xn[GetXnId(xcId)]` register value -> peer rank/eid resolution same as A5 (Sections 4.8/6.7); when `msIdMode==1`, write source / read destination is local MS (buf_type=ms); dmaOpCode 5's write-then-notify and 0x1A remote atomic count (no data plane) approximated as remote record (mask = notify value low 16 bits); `udfEnable` network reduce billed as normal UB transfer (Section 5.7 approximation); 0x19 and other dmaOpCode / `udfType != 0` warning skip data plane (setCKE post still executes) |
| TRANS (0x2) | SyncXnWtx (0xD), SyncAtx (0xE) | Remote `Notify_Record` (approximation); SyncAtx also has setCKE -> local `Notify_Record` | Cross-rank register sync: SyncWtX's `notifyValid==1` notify and SyncAtx's atomic write approximated as remote record (remote target type single-rank unresolvable, Section 6.6); main value sync as remote register write does not build node (A5 SyncXn same metric); SyncWtX tail setCKE field checker does not consume (local does not post); channel unresolvable warning skip (`eid_unresolved`) |
| REDUCE (0x3) | ReduceAdd (0x0), ReduceMax (0x1), ReduceMin (0x2) | `Reduce_Inline` | Local MS batch reduce (msId[8] list + `Xn[XnIdLength]` length); no embedded waitCKE; tail setCKE additionally `Notify_Record`; MS always local HBM billing (Section 6.3.2) |
| (Subgraph attach) | - (synthesized at graph phase, not microcode) | None (does not produce task node) | Same as A5 (Section 7.4.1 extra edges attach) |

> A6 established deviations with checker (HBM data plane, remote target type resolution, checker error scenario
> tolerant downgrade etc.) summary see `_MissionContextA6` class docstring and Section 6.7 comparison table;
> loop merge/pipeline concurrency recovery (Sections 6.5/6.5.1), queue parallelization (Section 7.4.1), SQE merge
> and stall resolution (Sections 6.1/6.6) etc. framework mechanisms shared by two generations.

### 11.4 HcclTaskProfInfo Three-Field Comparison (profInfo -> JSON Display)

From v3 collection on, db tasks carry `HcclTaskProfInfo` (Section 4.4.1, single-source defined
`include/sim_profiling_defs.h`). This section gives three-field "value -> enum -> JSON display string"
complete comparison and each value's correspondence with simulation TaskType (latency model classification, Section 7.3);
C++ side name functions (`ProfTaskParamTypeName`/`ProfLinkTypeName`/
`ProfTransportTypeName`) are display string's only authoritative source.

> **Implementation status**: `task_type`/`transport_type`/`link_type`/`protocol` four fields all per
> this section comparison landed (Python-side display name mapping in `db_reader.py`'s `PROF_*_NAMES`, character-by-character consistent with C++
> name functions, Section 11.4.5; `graph_builder._add_task` uniform fill - `task_type`
> via `Task.task_name`/`display_name`, Section 10.3 #18 wrapped up).

#### 11.4.1 profTaskType (ProfTaskParamType, 0~22 -> msprof timeline op name)

This table supplements each value's correspondence with simulation TaskType (`task_model.py`, latency model classification) - previously
db task and simulation task type matched via db taskType column (Section 11.2); after profInfo consumption, this table
`profTaskType -> simulation TaskType` as second match dimension (determination conditions unchanged; same db taskType
by jettyId/direction routed to different TaskType rows annotated):

| Value | Enum | JSON Display String (op name) | Simulation TaskType (latency model classification) | light_perf Occurrence Scenario |
|---|---|---|---|---|
| 0 | TASK_SDMA | `Memcpy` | `Memcpy` (jettyId invalid, HBM billing) / `Ub_Inline_Write` * `Ub_Write_Or_Read` (jettyId valid by direction, port billing) | Local copy, cross-die SDMA SQE (linkType distinguishes, Section 11.4.2) OK |
| 1 | TASK_RDMA | `RDMASend` | `Memcpy` (jettyId invalid, HBM billing approximation) | DPU host ibv normal read/write (`RDMASend`, hcomm UbMemTransport::Write/Read same metric) OK |
| 2 | TASK_REDUCE_INLINE | `Reduce_Inline` | `Reduce_Inline` (jettyId invalid, HBM billing) / `Ub_Inline_Write` * `Ub_Write_Or_Read` (network approximation, Section 5.7, port billing) | Local reduce OK |
| 3 | TASK_REDUCE_TBE | `Reduce_TBE` | - (does not appear) | 910 generation reserved, does not appear |
| 4 | TASK_NOTIFY_RECORD | `Notify_Record` | `Notify_Record` | SDMA/local notify record OK |
| 5 | TASK_NOTIFY_WAIT | `Notify_Wait` | `Notify_Wait` | notify wait OK |
| 6/7 | TASK_SEND_NOTIFY / TASK_SEND_PAYLOAD | `Send_Notify` / `Send_Payload` | - (does not appear) | hcomm reserved, does not appear |
| 8 | TASK_WRITE_WITH_NOTIFY | `Write_With_Notify` | `Write_With_Notify` | WRITE_WITH_NOTIFY (7) whole task (single-task mapping, Section 10.3 #19 supported) OK |
| 9 | TASK_WRITE_REDUCE_WITH_NOTIFY | `Reduce_With_Notify` (tool-side corrected spelling; hcomm source/msprof is misspelled `Reuce_With_Notify`, Section 11.4.5) | `Write_With_Notify` (isReduce=1 reduce write payload, data_type distinguishes display) | WRITE_WITH_NOTIFY (7) isReduce=1 |
| 10 | TASK_CCU | `Ccu` | (CCU placeholder whole SQE; expands to Chapter 6 subgraph, Section 11.3) | CCU placeholder whole SQE OK |
| 11/12 | TASK_AICPU_KERNEL / TASK_AICPU_REDUCE | `AicpuKernel` / `Aicpu_Reduce` | - (not consumed) | AICPU kernel delivery (goes through device process; current no collection landing point / not consumed) |
| 13 | TASK_AIV | `AivKernel` | - (skip) | AIV kernel task (skip) |
| 14 | TASK_UB_INLINE_WRITE | `Ub_Inline_Write` | `Notify_Record` (notify fixed record latency; display name returned OK) | UBDMA `inlineEn==1` notify inline write |
| 15 | TASK_UB_REDUCE_INLINE | `Ub_Reduce_Inline` | `Ub_Inline_Write` (write) / `Ub_Write_Or_Read` (read) (network reduce approximation, port billing) | UBDMA `udfFlag==1` network reduce |
| 16 | TASK_UB | `Ub_Write_Or_Read` | `Ub_Inline_Write` (write direction) / `Ub_Write_Or_Read` (read direction) (direction expressed by direction/is_write) | UBDMA normal read/write WQE (write/read two directions same value) |
| 17~22 | TASK_DPU_KERNEL / THREAD_FENCE / CHANNEL_FENCE / INLINE_WRITE / NOTIFY_WAIT / WRITE_WITH_NOTIFY | `DpuKernel` / `Dpu_ThreadFence` / `Dpu_ChannelFence` / `Dpu_Notify_Record` / `Dpu_Notify_Wait` / `Dpu_Write_With_Notify` | `Dpu_Notify_Record` -> `Notify_Record`, `Dpu_Notify_Wait` -> `Notify_Wait` (cross-rank fixed latency), `Dpu_Write_With_Notify` -> `Write_With_Notify` (single-task port billing, Section 11.4.6); `DpuKernel`/Fence class current collection path does not produce | host CPU RoCE (ibv_dpu_stub) task OK (Section 11.4.6) |
| 0xFF | UNKNOWN | `Unknown` | - (SYNC_STREAM etc. already skipped, does not become node) | SYNC_STREAM, level1 channel API tasks, v2 old BLOB etc. unlabeled sources |

> `task_type` already switched to op name (Section 10.3 #18 wrapped up); this table's "Simulation TaskType" column now is
> internal latency model classification index (Section 11.4.5 #3); OK row op name consistent with TaskType value.
> After SQE batch merge (Section 5.1.1), merged task's op name and link/transport takes batch's **last**
> data task's profInfo (description follows last task; length takes full batch sum); v3 and earlier BLOB has no
> sqeGroupId, does not merge, per-task display.

#### 11.4.2 linkType (ProfLinkType, 0~9 -> hcomm DfxLinkType enum name)

| Value | Enum | JSON Display String | light_perf Occurrence Scenario |
|---|---|---|---|
| 0 | LINK_ONCHIP | `ONCHIP` | Local Memcpy/Reduce_Inline/Notify_* |
| 1 | LINK_HCCS | `HCCS` | Reserved (current collection path does not produce) |
| 2 | LINK_PCIE | `PCIE` | Cross-die SDMA SQE (same device determined as local ONCHIP, Section 4.4.1 collection rules) |
| 3 | LINK_ROCE | `ROCE` | host CPU RoCE (DPU, ibv_dpu_stub) task OK (Section 11.4.6) |
| 4/5/6 | LINK_SIO / HCCS_SW / STANDARD_ROCE | `SIO` / `HCCS_SW` / `STANDARD_ROCE` | Reserved |
| 7 | LINK_UB | `UB` | Network task (UBDMA WQE all classes) |
| 8 | LINK_UBoE | `UBoE` | UBoE channel (collection side not yet distinguishes UB/UBoE, does not appear for now) |
| 9 | LINK_RESERVED | `RESERVED` | Reserved |
| 0xFF | UNKNOWN | `Unknown` | CCU/AIV/SYNC_STREAM, level1 channel API tasks |

> **Implementation status**: trace args' `link_type` field already per this table filled (`prof_link_type_name`,
> `graph_builder._add_task` uniform inject, Section 11.1 solution a); this table and original `link_type`'s carried
> `CommProtocol` channel protocol (HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO) are **two different
> concepts**: CommProtocol describes channel establishment protocol; DfxLinkType describes task execution link (profiling
> dimension). CommProtocol name already migrated to `protocol` field (when no channel info, `N/A`).

#### 11.4.3 transportType (ProfTransportType, 0~4 -> msprof L1 detail transportType value range)

| Value | Enum | JSON Display String | light_perf Occurrence Scenario |
|---|---|---|---|
| 0 | TRANSPORT_SDMA | `SDMA` | Cross-die SDMA SQE |
| 1 | TRANSPORT_RDMA | `RDMA` | Reserved (910 generation RDMA semantics) |
| 2 | TRANSPORT_LOCAL | `LOCAL` | Local task (Memcpy/Reduce_Inline/Notify_*) |
| 3 | TRANSPORT_UB | `UB` | Network task (**latency model anchor**: port bandwidth sharing billing, Section 5.5) |
| 4 | TRANSPORT_ROCE | `ROCE` | host CPU RoCE (DPU, ibv_dpu_stub) task OK (Section 11.4.6) |
| 0xFF | UNKNOWN | `Unknown` | Unlabeled tasks |

> **Implementation status**: trace args' `transport_type` field already per this table filled
> (`prof_transport_type_name`, `graph_builder._add_task` uniform inject).
> Same-name `Memcpy` task's local/cross-die difference expressed by linkType+transportType:
> ONCHIP+LOCAL (local) vs PCIE+SDMA (cross-die) - trace args now can losslessly distinguish these two classes of
> previously indistinguishable tasks in JSON.

#### 11.4.4 Ub_Inline_Write Bidirectional Semantic Misalignment and Switch Implementation Record

**Historical issue**: Before switch, JSON's `Ub_Inline_Write` (network write direction send) and hcomm
profiling's `Ub_Inline_Write` (TASK_UB_INLINE_WRITE, notify inline write) **refer to
different tasks**, and reverse-misaligned:

| Task (collection semantics) | db taskType | ProfTaskParamType (op name) | Pre-Switch JSON Type |
|---|---|---|---|
| UBDMA normal write WQE (network send, write direction) | MEM_CPY (3) | `TASK_UB` -> `Ub_Write_Or_Read` | `Ub_Inline_Write` <- name misalignment |
| UBDMA network reduce (write/read direction) | REDUCE (2) | `TASK_UB_REDUCE_INLINE` -> `Ub_Reduce_Inline` | `Ub_Inline_Write` / `Ub_Write_Or_Read` <- name misalignment |
| UBDMA inline write notify (`inlineEn==1`) | NOTIFY_RECORD (1) | `TASK_UB_INLINE_WRITE` -> `Ub_Inline_Write` | `Notify_Record` <- name misalignment |

**Implemented (Section 10.3 #18 wrap-up)** - trace/dump display name switched to `profInfo.profTaskType`'s
op name (`Task.task_name`, `graph_builder._add_task` uniform inject); three misalignments all eliminated:
1. Network data task (TASK_UB) write/read uniformly displays `Ub_Write_Or_Read`; **direction no longer encoded
   into type name**; expressed by trace args existing `direction` (write/read) and `is_write` field -
   latency model unchanged, still per port write/read direction bandwidth sharing billed separately (Section 5.5).
2. `Ub_Inline_Write` name **returned** to notify inline write task (db taskType=
   NOTIFY_RECORD + inlineEn==1), one-to-one with msprof timeline.
3. Network reduce displays `Ub_Reduce_Inline` (TASK_UB_REDUCE_INLINE), retaining reduce semantic distinguishable.
4. `Notify_Record`/`Notify_Wait`/`Reduce_Inline`/`Memcpy` four classes originally already consistent with op name;
   switch no change (`Memcpy` corresponds to TASK_SDMA).

**Implementation trade-offs (established decisions)**:
- v2 old BLOB (no profInfo, profTaskType=0xFF) displays `Unknown` - same metric as
  `link_type`/`transport_type`'s v2 behavior (strict semantics, does not do internal name fallback).
- CCU microcode subtasks (Section 11.3) no profInfo; display name falls back internal TaskType name - simulator
  semantic artifacts no reconciliation need; maintains status quo (network transfer unified `Ub_Write_Or_Read`; direction by
  `direction`/`is_write` expressed, Section 11.3.1).
- Internal `TaskType` enum retired as latency model classification (Section 11.4.5 #3); simulation/pairing/billing zero change.

#### 11.4.5 String Consistency Convention (Reconciliation Basis)

1. **Python-side display strings must be character-by-character consistent with C++ side three name function outputs**
   (`sim_profiling_defs.h` single source), including hcomm source's established spellings
   (`Ub_Write_Or_Read`, `Dpu_Notify_Record` etc.) - this is the basis for field-by-field reconciliation with msprof timeline /
   L1 detail.
   **Only exception**: `TASK_WRITE_REDUCE_WITH_NOTIFY`'s op name tool-side unified as corrected spelling
   `Reduce_With_Notify` (hcomm source both places misspelled `Reuce_With_Notify`:
   `profiling_common.h:42` and `dfx_profiling_handler_lite.cc:67`; msprof raw
   data displays misspelled version); when cross-system reconciliation encounters this string, needs to do
   `Reuce_With_Notify <-> Reduce_With_Notify` difference mapping.
2. trace args field-level switch (**all landed**, Section 10.3 #18 wrap-up):
   `transport_type` <- transportType name, `link_type` <- linkType name
   (`graph_builder._add_task` uniform inject; value range table see Sections 11.4.2/11.4.3); CommProtocol
   migrated to new field `protocol` (solution a, Section 11.1); `task_type` <- profTaskType op name
   (`Task.task_name`/`display_name`, Section 11.4.4 implementation record).
3. `TaskType` enum already retired as internal latency model classification (`task_name` bears display responsibility);
   trace event name (`traceEvents[].name`) and `args.task_type` both switched to op name;
   CCU microcode subtasks/CLI input no profInfo; `display_name` falls back internal name.

#### 11.4.6 DPU (host CPU RoCE) Task Collection and Merge

host DPU transfer (hcomm `HostCpuRoceChannel`, pod networking cross-node path)'s ibv hijack tasks
(`ibv_dpu_stub.cc`) previously not labeled profInfo (displays `Unknown`), WriteWithNotify split to two
tasks, peer resolution loops back to local. This section is that path's alignment implementation record.

**profInfo labeling** (aligned with hcomm `host_cpu_roce_channel.cc` DFX metric; link/transport
always ROCE/ROCE):

| ibv Operation | db taskType | profTaskType (op name) |
|---|---|---|
| Pure imm notify (len=0 WRITE_WITH_IMM / hybrid) | NOTIFY_RECORD | `TASK_DPU_INLINE_WRITE` -> `Dpu_Notify_Record` |
| Write+imm (len>0 WRITE_WITH_IMM) | **WRITE_WITH_NOTIFY (7) single task** | `TASK_DPU_WRITE_WITH_NOTIFY` -> `Dpu_Write_With_Notify` |
| Pure RDMA write/read (no imm) | MEM_CPY | `TASK_RDMA` -> `RDMASend` (aligned with hcomm `UbMemTransport::Write/Read` precedent) |
| poll_cq pairing | NOTIFY_WAIT | `TASK_DPU_NOTIFY_WAIT` -> `Dpu_Notify_Wait` |

**WriteWithNotify merged single task**: whole operation only produces one taskType=7 task
(`WriteWithNotifyTaskData`: src/dst rank, offsets, len, notifyId, protocol),
aligned with hcomm whole operation only reports one `Dpu_Write_With_Notify` metric; `DpuPendingNotify`
pairing mechanism unchanged. Simulation side `_map_write_with_notify` takes port write direction bandwidth sharing
(local RoCE eid -> ranktable `d2h` port; when `d2h` has no configured port bandwidth, falls back
`default_port_bandwidth`).

**Real peer resolution (phantom echo issue, socket timing binding)**: estimate mode socket layer constructs
phantom peer (local Send mirrored replay as Recv, `hccp_ra_socket_stub.cc`); QP handshake returned
"remote gid" is actually local itself; `RaTypicalQpModify`'s gid->IP resolution returns to local. Fix
(zero-byte read, only relies on API call timing):
- `RaGetSockets` by `SocketInfoT.remoteIp` (hcomm link config; client/server roles
  both carry) resolves real peer EndPoint into FdInfo (peerDeviceId/peerEid/localEid);
- `RaGetQpAttr` anchor (hccp_stub, only executes in estimate branch) records QP pending binding
  (`DpuRecordPendingBindQp`, `bindAttempted` prevents Modify duplicate query attr then mis-bind);
- Same thread's next `RaSocketSend` (i.e., hcomm ExchangeData's attr pack -> Send timing) via
  `BindDpuPendingQps` binds QP to that socket real peer: `DpuQpInfo` adds
  `peerRankId`/`lcEid` (peer rank by FdInfo's peerEid queries `RankEidMap`; estimate
  communicator table only has local rank; lcEid for WRITE_WITH_NOTIFY port billing).
`RaTypicalQpModify` maintains `remoteDeviceId`'s loopback plane semantics
(`DpuPendingNotify` pairing depends; under estimate = local); task metadata all takes binding fields.

**NotifyWait precise synthesis (hcomm primitive interception)**: correct semantics is "only peer post-inserted pending
generates NotifyWait for me"; under phantom peer does not run, that notification has no real source; and hcomm
`NotifyWait` does imm hard validation on poll result (`wc.imm_data == dpuNotifyId`; on mismatch immediately
errors). Implementation (`hcomm_api_stub.cc` intercepts 5 hcomm northbound primitives; real body pointer static
one-time dlsym resolution; **normal mode first line short-circuit pure forward, zero overhead**):
- post-class primitives (`HcommChannelNotifyRecordOnThread` / `HcommWriteWithNotify(+-Nbi)
  OnThread`, input carries `remoteNotifyIdx`) -> `InsertDpuPendingNotifyEstimate`
  records `channelSlots[channel][idx] = (imm, notifyId)` slot mapping - under echo
  `remoteDpuNotifyIds_ == localDpuNotifyIds_`; post side idx->imm mapping is wait side
  expected mapping (same idx same slot value);
- wait-class primitives (`HcommChannelNotifyWaitOnThread(+-WithDefaultTimeout)`, input carries
  `localNotifyIdx`) -> `ConsumeDpuPendingNotifyAndEmit` precise path: expected
  imm = `channelSlots[channel][idx]` **precisely known**; directly synthesizes "virtual peer post" constructs
  wc return; imm must pass hcomm hard validation; NotifyWait node records local real wait event;
  only "notification arrival" this external fact is virtualized (isomorphic with socket echo virtualizing "peer response");
- Design constraint: estimate mode wait consumption **only takes precise path** - all wait entries must be intercepted
  (single-card phantom under FIFO loopback consumption unreliable; when send/receive asymmetric, imm mismatch/timeout, already removed with old mechanism);
  un-intercepted entry -> `pollCq::miss` **ERROR** (per qpn+idx combination deduplicated once, prevents busy-loop
  flooding log) -> hcomm timeout exposes, based on this in `hcomm_api_stub.cc` adds interception.

**taskSrc source marker (BLOB v5)**: `HcclTaskMetaData` adds `taskSrc`
(0=default, 1=DPU); DPU stub launched tasks set 1, for Python side precise split by source
(does not rely on transport==ROCE etc. link semantic basis; future sources like AICPU RoCE unaffected).

**DPU sequence merge pass (Section 5.1.2, `sqe_group_merge.py::merge_dpu_write_sequences`)**:
hcomm `WriteWithNotify` for len > maxMsgSize large messages splits to N-1 normal RDMA_WRITE +
1 tail WRITE_WITH_IMM (whole operation still only reports one); collection side per WR logged as N-1 MEM_CPY +
1 WRITE_WITH_NOTIFY. pass by DPU stream order scans src/dst dual-address contiguous MEM_CPY chain;
address-contiguous WRITE_WITH_NOTIFY tail wraps and merges to one (len full-chain summed; description follows tail task including
notifyId; occupies tail slot); chains without tail not merged (independent pure writes kept per-task `RDMASend`);
only `src_type==1` data tasks participate; other tasks retained as-is, do not cut chain.

> **Verification** (222 networking 8MB all_reduce): First round (20260921) 15 `Unknown` -> 0
> (`Dpu_Notify_Record` x 5 / `Dpu_Notify_Wait` x 9 / `Dpu_Write_With_Notify` x 4,
> link=ROCE), peer rank=2/4/6 (real cross-pod peer), `eid_unresolved=0`,
> WRITE_WITH_NOTIFY enters `d2h` port billing. Final (20260922, after primitive interception):
> estimate (-np 1) makespan 166.596us, primitive interception 16 times, precise synthesis 11 times, miss=0;
> normal (-np 8) checker All Success (primitive pure forward, interception log 0 lines); single-machine 8p
> (118) SQE path regression no change.

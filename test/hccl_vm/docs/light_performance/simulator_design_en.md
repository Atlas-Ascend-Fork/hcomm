# light_performance Plugin Design Document (Including Simulator Design)

> Document location: `docs/light_performance/simulator_design.md`
> Requirement source: `docs/light_performance/HCCLSIM/requirement_description`
> Reference implementation: `src/plugin/checker/src/framework/task_graph_generator_v3/` (graph generation principle),
> `docs/light_performance/HCCLSIM/hccl_sim/` (simulator prototype, **authoritative baseline for simulation module**)
> Alignment legacy items and matching checklist: Chapter 10
> db task / CCU instruction → JSON task type mapping table: Chapter 11

---

## 1. Requirement Background and Document Description

### 1.1 End-to-End Requirement

Integrate light_performance (lightweight performance simulation tool) as an independent plugin into the current project. Complete the following end-to-end workflow:

```
hccl-vm workflow execution ──generate──> task sequence / instruction db files
        │
        ▼
light_performance plugin reads db files ──generate──> task graph for performance simulation
        │
        ▼
Performance simulation module performs simulation based on the task graph
        │
        ▼
Output performance profiling json file
```

### 1.2 Sub-Requirement Decomposition and Chapter Mapping

| Sub-Requirement | Content | Status | Chapter |
|----------------|---------|--------|---------|
| Sub-Requirement 1 | Plugin engineering: CLI, compilation and installation, complete plugin architecture | CLI and installation design refined (@estimate one-shot execution channel) | Chapter 3 |
| Sub-Requirement 2 | Integration with hccl-vm: db file reading, task graph generation | **Detailed design in this iteration** | Chapters 4, 5 |
| Sub-Requirement 3 | CCU mode task graph generation (SQE queue + microcode instruction sequence → task graph) | **Detailed design in this iteration** | Chapter 6 |
| Sub-Requirement 4 | Performance simulation module (task graph parsing, performance simulation) | **Aligned and incorporated per HCCLSIM baseline** | Chapter 7 |
| Sub-Requirement 5 | Performance profiling json output (performance metrics, task graph) | High-level design | Chapter 8 |

> Chapter 10 records **field gaps, type mismatches, and other conflict points** in this alignment (differences between the db graph generation path and the HCCLSIM simulation model) for future matching reference. Section 11.4 in Chapter 11 provides the mapping and switching target for v3 collected profInfo (`HcclTaskProfInfo` three fields) versus JSON display types.

### 1.3 Delivery Scope of This Iteration

This requirement focuses only on **db file reading, parsing, and graph generation module**:

- **Input**: db files under hccl_vm_install/data directory (format and data reference `hccl_vm_install/data/hccl_vm_data.db`).
- **Output**: Task graph `DependencyGraph` and `SimConfig` required by the performance module (hccl_sim).
- **Reference code**:
  - Basic graph generation logic: `src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_generator_v3.cc/h`;
  - CCU microcode instruction to task graph: `src/plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/`;
  - AIV mode graph generation: not considered for now.
- **Graph generation scope (core design premise)**: The db contains **single rank task data**. This plugin performs **single rank graph generation**. This is one of the main differences from checker (multi-rank joint graph generation). **All cross-rank task edges (post-wait relationships) are ignored**: only post-wait edges within a single rank are considered. Cross-rank edges are not considered (see Sections 5.1, 5.4, 6.6).
- **Relationship with checker**: The output structure of the light_performance graph generation module is not identical to checker, but the basic graph generation and instruction conversion principles are the same. The former does not need the subsequent memory conflict validation / semantic validation operations of checker, so appropriate simplification is possible (difference comparison see Sections 5.8 and 6.7).

---

## 2. Plugin Overall Design

### 2.1 End-to-End Data Flow and Module Division

```
┌────────────────────────── light_performance Plugin ─────────────────────────┐
│                                                                            │
│  ┌───────────┐   ┌───────────────┐   ┌───────────┐   ┌────────────────┐  │
│  │ db_reader  │──>│ graph_builder │──>│ simulator │──>│ profiling      │  │
│  │ (Ch. 4)    │   │ (Ch. 5, 6)    │   │ (Ch. 7)   │   │ (Ch. 8)        │  │
│  └───────────┘   └───────────────┘   └───────────┘   └────────────────┘  │
│   sqlite reading     SQE→Task mapping   simpy discrete    json + trace   │
│   BLOB decoding      notify pairing      event                              │
│   operator grouping  CCU microcode       port bandwidth sharing            │
│                      expansion             HBM dynamic sharing             │
│                                          eid→port resolution               │
│  Input: data/hccl_vm_data.db (+ data/hccl_sim.db, ranktable.json)         │
│  Output: (SimConfig, DependencyGraph, TopoResolver) → makespan →          │
│          perf_profiling.json (+ graph dump: DOT/JSON/HTML)                 │
└────────────────────────────────────────────────────────────────────────────┘
```

Data contracts between modules:

```
db_reader     ──OpExecutionGroup(intermediate data model, Section 4.5)──────>  graph_builder
graph_builder ──(SimConfig, DependencyGraph)─────────────────────────────>  simulator   [output of this requirement]
simulator     ──result dict(tasks with start/end + makespan + ports)─────>  profiling/report
```

### 2.2 Plugin Directory Structure (Source Location: `src/plugin/light_performance`)

```
src/plugin/light_performance/
├── __init__.py
├── __main__.py             # Standalone CLI entry (Section 3.3)
├── main.py                 # @estimate plugin one-shot execution entry (Sections 3.1/3.2)
├── manifest.json           # Plugin manifest (name=estimate; installed with package)
├── pipeline.py             # Shared pipeline: db → graph → simulation → perf_profiling.json
│                           #   (shared by run_perf_sim.py and main.py, Section 3.1)
├── db_reader.py            # Sub-Requirement 2: db reading and BLOB decoding (Chapter 4)
├── sqe_group_merge.py      # SQE batch merge preprocessing: merge UDMA data tasks within same (stream, sqeGroupId) batch into 1 (Section 5.1.1)
├── graph_builder.py        # Sub-Requirement 2: SQE → DependencyGraph (Chapter 5)
├── graph_dumper.py         # Task graph dump: DOT + JSON + HTML
├── ccu_graph_builder.py    # Sub-Requirement 3: CCU microcode → task graph (Chapter 6)
├── register_model.py       # CCU register/CKE execution model (Section 6.4)
├── sim_config.json         # Default latency/bandwidth configuration (Section 7.2, units us/GBps; installed with package)
├── bw_profile.json         # Default bandwidth utilization configuration (installed with package)
└── simulator/              # Sub-Requirement 4: simulation module (migrated from HCCLSIM, Chapter 7)
    ├── config.py           #   SimConfig (JSON units us/GBps)
    ├── task_model.py       #   Task / TaskType / TRANSFER_TYPES / INVALID
    ├── dependency_graph.py #   DependencyGraph (int stream id)
    ├── task_timing.py      #   Task latency formulas (single-point maintenance)
    ├── bandwidth_profile.py#   Bandwidth utilization table
    ├── topo_resolver.py    #   eid→ports topology resolution (ranktable.json)
    ├── input_builder.py    #   JSON task file construction + load_config
    ├── simulator.py        #   HcclSimulator / NetworkPort / _BandwidthManager
    ├── report.py           #   Text report + Chrome trace
    └── __main__.py         #   hccl_sim compatible CLI (independent of db path)
```

> The `simulator/` sub-package is migrated from `docs/light_performance/HCCLSIM/hccl_sim/`. The interface uses HCCLSIM as the baseline (Chapter 7). The `dump_graph.py` at the repository root is a one-click dump entry (db → DOT/JSON/HTML, optionally with simulation timestamps).
> `manifest.json` and `main.py` (one-shot execution entry, Chapter 3) are placed directly in this package directory. During installation, the entire directory becomes the `plugin/light_performance` plugin directory (installation layout see Section 3.1, CMake rules see Section 3.4).

### 2.3 Relationship with Existing Plugins

| Plugin | Responsibility | Relationship with light_performance |
|--------|---------------|-------------------------------------|
| checker | Communication correctness validation (deadlock/memory conflict/semantics) | Reuses its graph generation principle; main difference: checker performs **multi-rank joint graph generation**, light_performance performs **single rank graph generation** (cross-rank pairing edges are not built, remote synchronization tasks are preserved with fixed remote latency approximation), and no validation post-processing is performed |
| runner | Virtual runtime (task-by-task simulated execution of data plane) | Also consumes opTask/ccuInstr; light_performance reads only and does not write, does not advance isDone |
| insight | Visualization | The profiling json / chrome trace from light_performance can be consumed by it (future) |

> **Plugin tag versus directory/module name**: This plugin installs with tag `estimate` (the host `@estimate` command channel and mode gate are hardcoded by tag, see Section 3.1), while the plugin directory and Python package name are both `light_performance` (`plugin/light_performance`). The tag corresponds to the "host routing identity", and the directory/package name corresponds to the "code organization identity".

---

## 3. Plugin Engineering Design (Sub-Requirement 1)

### 3.1 Plugin Identifier and manifest.json

**The plugin tag is fixed as `estimate`** (not the module name light_performance): the host CLI has hardcoded gates and path conventions for this plugin:

- The execution channel for `hccl-vm plugin run @estimate` is in `src/cmd/cmd_base_utils.cc:599-680` (`RunUserPlugin`) with special handling for `estimate`: only available in estimate mode (`sim::IsEstimateMode()`), the plugin directory is fixed as `<installRoot>/plugin/light_performance` (tag decoupled from directory name). The host reads its `manifest.json` and synchronously executes the `entry` command (see Section 3.2);
- Mode mutual exclusion: `@checker` is unavailable in estimate mode (`cmd_base_utils.cc:618-624`), `@estimate` is unavailable in non-estimate mode. Correctness validation and performance simulation belong to the concluding actions of two different run modes;
- The plugin scan/resident model (`src/plugin/hccl_plugin_manager.cc:94-169`, `src/plugin/hccl_plugin.cc`) matches the tag by the `name` field in manifest (scans `<installRoot>/plugin`, depth ≤ 2).

manifest.json (source and installation are both within the package directory: `src/plugin/light_performance/manifest.json` → `plugin/light_performance/manifest.json`):

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

- `name`: Plugin tag. The host routes by it (`hccl-vm plugin run @estimate`).
- `entry`: Startup command within the plugin directory. @estimate uses the one-shot synchronous execution channel (Section 3.2). The cwd during execution is the plugin directory. Relative paths are resolved relative to the plugin directory.
- `setting`: Plugin-private configuration key-value pairs. main.py reads them directly (following the approach of checker `SettingManager` for self-managed configuration. The Python side parses json directly). Path-type configurations are resolved relative to the plugin directory (the package directory): `db` defaults to `<installRoot>/data/hccl_vm_data.db` (same source as checker `Loader::LoadOpTaskFile`); `sim_config`/`bw_profile` are the latency/bandwidth and utilization configurations in the format of Section 7.2, defaulting to `sim_config.json`/`bw_profile.json` in the same directory as manifest and installed with the package; `ranktable` defaults to `ranktable.json` in the same directory as the db; `device`/`iter` correspond to CLI `--device`/`--iter`; `out_dir` is the output root directory (subdirectories are created under it by run timestamp); `per_op_mode` currently only supports per-operator mode (when false, a warning is issued and per-operator execution is used); `enable_trace`/`dump_graph`/`max_trace_events` control additional artifacts.

Source and installation layout (plugin directory equals package directory, no nesting):

```
Source = Installation artifact structure (src/plugin/light_performance → hccl_vm_install/plugin/light_performance):
plugin(light_performance)/            # Plugin directory = light_performance package directory
├── manifest.json                     # Plugin manifest (name=estimate)
├── main.py                           # @estimate one-shot execution entry (reuses pipeline.py)
├── __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
├── sim_config.json                   # Default latency/bandwidth (Section 7.2)
├── bw_profile.json                   # Bandwidth utilization
├── simulator/                        # Simulation sub-package (Chapter 7)
└── data/                             # Runtime output (perf_profiling.json etc., Chapter 8;
                                      #   only produced in installation tree, source tree does not have this directory)
```

> main.py is located within the package directory but executes as a script (`python3 main.py`). It adds the **parent directory** of the package (`plugin/`, source tree is `src/plugin/`) to `sys.path` before `from light_performance.pipeline import ...`. Relative imports within the package are not affected. The source tree and installation tree layouts are consistent.
> The `run_perf_sim.py` at the repository root (manual entry) shares the same pipeline implementation with `main.py`: main.py adds "installation tree path conventions + manifest setting reading" on top of it.
> Source tree debugging still uses run_perf_sim.py (Section 3.3). The installation tree triggers main.py through `hccl-vm plugin run @estimate`.

### 3.2 Call Chain and Process Model (@estimate One-Shot Execution)

The host side requires only one path adaptation. The one-shot execution channel for `@estimate` is already implemented (`cmd_base_utils.cc:599-680`). This iteration changes the plugin directory in the special handling branch from `plugin/estimate` to `plugin/light_performance` (tag remains `estimate`, directory equals package directory):

```
hccl-vm plugin run @estimate
    │
    ▼
RunUserPlugin("@estimate")                                  # subcmd_plugin.cc:55-57 routing
    ├─ Gate: must be in estimate mode, otherwise error exit  # cmd_base_utils.cc:611-617
    ├─ Read <installRoot>/plugin/light_performance/manifest.json  # Missing file/entry field → error
    └─ system("cd <installRoot>/plugin/light_performance && python3 main.py")   # Synchronous blocking execution
       └─ Non-zero exit code → HCCL_VM_ERROR, command returns failure overall
```

**Client wait strategy for long-running commands** (`StartHostClient`/`ServerListen` in `cmd_base_utils.cc`):
Within a session, `hccl-vm` sends to the host through POSIX message queue. **The host responds only after `ParseCommand` returns (the command finishes execution)**; CCU MS mode microcode expansion can produce 10^5 level tasks (measured 192,005 tasks, software simulation approximately 54s), far exceeding the 5s response limit for ordinary commands. For this:

- After the client identifies `plugin run @estimate`, the client adopts a long wait limit (default 30 minutes, environment variable `HVM_CLIENT_RESP_TIMEOUT_MS` can override). During segmented waiting, the client prints "command still executing on host" to the terminal every 30s (first at 5s). On timeout, the client prints a clear warning; other commands maintain the 5s limit.
- The response carries the original command string (`ACK:<cmd>`). Before sending a command, the client drains the response queue and only accepts responses that strictly match the current command. This avoids misreading late responses from previously timed-out commands and returning prematurely (especially important during interrupt-retry of the same command).
- The host prints `[HVM] @estimate performance simulation started/ended` markers to the terminal at simulation start and end (stdout, with report output). This mechanism ensures **the prompt returns only after simulation truly ends**, eliminating the misconception that "simulation has already ended".

main.py (plugin side) execution flow:

```
RunEstimate():
    setting   ← Parse the setting section of manifest.json (paths resolved relative to plugin directory)
    db        ← setting.db (default <installRoot>/data/hccl_vm_data.db, derived through __file__ as ../../data)
    config    ← setting.sim_config / bw_profile (use built-in defaults when missing, Section 7.2)
    ranktable ← ranktable.json in same directory as db (_find_ranktable default rule; consistent with host InitHvmCommEnv
                writing data/ranktable.json, cmd_base_utils.cc:437-450, no configuration needed)
    groups    ← db_reader.load_op_executions(db)            # Chapter 4
    topo      ← TopoResolver(ranktable)                     # Section 7.6
    for group in groups:                                    # Per-operator (Section 5.1)
        config, dep_graph = graph_builder.build(group)      # Chapters 5, 6
        result = HcclSimulator(config, dep_graph, topo).run()   # Chapter 7
    profiling.dump(<plugin>/light_performance/data/<timestamp>/perf_profiling.json)   # Chapter 8
```

Comparison and selection basis versus checker resident process model (stdin single-line JSON command loop):

| Dimension | checker (resident model) | estimate (one-shot model) |
|-----------|------------------------|---------------------------|
| Launch method | fork+pipe resident during `plugin install` (`hccl_plugin.cc:74-156`) | `cd && entry` synchronous execution during `plugin run` (`cmd_base_utils.cc:655-660`) |
| Trigger protocol | stdin JSON: `{"action":"status","payload":{"status":"finish","data_id":...}}` | No protocol: entry starts immediately, input paths use conventions + manifest setting |
| data_id | Carried in message (generated by host `DumpData`) | Not passed, output directory uses run timestamp (Section 10.3 #13) |
| Lifecycle | Reclaimed on stop command/pipe EOF/host exit | Process exits naturally, exit code is the execution result |
| Adaptation reason | Validation needs host trigger at workflow end, may run multiple times | Simulation is offline batch processing: consumes already-written db, runs once and exits |

> **Evolution direction** (not implemented in this iteration): If "host automatically triggers simulation at workflow end" (resident model) is needed later, main.py can add a stdin command loop to reuse the same pipeline. The host message channel (`SendMessageToPlugin`) and resident launch (`RegisterPlugin`) are already ready. At that time, only switch @estimate in `RunUserPlugin` from the special handling branch back to the general message path.

Data contracts (fixed conventions within the installation tree, all aligned with existing mechanisms):

| Path | Source | Description |
|------|--------|-------------|
| `data/hccl_vm_data.db` | Written during host workflow execution | Main simulation input; estimate mode produces **single rank task data** (phantom remote end), naturally aligned with single rank graph generation premise in Sections 4.6/5.1 |
| `data/ranktable.json` | Written by `InitHvmCommEnv` | eid→port resolution (Section 7.6); `_find_ranktable` defaults to same directory as db, no explicit configuration needed |
| `plugin/light_performance/sim_config.json`, `bw_profile.json` | Installed with package directory by CMake (Section 3.4) | Section 7.2 format (units us/GBps) |
| `plugin/light_performance/data/<timestamp>/` | Created at runtime by main.py | Output directory. **Must not be placed under `data/`**: the host executes `remove_all("data")` at each start (`src/main.cc:32`). Contents of the data directory are not preserved across sessions |

### 3.3 Command Line Design

Host integration mode (`plugin run` serves as the simulation entry, replacing manual execution of run_perf_sim.py):

```bash
# ① Start environment in estimate mode (choose one)
hccl-vm start <topo> --estimate             # Interactive mode, PS1 displays (estimate)$>
hccl-vm run <topo> --estimate <sample command>     # One-shot mode
# (Within an already started session, switch with hccl-vm mode estimate, subcmd_mode.cc)

# ② Run workload → produce data/hccl_vm_data.db (single rank task list)

# ③ One-click performance simulation (only available in estimate mode, Section 3.2 gate)
hccl-vm plugin run @estimate
# Artifacts: plugin/light_performance/data/<timestamp>/perf_profiling.json (+ trace / graph dump)
```

- The one-shot execution channel does not pass through command line arguments; simulation parameters (target device, opIter filter, trace/dump switches, etc.) are configured through the manifest `setting` section (Section 3.1). After modifying the configuration, re-run `plugin run @estimate` for changes to take effect.
- `plugin install @estimate` is also executable (uses the `RegisterPlugin` resident launch channel). main.py finishing and exiting after one run is acceptable degraded behavior (Section 10.3 #14); regular use does not require install.

Standalone debug mode (does not depend on host, convenient for offline db analysis; already implemented):

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
|-----------|-------------|
| `--db` | hccl_vm_data.db path (required) |
| `--config` | Latency/bandwidth configuration (Section 7.2 format, units us/GBps; defaults to built-in defaults) |
| `--bw-profile` | Bandwidth utilization configuration (default utilization is 100%) |
| `--ranktable` | Topology file for eid → physical port resolution (defaults to `ranktable.json` in same directory as db; when network tasks exist and no topology is available, **skip simulation and only dump**) |
| `--device` | Target device (deviceId) for single rank graph generation; defaults to first device in opDetails, only needed when db contains multi-device data |
| `--iter` | Simulate only the operator with specified opIter (default all) |
| `--trace` | Export Chrome tracing JSON (for multiple operators, split files by `_<commId>_<opIter>`) |
| `--dump-dir` | Task graph dump directory (.dot/.json/.html per operator, includes simulation timestamps) |
| `-v` / `-s` | Task-level detail / summary only |

### 3.4 Compilation and Installation Design (hccl_vm_install/plugin/light_performance)

Add install rules in `src/plugin/CMakeLists.txt` (following the directory installation precedent of insight; Python plugins do not need compilation targets. The plugin directory equals the package directory. manifest.json/main.py/default configurations are all within the package directory. A single DIRECTORY rule can carry all content):

```cmake
# estimate performance simulation plugin: plugin directory equals light_performance package directory, install entire directory
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/light_performance"
    DESTINATION "plugin"
    FILE_PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
    PATTERN "__pycache__" EXCLUDE
    PATTERN "data" EXCLUDE)      # Runtime output directory not included in installation package
```

Comparison with existing plugin installation methods:

| Plugin | Installation Method | Artifact |
|--------|-------------------|----------|
| checker | `install(TARGETS checker DESTINATION plugin/checker)` + `install(FILES manifest.json)` (`src/plugin/checker/CMakeLists.txt:67-82`) | Compilation artifact + manifest |
| insight | `install(DIRECTORY insight DESTINATION plugin/visualization PATTERN frontend EXCLUDE)` (`src/plugin/CMakeLists.txt:26-39`) | Python directory + manifest |
| **estimate (this iteration)** | `install(DIRECTORY light_performance DESTINATION plugin)` (single rule; manifest/main.py/default configs all installed with package directory) | `plugin/light_performance/`: Python package + manifest (plugin directory and package directory unified, no nesting) |

Build and verify: After `bash build.sh` (internally `cmake && make && make install`, installation prefix defaults to `hccl_vm_install/` within the repository, `CMakeLists.txt:89-92`), check the installation artifacts:

```
hccl_vm_install/plugin/light_performance/
├── manifest.json
├── main.py
├── __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
├── sim_config.json / bw_profile.json
└── simulator/ ...
```

- Packaging path: `build.sh --pkg` packages the entire `hccl_vm_install` into a tar archive. The estimate plugin is distributed with it.
- Dependencies: python3 (≥3.8, including sqlite3 standard library) and **simpy** (discrete event simulation engine, current environment 4.1.2, imported by `simulator/simulator.py`); deployment environments need pre-installation (the insight plugin also depends on python3, convention is consistent), see Section 10.3 #16.

---

## 4. db File Reading and Parsing (Sub-Requirement 2 - First Half, Focus of This Iteration)

### 4.1 db Location and Opening

- The path is fixed as `<installRoot>/data/hccl_vm_data.db` (consistent with checker `Loader::LoadOpTaskFile`).
- Open read-only: Python `sqlite3` standard library, `mode=ro` URI, to avoid write conflicts with running hccl-vm:
  ```python
  conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
  ```
- Auxiliary library `hccl_sim.db` (same directory): only used to resolve `opDetails.commId` to communicator identity (`Communicator` table, `comm_id[128] + comm_hash` unique identifier), optional.

### 4.2 Table List and Association Model

hccl_vm_data.db tables are created by `OpDbOps::InitSchema()` (`src/runnerdb/modeldb/db_hccl_op_db_ops.cc:76-184`). The tables consumed by light_performance are as follows:

| Table | Primary Key/Association | Content | Usage |
|-------|------------------------|---------|-------|
| `opDetails` | `id`; `pid`→opTask table name | One row per operator per rank: pid/deviceId/rankId/commId/opIter/syncIter/streamId(main stream)/root/rankSize/srcRank/dstRank + `opDetail` BLOB (operator parameters) + `algorithmName` (algorithm selection result, DFX hijack/log dual-path interception and backfill, see note below) | Operator grouping, main stream identification, operator parameters, algorithm selection results |
| `opMemInfo` | `opDetailId` → opDetails.id | Addresses and sizes of input/output/ccl memory segments | Direction determination (Section 5.5), data volume validation |
| `opTask_P_<pid>` | `opDetailId` → opDetails.id | SQE task queue: deviceId/streamId/taskType/isDone + `opTaskMeta` BLOB | **Main graph generation input** |
| `ccuChannels` | `(deviceId, dieId, channelId)` | CCU channels: src/dst die/device/rank + leid/reid(16B EID) + protocol + jettyNum + jettyId array | CCU channel → remote end/jetty mapping (Chapter 6) |
| `ccuInstrRes` | `(deviceId, dieId)` | Microcode instruction space per die: instrCount + instrSpace BLOB (32K×32B) | CCU microcode source (Chapter 6) |
| `ccuInstr` | `ccuInstrResId` → ccuInstrRes.id | One row per microcode dispatch (startId/instrInfoSize) | Only used for sensing "new microcode dispatched", decoding does not depend on it |
| `syncRecords` | — | Host-side stream synchronization records | Not used in this iteration |
| `JettyMaps` | `opDetailId` | eid mapping recorded per operator | Backup (not used when ccuChannels covers it) |

> Table structure definitions see `include/runnerdb/sim_op_db_types.h`; writer semantics see `src/runnerdb/tracedb/db_sim_op_db_ops.cc`.

> **Source of algorithmName column (algorithm selection result interception)**: Dual-path interception, both use last-wins to backfill `sim::UpdateOpDetailAlgName`, serving as mutual backup and writing the same value.
>
> 1. **DFX hijack stub (primary path)**: After algorithm selection completes, the workload calls `HcclDfxRegOpInfoByCommId(commName, &HcclDfxOpInfoCompat)` per operator (hccl repository op_common.cc). Its `algTag` carries the selected algorithm name (format `{opType_commName}_{algName}_{host|device}[_{dataType}...]`, graph mode `Graph_{algName}_{...}`). Level=2 mode (default) only mounts level0+level2. The level1 dfx_stub is not in the process. `src/proxy/level2/hcomm_api_stub.cc` preempts resolution with level2 strong symbols (preload order precedes the real libhcomm.so loaded by the workload through the python plugin chain). After parsing the algorithm name from algTag and backfilling, it **forwards to the real implementation through `DlsymRealWithFallback` (RTLD_NEXT + libhcomm.so SONAME fallback + dladdr self-call prevention), passes through the return value, and does not change call semantics**. The structure layout is replicated per hccl_res_dl.h (`static_assert` locks algTag offset 68). Content validation (commId prefix + host/device suffix) filters registrations without algorithm names such as op-begin, and guards against workload header drift. In Level=1 mode, this stub is shadowed by the level1 dfx_stub (behavior same as old version), and interception falls back to the log path.
> 2. **Log interception (backup path)**: The tool is the implementer of the workload log underlying interface (workload `DlogRecord` weak symbol resolves to `src/proxy/level2/log_stub.cc`). The level0 stub `RecordOpDbInfo` first inserts the opDetails row (sets the runnerdb global `g_currOpDetailId`) then calls the real workload. The INFO-level algorithm selection log output by the selector within the workload (`[HcclGetAlgRes] algName=` / `SelectMinCost: selected algName=` / old selector `the selected algo type is ` three types of markers, last match wins for same operator) is matched and extracted through the `DlogPrintStub` entry. The interception executes before messages enter the tool log output (immune to `log_config.yaml` level). The proxy layer also provides `acllogCheckDebugLevel` always-pass stub (immune to workload-side log level threshold).
>
> Old databases automatically get `ALTER TABLE` column addition by `OpDbOps::InitSchema()`; plugin reading falls back to empty values for old databases without the column.

**Operator grouping model**: One operator instance = rankSize opDetails rows under the same `(commId, opIter)` (one per rank). Each row has one opMemInfo and a set of tasks (`WHERE opDetailId=?`, table name derived from the `pid` column of that row as `opTask_P_<pid>`, reference checker `QueryCompositeOpDetailByOpIter`).

### 4.3 opTask Dynamic Tables and Multi-pid Handling

- Each task-writing process creates its own table `"opTask_P_" + getpid()`; AICPU device-side processes use `getppid()` to share the same table as the host.
- **Discover all task tables** (enumeration is required, multi-pid is the norm; early sample db already has `opTask_P_477776`/`opTask_P_477780` two tables, current sample has a single table):
  ```sql
  SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'opTask_P_%'
  ```
- Consumption order: within the same operator, by `id` ascending order (= SQE dispatch order, this is the basis for notify first-come-first-served pairing).

### 4.4 BLOB Decoding Specification (pack(1), Little-Endian)

All BLOBs are raw C structure memcpy (no serialization framework). The Python side uses the `struct` module for decoding. The following layouts have all been verified with `hccl_vm_install/data/hccl_vm_data.db` (v5 collection onward sample opTaskMeta length 190B, historical samples 189B/188B).

#### 4.4.1 opTaskMeta (190 Bytes)

The structure definition shares a single source with the writer: `include/sim_profiling_defs.h` (profiling reconciliation information single-source definition) and `src/plugin/checker/header/internal/task_meta_defs.h` (HcclTaskMetaData).

v5 collection structure (compared to v4 189B version): added `taskSrc` (1 byte) after `sqeGroupId`. This is the task source marker (`HcclTaskSrc`: 0=default SQE/aclrt native path, 1=DPU meaning host CPU RoCE `ibv_dpu_stub` collection path). It enables the performance simulation DPU sequence merge pass to split precisely by source (Section 5.1.2). taskData start offset changes from 69 to 70.

v4 collection structure (compared to v3 188B version): added `sqeGroupId` (1 byte) after `jettyId`. This is the UDMA SQE batch group marker (all WQEs within one doorbell share the same group). It enables the performance simulation entry to merge by batch (Section 5.1.1). taskData start offset changes from 68 to 69. Group number semantics: **circular incrementing doorbell batch number 1..255** (wraps at 255, collection side `device_sqe_parse_stub.cc` maintains global `g_currSqeGroupId`, `% 255 + 1`). Batch numbers are unique within a 255 window. Doorbells from other operators are not visible in the current operator task list but also consume batch numbers, so same stream and same number within the list must be from the same doorbell; **0 is reserved as "unmarked"** (HcclTaskMetaData construction default), the counter never takes 0. The merge entry treats `sqeGroupId <= 0` as not participating (Section 5.1.1).

v3 collection structure (compared to v2 185B version): added `HcclTaskProfInfo` sub-structure (3 consecutive bytes, checker/runner passes through as opaque) after `taskType`: `profTaskType` (aligned with hcomm TaskParamType value range 0~22) and `linkType`/`transportType` (aligned with hcomm DfxLinkType value range 0~9 / DfxTransportType value range 0~4, used for profiling data reconciliation). taskData start offset changes from 65 to 68.

> **Implementation status**: `db_reader.py` automatically identifies layout by BLOB length. `>= 190B` uses v5 (`<BBBBQIQQIBB16s16s`, taskSrc decoded into `HcclTaskMetaData.src_type`; this branch **must precede v4 determination**, otherwise 190B would hit the `>= 189` v4 branch first causing eid/taskData offset misalignment). `>= 189B` uses v4 (`<BBBBQIQQIB16s16s`, sqeGroupId decoded into `RawTask.sqe_group_id`, src_type=0). `188B` uses v3 (`<BBBBQIQQI16s16s`, sqe_group_id filled with -1, not participating in merge). `185~187B` uses old v2 (`<BQIQQI16s16s`, profInfo filled with 0xFF=UNKNOWN) for backward compatibility. `< 185B` is discarded with warning. profInfo consumption (`task_type` display name + link/transport) **is implemented** (#18, Section 11.4).

v2 collection structure (compared to earlier 169B version): added `lcEid[16]` (explicit local EID) in header. taskData members previously had `srcRankId/dstRankId` fields added.

> **rank field rollback: srcRankId/dstRankId are deprecated (current convention)**: The `srcRankId/dstRankId` fields previously added to taskData members have been rolled back and deleted. All now use `srcDeviceId/dstDeviceId` uniformly. The collection side (estimate mode, `device_sqe_parse_stub.cc` etc.) writes the communicator rankId directly into the `srcDeviceId/dstDeviceId` fields, meaning **the deviceId field carries rankId** (remote rank is `0xFFFFFFFF` when unresolved). `db_reader.py` consumes directly as `src_rank_id/dst_rank_id` after unpacking (reads `srcDeviceId` as `srcRankId`). This is a hard switch: old BLOBs collected before the field rollback (dual rank/device field layout) are no longer supported.

```
Header 70B:  struct.unpack_from("<BBBBQIQQIBB16s16s", blob, 0)
┌────────┬──────┬──────────────┬────────┬───────────┬──────────────────────────┐
│ Offset │ Size │ Field        │ Type   │ Sentinel  │ Description              │
├────────┼──────┼──────────────┼────────┼───────────┼──────────────────────────┤
│ 0      │ 1    │ taskType     │ u8 enum│ —         │ Redundant with table column taskType │
│ 1~3    │ 3    │ profInfo     │ struct │ 0xFF      │ HcclTaskProfInfo:        │
│        │  1   │ .profTaskType│ u8 enum│ 0xFF      │ Aligned with hcomm TaskParamType │
│        │  1   │ .linkType    │ u8 enum│ 0xFF      │ Aligned with hcomm DfxLinkType │
│        │  1   │ .transportType│u8 enum│ 0xFF      │ Aligned with hcomm DfxTransportType │
│ 4      │ 8    │ commId       │ u64    │ —         │ Communicator ID          │
│ 12     │ 4    │ rankId       │ u32    │ 0xFFFFFFFF│ Backfilled through device→rank when default │
│ 16     │ 8    │ deviceId     │ u64    │ —         │ Physical device ID       │
│ 24     │ 8    │ streamId     │ u64    │ —         │ Stream ID               │
│ 32     │ 4    │ jettyId      │ u32    │ 0xFFFFFFFF│ Invalid = no network    │
│ 36     │ 1    │ sqeGroupId   │ u8     │ 0         │ Doorbell batch number 1..255 circular incrementing (v4 new, Section 5.1.1); 0 = unmarked │
│ 37     │ 1    │ taskSrc      │ u8 enum│ 0         │ Task source (v5 new, Section 11.4.6): 0=default, 1=DPU(host ibv) │
│ 38     │ 16   │ rmEid[16]    │ bytes  │ All 0     │ Remote EID              │
│ 54     │ 16   │ lcEid[16]    │ bytes  │ All 0     │ Local EID (v2 new)      │
│ 70     │ 120  │ taskData     │ union  │ Interpreted by taskType, see below  │
└────────┴──────┴──────────────┴────────┴───────────┴──────────────────────────┘
```

taskData union internal layout (relative offset 69):

| taskType | Member | Format String | Fields (in order) |
|----------|--------|---------------|-------------------|
| 0/1 (notify) | `NotifyTask` | `<IQIBB` | srcDeviceId(carries rankId), notifyId, dstDeviceId(carries rankId), notifyCount, protocol |
| 3 (transMem) | `TransMemTask` | `<IQIQQB` | srcDeviceId(carries rankId), srcOffset, dstDeviceId(carries rankId), dstOffset, len, protocol |
| 2 (reduce) | `ReduceTask` | `<IQIQQBBBB` | srcDeviceId(carries rankId), srcOffset, dstDeviceId(carries rankId), dstOffset, dataCount(actually byte size), dataType, reduceOp, protocol, resv |
| 4 (ccu) | `CcuTask` | `<BBHHHII` + `<13Q` | dieId, missionId, timeout, instStartId, instCnt, key, argSize, args[13] |
| 5 (aiv) | `AivTask` | `<Q` | launchIdx |
| 6 (sync) | `SyncStreamTask` | `<Q` | syncIdx |
| 7 (writeWithNotify) | `WriteWithNotifyTask` | `<IQIQQQBBBBB` | srcDeviceId(carries rankId), srcOffset, dstDeviceId(carries rankId), dstOffset, len, notifyId, isReduce, dataType, reduceOp, protocol, notifyCount |

> Both `len` and `dataCount` are **byte sizes** (the reduce field name has historical reasons, actually used as size). `WriteWithNotifyTask.len` is also byte size; `isReduce=1` indicates the write payload is reduction data (`dataType`/`reduceOp` are valid in this case), mapped to a single `Write_With_Notify` task (Section 5.2).

#### 4.4.2 taskType Enumeration Full Table

`HccLTaskMetaType` (`include/sim_common_defs.h:184-193`):

| Value | Enumeration | light_performance Handling |
|-------|-------------|---------------------------|
| 0 | `NOTIFY_WAIT` | → local_wait / remote_wait (Section 5.2) |
| 1 | `NOTIFY_RECORD` | → local_record / remote_record |
| 2 | `REDUCE` | → memcpy / local_copy (reduction computation approximation, Section 5.7) |
| 3 | `MEM_CPY` | → memcpy / local_copy (determined by jettyId) |
| 4 | `CCU_GRAPH` | → CCU subgraph expansion (Chapter 6) |
| 5 | `AIV_GRAPH` | Not supported, skip with warning |
| 6 | `SYNC_STREAM` | → Skip, do not generate node (Section 5.6) |
| 7 | `WRITE_WITH_NOTIFY` | → Single task mapping `Write_With_Notify` (network write + attached notify, no splitting, Section 5.2) |

#### 4.4.3 opDetails.opDetail (28 Bytes)

`OpDetails` (`task_meta_defs.h:141-165`): `<HHH` header + union:

```
opType u16@0 | dataType u16@2 | reduceType u16@4
opV1: count u64@6 | dataType u16@14 | reduceOpType u32@16          (total 28B)
opV2: sendCount u64@6 | sendDataType u16@14 | recvCount u64@16
      | recvDataType u16@24 | extInfo u16@26                       (AllToallV type)
```

opType is `HcclCMDType` (ALLREDUCE=2, ALLGATHER=6, REDUCE_SCATTER=7, ALLTOALL=10 etc., complete enumeration see `src/plugin/checker/header/external/hccl_types.h:162-189`), used for operator labeling in profiling reports. `opExtInfo` BLOB (optional) stores V operator counts/displs, AllToallV send matrix, etc. This iteration only passes it through.

V1/V2 layouts are distinguished by opType (level0 stub writer-side convention, `hccl_op_stub.cc`): **V2 only for AllToAll(10)/AllToAllV(8)/AllToAllVC(9)** (`BuildOpDetailsV2`, sendCount/recvCount are the total elements sent/received by this rank); **AllGatherV(16)/ReduceScatterV(17) use V1 + opExtInfo** (count = element count on this rank side). When db_reader decodes, it selects the layout by opType. V2 operators decode send and receive side element counts and types (`send_count/recv_count/send_data_type/recv_data_type`). `data_size/data_type` follow the send side to maintain old consumer compatibility. `reduce_op` is set to 0.

#### 4.4.4 ccuChannels BLOB

| Field | Format | Description |
|-------|--------|-------------|
| `leid` / `reid` | Each 16B raw bytes | Source/destination endpoint EID (`URMA_EID_LEN=16`) |
| `jettyId` | `jettyNum × u32` little-endian | Continuous jetty number array (C structure reserves `u32[64]`; sample measured jettyNum=1, jettyId=(1024,)) |
| `protocol` | int column | CommProtocol: HCCS=0/TCP=1/ROCE=2/UB_CTP=3/UB_TP=4/PCIE=5/SIO=6 |

Note: The `srcRankId/dstRankId` columns actually contain device_id (channel establishment precedes communicator, see writer `hccp_ccu_stub.cc:401-405` comment). Rank semantics need correction through opDetails.

#### 4.4.5 ccuInstrRes instrSpace

- `instrSpace` = **Complete microcode instruction space snapshot** for one die: `32K entries × 32B = 1 MiB` (`CCU_INSTRUCTION_NUM=32*1024`, `CCU_INSTRUCTION_SIZE=32`, `include/sim_common_defs.h:35-36`).
- The i-th instruction is at byte offset `i*32`; `instrCount = max(historical value, startId+instrCnt)`, **holes are allowed**.
- When reading, only take the first `instrCount` entries; CCU task `instStartId/instCnt` are used directly as index ranges.
- The `ccuInstr` table is only for runtime awareness (runner polls `COUNT(ccuInstr)+COUNT(ccuInstrRes)` changes). light_performance decoding does not depend on this table.

### 4.5 Intermediate Data Model (db_reader Output)

```python
@dataclass
class OpExecutionGroup:          # One operator instance (single rank perspective: only target device task data)
    comm_id: int
    op_iter: int
    op_type: int                 # HcclCMDType
    rank_size: int
    main_stream_ids: dict[int, int]      # deviceId → main stream streamId (opDetails.streamId)
    tasks: list[RawTask]         # All SQEs for this operator, globally ordered
    mem_info: dict[int, MemInfo]         # deviceId → input/output/ccl ranges
    channels: dict[tuple, ChannelInfo]   # (srcDev, srcDie, channelId) → channel information
    instr_space: dict[tuple, list[bytes]]# (deviceId, dieId) → 32B instruction list
    algorithm_name: str          # Algorithm selection result (opDetails.algorithmName, DFX/log dual-path interception backfill; empty string for old databases)
    send_count / recv_count: int # Send/receive side element counts (V2 layout operators; same as count for V1)
    send_data_type / recv_data_type: int # Send/receive side data types (same as above)

@dataclass
class RawTask:
    global_id: int               # Global sequence across opTask_P_* tables (sorted by table pid value + in-table id)
    op_detail_id: int
    device_id: int
    stream_id: int
    meta: HcclTaskMetaData       # 4.4.1 decoding result (dataclass form)
```

`meta` contains v2 new fields: header `lc_eid` (explicit local EID, Section 5.5 local eid resolution uses this with priority); NotifyTaskData/TransMemTaskData both carry `src_rank_id/dst_rank_id` (under estimate collection convention, unpacked from rankId values carried in deviceId fields, see Section 4.4.1 rank field rollback).

### 4.6 Data Integrity Validation

For each `OpExecutionGroup`:

1. **Single rank filtering (design premise)**: The performance plugin only consumes **single rank task data** (sample db contains only device 1, this is design norm not data deficiency). If the db contains multi-device data, filter opDetails and their tasks by target deviceId (`--rank`, defaults to first device); `rankSize` is only passed through as metadata. It is not required that `len(opDetails rows) == rankSize`.
2. **notify remote end** (no backfill):
    - Estimate collection convention (`device_sqe_parse_stub.cc`, consistent for both task types): for notify, `srcDeviceId` (carries rankId) = record side, `dstDeviceId` (carries rankId) = wait side. Both are explicitly filled by the launch end; remote rank is `0xFFFFFFFF` when unresolved.
    - The early version `dstDeviceId == 0` backfill fallback has been deleted along with the rank field rollback (0 sentinel conflicts with valid rank 0, and the launch end always fills explicitly so backfill is unnecessary).
    - The notify carried by `WRITE_WITH_NOTIFY` (7) targets the **write remote end** (wait side is not in this db, single rank simulation premise). It does not participate in local pairing; its `dstDeviceId` (carries rankId) is the write destination.
    - Note: The runner/checker consumer side (`DecodeAndExpandTaskMeta`) splits 7 into [write, NOTIFY_RECORD] two sub-tasks at the decoding boundary; light_performance **does not split**. It uses single task mapping `Write_With_Notify` (Section 5.2) to ensure simulation tasks correspond one-to-one with db tasks.
3. **Decoding length validation**: `len(opTaskMeta) >= 185` (adapts to v4/v3/v2 layout by length, Section 4.4.1), `len(opDetail) >= 28`, `len(instrSpace) >= instrCount*32`. If not met, discard the row and issue a warning (checker does direct memcpy, this is more lenient).

### 4.7 Decoding Example (Measured hccl_vm_data.db)

Sample db (REDUCE_SCATTER, opType=7, dataType=4/FP32, count=8, rankSize=2, devType=6/950) measured decoding results:

- `opTaskMeta` length is consistently 188B (v3 collection sample; v4 collection onward is 189B, Section 4.4.1); taskType distribution: 4 (CCU_GRAPH) and 6 (SYNC_STREAM) alternating.
- opDetails contains only device 1 (rank 0) → single rank data (design norm); main stream is stream 2.
- Task sequence (per opIter main stream):

```
stream 2 (main stream):  CCU_GRAPH(die=1, mission=0, instStartId=14, instCnt=100, argSize=13)
                          SYNC_STREAM(syncIdx=n)          # Microcode dispatch + stream synchronization
```

- The args of CCU tasks match item-by-item with the data directory `sqe_info_device_1_die_1_mission_0_startId_14.txt` (args[0]=281475405602881, args[1]=281475405602945 ...); `args[0]/args[1]` are measured to be the inputAddr/outputAddr from opMemInfo.
- `ccuInstrRes` (dev 1, die 1) instrCount=114 = instStartId(14) + instCnt(100).
- In early 169B samples containing MEM_CPY/NOTIFY: MEM_CPY srcOffset/dstOffset can both fall within opMemInfo input/output/ccl ranges (`input ⊃ output`, ReduceScatter output is the first 1/rankSize segment of input); network copies with jettyId also show offsets as "local range addresses" (communicator memory symmetric layout, remote buffer and local VA are the same), therefore **direction determination cannot rely on addresses alone** (see Section 5.5). This conclusion also applies to v2 data.

#### 4.7.1 v2 AICPU Mode Sample (opTask_P_118800)

REDUCE_SCATTER, rankSize=8, count=32768 (FP32, output=128KB), 76 tasks, stream 2 (main stream)/6 (scheduling stream)/7..13 (one stream per remote end). Key differences from CCU mode sample:

- taskType distribution: NOTIFY_WAIT×30, NOTIFY_RECORD×30, MEM_CPY×8 (7 network write jet=1..7 + 1 local), REDUCE×7 (local), SYNC_STREAM×1; no CCU_GRAPH, `ccuChannels` is empty.
- **Direction entirely determined by rank dimension** (Section 5.5 rule 1): all 7 network MEM_CPY have `srcRank=0/dstRank=1..7` (write). Task offsets do not intersect with opMemInfo ranges (range inference fails), `direction_inferred=0`.
- **notifyId reused across ranks**: for example `RECORD(nid=30, dst=dev2)` (wait side is on remote end, skipped) appears paired with local `WAIT(nid=30)` (record side is on remote end, not in this db), two rounds of 7×2 each; 16 locally paired pairs (nid 3/13 main↔scheduling stream, 15..27 scheduling stream→remote stream, 6..12 remote stream→scheduling stream), 14 unpaired local WAITs → charged at `remote_wait_delay` (Section 5.4).
- lcEid/rmEid match ranktable.json verbatim (`…0040…01010001` → port `1/0` etc.), all network task ports are resolvable.

### 4.8 eid→rank Index (`build_rank_index`, Remote Rank Resolution)

Remote rank resolution (Section 5.5 SQE fallback, Section 6.7 CCU primary path) shares one eid(hex) → rank_id index. Construction priority and basis:

1. **RankEidMap table (primary path)**: The `RankEidMap` table in runnerdb `hccl_sim.db` (same directory as hccl_vm_data.db). In estimate mode, the Communicator table has only one row for the local rank. Use this table as authoritative for remote ranks (`sim_models.h` comment; host precedent `GetRmtRankIdByEid`, `device_sqe_parse_stub.cc:889`). Row BLOB layout is consistent with `sim::RankEidMap`: `id(u64) + rankId(u32) + eid(16B) + pad(4B)`; eid is "hex string byte order", same source and order as `ccuChannels.leid/reid`. Python side matches directly by `bytes.hex()` equality. This table is written by ranktable.json parsing (`topo_ascend_cluster_parser.cc:229`). In mock-comm scenarios, it is overwritten along with communicator initialization, so it is authoritative.
2. **ranktable.json (fallback)**: When RankEidMap is empty/`hccl_sim.db` is missing/the table does not exist, expand `rank_list[].rank_id ↔ level_list[].rank_addr_list[].addr` (addr_type=EID) to build an isomorphic index. Measured results show both path indexes are fully consistent (8 ranks × 9 eids = 72 entries).

Open read-only (`file:…?mode=ro` URI); on read failure (including table not existing), degrade with fallback rather than error. Note **`ccuChannels.srcRankId/dstRankId` cannot be directly used as communicator rank**: the writer side (`hccp_ccu_stub.cc:401-405`) explicitly fills global endpoint deviceId (channel establishment precedes communicator, the same channel can be reused by multiple domains). There is an offset from the opDetails.rankId numbering system. Checker deprecates this field for this reason (`g_allRankChannelInfo` only stores dstDeviceId/dstDieId).

---

## 5. SQE Task Graph Generation (Sub-Requirement 2 - Second Half, Focus of This Iteration)

### 5.1 Overall Pipeline

```
OpExecutionGroup
   │
   ▼
⓪ SQE batch merge (sqe_group_merge.py, Section 5.1.1): merge UDMA data tasks within same (stream, sqeGroupId) batch into 1
   │
   ▼
⓪' DPU sequence merge (sqe_group_merge.py, Section 5.1.2): merge DPU large message N-1 front blocks + tail block WRITE_WITH_NOTIFY into 1
   │
   ▼
① Per-SQE → Task node (5.2 mapping table; CCU_GRAPH handed to Chapter 6 for expansion)
   │     stream_id: int (SQE stream uses raw streamId, Section 6.5)
   ▼
② Stream serial chain: group by (deviceId, streamId), ascending global_id → dep_graph.add_stream_chain
   ▼
③ Notify pairing: cross-rank tasks do not participate in pairing (nodes preserved), only local notify first-come-first-served → dep_graph.add_match (Section 5.4)
   ▼
④ Network task parameter completion: direction inference / eid resolution (Section 5.5)
   ▼
⑤ CCU subgraph expansion (Chapter 6): CCU_GRAPH placeholder → microcode instruction interpretation → internal nodes + CKE pairing
   ▼
⑥ Output assembly: (SimConfig, DependencyGraph) → HcclSimulator(config, dep, topo)
```

### 5.1.1 SQE Batch Merge (Pre-Processing Before Graph Generation)

On the collection side, one UDMA doorbell (SQE) can carry multiple WQEs. Tasks are still stored per WQE, and marked with `sqeGroupId` for batch membership (circular incrementing doorbell batch number 1..255, wraps at 255, never takes 0; 0 = construction default "unmarked", Section 4.4.1). Before graph generation, split batches by **(streamId, sqeGroupId)**. The batch key explicitly includes stream. All data tasks within a batch (MEM_CPY/REDUCE/WRITE_WITH_NOTIFY) are merged into a single entry, aligned with hcomm batch transfer profiling `ExecProfilingAll` granularity. The three entry points (`pipeline.py`/`__main__.py`/`dump_graph.py`) call uniformly to ensure simulation and dump granularity are consistent:

- **Stream-grouped isolation guarantee**: The batch key contains streamId. **Cross-stream tasks never merge into the same batch** (one doorbell belongs to only one stream, a single doorbell batch is not split by the key); group numbers (doorbell batch numbers) are unique within a 255 window. Doorbells from other operators are not visible in the current operator task list but also consume batch numbers, so **same stream and same number must be from the same doorbell**. No cross-doorbell mis-merging (multi-operator db safe).
- **Descriptor information** (type/address/endpoint/notifyId/reduction parameters/profInfo) takes the **last** data task in the batch. **Length takes the full batch sum**. This is consistent field-by-field with `ExecProfilingAll(loc[insNum-1], rmt[insNum-1], totalSize, transferOp[insNum-1])`; trace/dump display names similarly take the op name of the last task (Section 11.4.1). When a batch mixes write/read directions or different dataTypes, it similarly follows the last task (single SQE batches are usually homogeneous, mixing is a rare scenario).
- **Not participating in merge**: notify/SDMA/synchronization tasks (notifyId is unique, pairing depends on order; local tasks are preserved in place and do not affect batch boundaries); unmarked/no-group-number tasks (`sqeGroupId <= 0`: 0 = unmarked, -1 = v3 and earlier BLOBs without this field) are preserved in place and **cut the current batch** (batch membership cannot be determined, conservatively end the current batch); UDMA non-data tasks (inline write notify) only participate in batch splitting, those interleaved within a batch are placed after the merged task (notify causality).
- The merged task occupies the slot of the first data task in the batch (batch head). The batch key containing stream ensures tasks within the batch share the same stream and same jetty. The stream serial chain (Section 5.3) and notify pairing (Section 5.4) relative order is not affected; local (non-UDMA) tasks do not participate in merge and do not reset batch boundaries (preserved in place, do not affect batch continuity).

### 5.1.2 DPU Sequence Merge (Pre-Processing Before Graph Generation)

DPU (host CPU RoCE) path large message WriteWithNotify sequence merge executes after 5.1.1 (`pipeline.py` unified call). Rules and semantics see Section 11.4.6: scan by DPU stream order for MEM_CPY chains with **contiguous src/dst dual addresses** (only `src_type==1` DPU source data tasks participate, other tasks are preserved in place and do not cut the chain). The chain is terminated and merged into a single entry by the address-contiguous WRITE_WITH_NOTIFY. **The terminating task is a natural terminator**. Two independent WriteWithNotify calls cannot merge across (the tail block of each call necessarily closes the current chain). The only remaining false merge scenario is "an independent pure write whose address happens to adjoin the WriteWithNotify start address" (semantically equivalent to a contiguous data segment + terminating notification, no difference in simulation results). Chains without termination are not merged (independent pure writes remain per-entry `RDMASend`); the merged task occupies the terminating task slot (notify causality). Merged front blocks are removed from their original positions. Unmerged tasks maintain original stream order.

**Single rank graph generation strategy (design premise, main difference from checker)**: The db contains only single rank task data. DependencyGraph describes only the tasks and dependencies of that rank. **All cross-rank pairing edges (post-wait relationships) are not built (task nodes preserved, approximated with fixed remote latency)**:

- SQE level: RECORD with waitDev ≠ local device (cross-rank post), WAIT with recordDev ≠ local device (cross-rank wait) **preserve task nodes**. They are graphed normally in stream order (src≠dst expresses remote, uses fixed remote latency), but **no pairing edges are built**. Only post-wait pairing within a single rank is considered (Section 5.4).
- CCU level: instructions that post to remote CKE, waits that depend on remote CKE being set **also generate task nodes** (Notify_Record/Notify_Wait, src=local, dst=INVALID → fixed remote latency), but no cross-rank pairing edges are built (Sections 6.4/6.6).
- **Cross-rank data transfers are preserved** (direction=write/read MEMCPY): they are the main source of network bandwidth usage from this rank's perspective. Only synchronization dependency edges are ignored, not the transfers themselves.
- makespan semantics: single rank perspective operator end-to-end duration = computation/copy within this rank + network transfers initiated by this rank + synchronization waits within this rank; when the remote end becomes ready is not modeled (equivalent to assuming the remote end is immediately ready).

**Simulation granularity**: default per-operator (one DependencyGraph per `(commId, opIter)`, one simulation, one record), consistent with checker per-operator validation granularity; `--all-in-one` optional full mode (multiple operators on the same stream are naturally serial).

### 5.2 taskType → hccl_sim Task Mapping

Aligned with HCCLSIM new task model (`TaskType`: `NOTIFY_RECORD`/`NOTIFY_WAIT`/`REDUCE_INLINE`/`MEMCPY`/`UB_INLINE_WRITE`/`UB_WRITE_OR_READ`/`WRITE_WITH_NOTIFY`, see Section 7.3):

| db taskType | Condition | hccl_sim TaskType | Task Parameters | Latency Model |
|-------------|-----------|-------------------|-----------------|---------------|
| NOTIFY_RECORD (1) | waitRank == local rank | `NOTIFY_RECORD` | notify_id=int, src=dst=local_rank | Fixed + trigger Event |
| NOTIFY_RECORD (1) | waitRank != local rank | `NOTIFY_RECORD` (cross-rank post: node preserved, not paired) | notify_id=int, src=local_rank, dst=remote rank (INVALID if unresolved) | Fixed remote_record_delay, no Event trigger |
| NOTIFY_WAIT (0) | recordRank == local rank | `NOTIFY_WAIT` | Same as above | Wait for Event |
| NOTIFY_WAIT (0) | recordRank != local rank | `NOTIFY_WAIT` (cross-rank wait: node preserved, not paired) | notify_id=int, src=remote rank (INVALID if unresolved), dst=local_rank | Fixed remote_wait_delay |
| MEM_CPY (3) | jettyId == INVALID | `MEMCPY` | size=len, src/dst_rank=taskData endpoint rank (default to local if unfilled) | HBM dynamic shared bandwidth |
| MEM_CPY (3) | jettyId valid, write direction | `UB_WRITE_OR_READ` | size=len, src_rank=local, dst_rank=remote rank, src/dst_eid, jetty_id, protocol, is_write=True | Port write direction bandwidth sharing |
| MEM_CPY (3) | jettyId valid, read direction | `UB_WRITE_OR_READ` | size=len, src_rank=remote rank, dst_rank=local, src/dst_eid, jetty_id, protocol, is_write=False | Port read direction bandwidth sharing |
| REDUCE (2) | jettyId == INVALID | `REDUCE_INLINE` | size=dataCount, data_type=task-level dataType enum name, src/dst_rank same as MEMCPY local branch | HBM dynamic shared bandwidth |
| REDUCE (2) | jettyId valid | Same as MEM_CPY network branch (Section 5.7 approximation) | Same as above (including data_type) | Same as above |
| WRITE_WITH_NOTIFY (7) | Always write direction (local src → remote dst, no direction inference) | `WRITE_WITH_NOTIFY` (**single task, no splitting**) | size=len, notify_id=notifyId, src_rank=local, dst_rank=remote rank (fallback through eid→rank index if unresolved), src/dst_eid, jetty_id, protocol, is_write=True; when isReduce=1, data_type=dataType enum name | Port write direction bandwidth sharing + fixed record latency after write lands (remote) |
| CCU_GRAPH (4) | — | (Expand, Chapter 6) | — | — |
| AIV_GRAPH (5) | — | Skip | — | Warning |
| SYNC_STREAM (6) | — | **Skip** (host-side stream sync point, Section 5.6) | — | — |

> waitRank/recordRank are taskData endpoint rankIds (carried by deviceId field, src=record side, dst=wait side, Sections 4.4.1/4.6). They are only used for **determining local/cross-rank**: cross-rank notify tasks preserve nodes during the mapping phase but do not enter pairing (Section 5.1 strategy). Pairing only occurs among local notifies. Local/remote for notify in the new model is expressed by `src_rank == dst_rank` (`is_local_notify`).
> **WRITE_WITH_NOTIFY (7) single task mapping** (one-to-one with db tasks, not split into write + NOTIFY_RECORD two sub-tasks): write direction is always write (launch end convention: src local → dst remote, no Section 5.5 direction inference). It uses port write direction bandwidth sharing; after write data lands, it sends notify to the remote end, charged at fixed record latency (remote wait is not in this db, single rank simulation premise, same as cross-rank record semantics; self-write src==dst edge case uses local record latency); `notifyId` is preserved with the task (trace/dump display). When `isReduce=1`, it only affects `data_type` display (same enum name mapping as REDUCE branch).
> **Network tasks directly consume taskData for rank** (deviceId field carries rankId; launch end convention: write=src local/dst remote, read=src remote/dst local): write fills `dst_rank=remote rank`, read fills `src_rank=remote rank`; when remote rank is unresolved (`0xFFFFFFFF` or equals local), fall back to INVALID (Section 10.1). Direction is explicitly carried as `is_write`. The simulator no longer depends on `src_rank==0` heuristic (Section 10.3 #1 fixed; task.txt old input path still uses heuristic).
> Local eid prioritizes SQE explicit lcEid (v2 collection new addition). Default falls back to jettyId → ccuChannels channel leid derivation (Section 5.5, Section 10.1).
> The TaskType in this table is for internal latency model classification; trace/dump task display names have switched to profInfo.profTaskType op names (Section 11.4.1 mapping table, Section 10.3 #18 implemented). The correspondence with this table TaskType follows the "Simulation TaskType" column in Section 11.4.1.

### 5.3 Stream Serial Chain Construction

- Group by `(deviceId, streamId)`. Within each group, form a chain by `global_id` ascending order (= SQE dispatch order).
- The simulator launches one simpy process per chain for serial execution (Chapter 7). **No need for checker mainStart virtual start point and main stream special handling**: simulation focuses on end-to-end makespan. Each chain starts in parallel from time 0; the WAIT of the first task in each chain naturally forms the actual synchronization with the main stream.

### 5.4 Notify Pairing Algorithm (Local Only)

**All cross-rank post-wait edges are ignored** (Section 5.1 strategy): before pairing, classify by Section 4.6 remote end backfill results. Cross-rank notify tasks (RECORD with waitDev ≠ local device, WAIT with recordDev ≠ local device) preserve nodes during Section 5.2 mapping (graphed in stream order) but do not enter pairing. Therefore the pairing key simplifies from checker's triple to a single notifyId:

```
checker semantics:   pairing key = (recordDeviceId, waitDeviceId, notifyId)   ← multi-rank
This plugin:         filter cross-rank first → pairing key = notifyId          ← single rank
```

Consistent with checker `AddLocalNotifyEdges` (dispatch order simulation + first-come-first-served consumption, `task_graph_generator_v3.cc:847-946`) in principle, light_performance simplifies to **two-pass scan + first-come-first-served**:

```
Input: All local notify tasks for an operator (cross-rank nodes preserved but not participating in pairing)
Pass 1: Scan in ascending global_id order
    RECORD → pending_records[notifyId].append(task_id)                        # Queue
    WAIT   → if pending_records[notifyId] is not empty:
                  rec = pop front; dep_graph.add_match(wait_id, rec_id)       # First-come-first-served
              else:
                  unmatched_waits.append(wait_id)                              # Wait for "later" record
Pass 2: Handle unmatched_waits (cases where WAIT appears before RECORD)
    Replay in global_id order: unmatched WAIT is suspended, RECORD pairs with earliest suspended WAIT
    (Successfully paired RECORD is simultaneously removed from pending_records to ensure diagnostic statistics accuracy)
Cleanup: Remaining WAITs → normal (see below); remaining RECORDs → warning, should not appear in normal data
```

Notes:

- The db id order is the dispatch order, and the stream serial chain already guarantees "tasks after a WAIT on the same stream do not execute first". Therefore the complete queue simulation and deadlock detection of checker are not needed (deadlocks surface as makespan anomalies/unexecuted tasks in simulation).
- Multiple RECORD/WAIT with the same notifyId (for example n11/n15 in early 169B samples repeatedly appear on stream 6, main stream two iterations) are naturally paired by the first-come-first-served rule. **notifyId reuse is allowed**.
- **Unpaired local WAIT is normal semantics**: in the v2 AICPU sample (Section 4.7), notifyId is reused across ranks. Local `RECORD(nid=30, dst=dev2)` (wait side on remote end, node preserved but not paired) appears paired with local `WAIT(nid=30)` (record side on remote end, not in this db); this WAIT is charged at `remote_wait_delay` fixed latency in simulation (Section 7.6). The warning is only informational.
- Cross-rank pairs in early 169B samples: `RECORD(n18, s1→d2)` (notify dev2) and `WAIT(n18)` (backfilled recordDev=2) both preserve nodes, no cross-rank pairing edges built: graphed in stream order, `src≠dst` uses fixed remote latency; data transfer tasks on stream 8 (MEMCPY(jet=3) etc.) are not affected.

### 5.5 Network Task Direction and eid Resolution

**Direction determination**: network data tasks are uniformly mapped to `UB_WRITE_OR_READ`. Direction determines port charging direction (is_write) and rank endpoint values. taskData endpoint rankIds are carried by deviceId fields (Section 4.4.1 rank field rollback). Launch end convention (`device_sqe_parse_stub.cc`): write = src local/dst remote, read = src remote/dst local:

| Priority | Rule | Basis |
|----------|------|-------|
| 1 | srcRank == local and dstRank ≠ local → write; dstRank == local and srcRank ≠ local → read | Collection endpoint rankId (carried by deviceId field, Section 4.4.1). Measured samples (Section 4.7) all determined by this rule, no inference needed |
| 2 | When rank does not distinguish (equal or both unfilled), use memory range: dst falls within local output range → read; src falls within local input range → write | Induced from measured samples (Section 4.7): fails when task offsets do not intersect with opMemInfo ranges |
| 3 | Fallback: default write, diagnostic statistics `direction_inferred` count | — |

> Network tasks explicitly carry `is_write` on Task (Section 10.3 #1 fixed; CCU path Section 6.7 also fixed). The simulator selects write/read charging direction and port query eid (`src_eid`/`dst_eid`) based on this. It no longer depends on `src_rank==0` heuristic; task.txt old input path does not have this field and falls back to heuristic determination.
> `WRITE_WITH_NOTIFY` (7) **does not participate in direction determination**: semantics are always write (local src → remote dst). Graph generation directly sets `is_write=True`.

**eid resolution**: network tasks look up ranktable by eid to get physical ports (`TopoResolver.get_ports_by_eid`):

```
SQE:        Local eid prioritizes SQE explicit lcEid (v2 new); default through jettyId
            ──ccuChannels(local device outbound channel)──> leid(local eid)
            Remote end prioritizes channel reid, default falls back to SQE rmEid; when channel is missing, local eid degrades to any local channel leid
CCU microcode: channelId → channels[(dev,die,channelId)] → leid/reid + jettyId list
            When channel is missing, local eid prioritizes the mission SQE lcEid, degrades to any channel leid on the same die
write direction: simulator queries task.src_eid → ports; read direction queries task.dst_eid → ports
```

- `protocol`: channel protocol column → HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO (CommProtocol name, display only); `link_type`/`transport_type`: task profInfo (DfxLinkType/ProfTransportType name, Sections 11.4.2/11.4.3, `_add_task` unified population, for reconciliation with msprof L1).
- When local eid cannot be resolved, issue warning + `eid_unresolved` count; when network tasks exist but no ranktable is available, skip simulation entirely (TopoResolver throws error directly for missing eid, see Section 10.3).

**Remote rank resolution**: remote rank for SQE network tasks (`_peer_rank`): taskData endpoint rankId (carried by deviceId field, launch end already resolved through `GetRmtRankIdByEid`) takes priority; when unresolved (UINT32_MAX/local value), fall back through eid→rank index (Section 4.8) by remote eid. If still unavailable, use INVALID + `rank_unresolved` count.

### 5.6 SYNC_STREAM Handling

Host-side stream synchronization point (appears once per iteration on the sample main stream). It indicates the host waits for the current stream to complete. **It does not represent a device-side task**: no task node is generated, no time is consumed (makespan is the end time of the last task in the stream chain).

- Diagnostic statistics `skipped_sync` count;
- Previously approximated by "unpaired local `NOTIFY_RECORD` fixed latency placeholder". This was removed because it introduced false `local_record_delay` at stream tail (and appeared as pseudo notify tasks with notify_id=invalid in trace).

### 5.7 REDUCE Approximation

Reduction = network transfer + computation. The current model approximates with data transfer:

- Network (jettyId valid): charge port bandwidth latency as `UB_WRITE_OR_READ` (direction determined same as Section 5.5).
- Non-network (jettyId INVALID): map to `REDUCE_INLINE`, charge latency by HBM bandwidth dynamic sharing (`size / hbm_bandwidth`); computation latency is not separately listed (can be calibrated by `dataType×dataCount` later, see Section 10.1 data_type field gap).

### 5.8 Comparison and Simplification versus Checker Graph Generation

| Checker Step | light_performance | Reason |
|--------------|-------------------|--------|
| mainStart virtual node + main stream incoming edge special handling | Removed | Simulation only needs makespan, chain-head WAIT already expresses synchronization |
| AddLocalNotifyEdges queue simulation + deadlock detection (GRAPH_DEADLOCK) | Two-pass scan first-come-first-served, no deadlock detection | No validation requirement; anomalies surface as simulation results |
| AddInterRankNotifyEdges cross-rank pairing (full-graph BFS simulation) | **Completely removed**: all cross-rank post-wait pairing edges ignored (notify task nodes preserved, graphed in stream order) | Single rank graph generation (core difference) |
| MemSlice address resolution + memory layout table | Only range attribution for direction determination (optional) | Simulation only needs len, does not need precise memory model |
| CheckSlaveTaskQueue slave stream structure validation | Removed | Not a validation tool |
| Memory conflict/semantic validation/synchronization conflict/reachability/single-task checks | All removed | Not a validation tool |
| CompactSyncNodes (SYNC_STREAM merging) | Skip, do not generate nodes (host-side sync point, not device-side task) | makespan is the end time of the last task in the stream chain (Section 5.6) |
| AIV subgraph expansion | Not supported (skip with warning) | Requirement explicitly does not consider this |
| CCU subgraph: queue parallelization transformation, loop merge, resource conflict validation | Queue parallelization simplified implementation (Section 7.4.1: assembly-phase skeleton/unit splitting + fork/completion set joining and extra edge attachment + `#v` virtual stream); loop merge equivalently implemented (Section 6.5); resource conflict validation removed | Parallelization eliminates queue serialization distortion (Section 7.4.1); no validation needed, node count is manageable |

### 5.9 Output Assembly

```python
def build(group: OpExecutionGroup, sim_config: SimConfig | None = None
          ) -> tuple[SimConfig, DependencyGraph]:
    dep = DependencyGraph()
    # ① SQE → Task (CCU_GRAPH temporarily stored as placeholder)
    # ② Stream chains (int stream id)
    # ③ Notify pairing (add_match)
    # ④ Direction/eid completion
    # ⑤ CCU expansion (ccu_graph_builder.expand_ccu_graph(..., local_rank, alloc_stream))
    return sim_config, dep
```

- `SimConfig` source: JSON specified by `--config` / manifest setting (Section 7.2 format, units us/GBps), with bandwidth utilization configuration overlaid.
- Topology: `TopoResolver` is constructed by the CLI layer (ranktable location see Section 3.3) and passed to the simulator.
- The output can immediately be sent to `HcclSimulator(config, dep_graph, topo).run()` (returns dict, see Section 7.1).

---

## 6. CCU Mode Task Graph Generation (Sub-Requirement 3, Focus of This Iteration)

### 6.1 CCU Data Organization in db

```
SQEs with taskType=4 (CCU_GRAPH) in opTask_P_<pid>
    └─ taskData.ccu: {dieId, missionId, timeout, instStartId, instCnt, key, argSize, args[13]}
           │  instStartId/instCnt define instruction execution window [instStartId, instStartId+instCnt)
           │  args[13] consumed one by one by LOAD-type instructions (cursor advances to next SQE after reading all 13 args)
           ▼
ccuInstrRes (deviceId, dieId) → instrSpace: 32K × 32B microcode instruction space
ccuChannels (deviceId, dieId, channelId) → {dstDeviceId, dstDieId, leid/reid, jettyId[]}
    └─ Channel register values → remote location and jetty
```

- SQEs are grouped by `(dieId, missionId, queue)` (single device graph generation, deviceId is constant; queue is the SQE stream) into dispatch contexts (checker `CcuMissionKey` + latest node append semantics, `task_meta_translator_v3.h:47-62`): SQEs with the same key merge into the same context **only when contiguous with the previous one** (previous `instCnt==13` and `instStartId` connects end-to-end, `IsContiguousCcuSqe`, `task_meta_translator_v3.cc:213-231`). Otherwise **create a new context**. Multiple dispatches of the same mission (instruction window wraparound re-dispatch, typical pattern "head SQE (13 Load SqeArgs) + body SQE" dispatched twice) each expand independently: each context fully replays its own instruction window (including Load SqeArgs header, re-initializing loop counters and other registers with **this dispatch's args**). This prevents the later dispatch from using residual register state from the previous one, which would cause data transfers to be lost (Section 10.3 #20).
  - **args consumption** (checker `CcuGraphStateV3::GetSqe` semantics): SQEs within a context are arranged in dispatch order. Args are consumed sequentially per slot. After reading all 13 args (reading the last slot `sqeArgsId==12`), the cursor advances to the next SQE; in the typical layout, the 13 Load SqeArgs of the head SQE consume head SQE args[0..12], and subsequent Load instructions within the body SQE window consume body SQE args.
  - **Register/CKE bitmap shared globally across contexts** (checker `AllRankParamRecorder::Global()` semantics): waits in a later dispatch can be resolved by CKE bits / residual post accounting left by an earlier dispatch. This is consistent with checker multi-graph shared recorder behavior.
  - **Dispatch contexts of the same queue advance strictly serially in dispatch order** (checker queue semantics: a single queue strictly follows instruction order, the next TaskCcuGraph on the same stream must wait for the previous one to complete); cross-queue round-robin joint advancement (Section 6.6).
- Subgraph attachment point: the position of CCU_GRAPH SQE in the stream chain is the subgraph entry; after expansion, this placeholder is removed (no placeholder nodes are introduced in the task graph). The subgraph attaches to SQE predecessor/successor tasks through extra edges (Section 7.4.1).

### 6.2 Microcode Instruction Format (32B Fixed Length)

Same source as `${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode` (reference copy in this repository: `ai_skills/skills/convert-ccu-microcode/references/ccu_microcode_v1.h`):

```
CcuInstr = CcuInstrHeader (2B) + union { V1(A5) | V2(A6) | V3 } (30B)

CcuInstrHeader bit field (little-endian u16):
    code     : 11 bit   (header & 0x7FF)      Instruction code
    type     :  4 bit   ((header >> 11) & 0xF) Instruction category
    reserved :  1 bit

Instruction categories: LOAD_TYPE=0x0 | CTRL_TYPE=0x1 | TRANS_TYPE=0x2 | REDUCE_TYPE=0x3
```

Chip generation → instruction set version mapping (checker `GetCcuInstrVersion`, `ccu_task_transform_v3.cc:175-184`): `Ascend950 → A5/V1`, `Ascend960 → A6/V2`, other devType falls back to A5; derived from Device soc_version (hccl_sim.db) or `opDetails.devType` (`expand_ccu_graph` selects `_MissionContext`/`_MissionContextA6` interpreter by `dev_type`). **Both instruction set generations are implemented** (A6 ported from checker `ccu_task_transform_instruct_a6_v3.cc`, Section 11.3.3); A5/A6 instruction set difference overview see Section 6.8.

### 6.3 Instruction Type → Task Mapping

#### 6.3.1 Summary Table

| Instruction Category | Representative Instructions | light_performance Behavior | Produced Task |
|---------------------|------------------------------|---------------------------|---------------|
| LOAD (register write) | LoadSqeArgsToXn/GSA, LoadImd, LoadX, arithmetic-logic (A6 Add/Sub/...) | Update register model, data plane **produces no nodes**; A6 LOAD instructions commonly carry `setCKE` tail field (mask non-zero → local `NOTIFY_RECORD`, checker `ProcessSetMask` semantics); A6 Load/Store (HBM↔Xn data plane) does not model HBM content, registers are not backfilled (only setCKE synchronization semantics preserved, warning logged) | —(Notify_Record when A6 setCKE non-zero) |
| CTRL: SetCKE/SetCKBit | Set CKE bit | Local → `NOTIFY_RECORD` (register CKE accounting, with mask); remote target (cross-rank post, SyncCKE etc.) → remote `NOTIFY_RECORD` (not paired, Section 6.6) | Notify_Record |
| CTRL: Wait/Jmp/Fence | Wait CKE bit / jump / fence | CKE wait (A5 embedded waitCKE field): if bits insufficient, pause the queue pointer; if bits satisfied, generate `NOTIFY_WAIT` (split by contributing bits, with mask); waits depending on remote bits → remote `NOTIFY_WAIT` (not paired, fixed latency, Section 6.6). A6 independent Wait instruction (Xn value comparison spin, checker `TransformWaitInstr`): satisfied → pass through with no node; unsatisfied = depends on remote register write (SyncWtX/TransMem notify) → top-level resolves by stall, generates remote `NOTIFY_WAIT` (fixed `remote_wait_delay`), within loop body directly generates node and continues. Jmp: A5 evaluates by `Xn[conditionXnId] != expectData` (checker `TransformJmpInstrA5` semantics); A6 supports 6 condition codes (==/!=/>/>=/</<=, ≥6 unconditional) + relative/absolute two modes (jumpMode, target `Xn[relTarInstrXnId]`); target limited to current instruction window, out-of-bounds warning pass through; pass through within loop body; backward jump infinite loop defense with step budget. Fence (A6 only): checker does not support, warning pass through (queue serial model naturally guarantees order) | Notify_Wait |
| CTRL: ClearCKE | wait + clear bits | Wait semantics; A6 clears consumed wait bits per checker `ClearWaitMask` semantics (`clearMask != 0` is checker unsupported scenario, warning + lenient clear of target bits) | Notify_Wait |
| CTRL: Loop/LoopGroup | Loop | loopCnt from register, **merge expansion** (Section 6.5, fallback to full expansion on failure); A6 Loop `mode==1` is conditional expansion (only executes when `wishCKEBit == Xp.loopCtxId + 1024` matches, always takes full expansion path) | None (body instructions mapped per this table; under merge path, data task size × total iteration count) |
| TRANS (transfer) | LocMem↔LocMS/RmtMem/RmtMS (A5), local 4-code + unified TransMem dmaOpCode (A6) | Map by role (6.3.2) | `Memcpy`/`Ub_Write_Or_Read` |
| TRANS: Sync (cross-rank synchronization) | SyncCKE/GSA/Xn (A5), SyncXnWtx/SyncAtx (A6) | Remote set/notify → remote `NOTIFY_RECORD` (not paired, Section 6.6); value synchronization is remote register write, no node built | Notify_Record (remote) |
| REDUCE | Add/Max/Min | Same as SQE REDUCE approximation (Section 5.7); MS is rank-local buffer, always local HBM reduction | `Reduce_Inline` |

#### 6.3.2 TRANS Transfer Role Determination (Consistent with checker `ccu_loop_merge_v3.cc:73-144`)

By device ownership of src/dst:

| src | dst | Role | hccl_sim Task |
|-----|-----|------|---------------|
| Local | Local | LOCAL_COPY | `Memcpy` (HBM dynamic sharing) |
| Local | Remote | WRITE | `Ub_Write_Or_Read` (src_rank=local, dst_rank=remote, is_write=True) |
| Remote | Local | READ | `Ub_Write_Or_Read` (src_rank=remote, dst_rank=local, is_write=False) |

> Cross-die/card read and write are both mapped to `Ub_Write_Or_Read` in both directions. Direction is expressed through `is_write`/trace `direction` field. Ports are charged independently by write/read direction; remote rank is resolved through channel reid → eid→rank index (INVALID if unresolved). To preserve microcode-distinguishable buffer information, CCU-converted transfer tasks carry `src_buf_type`/`dst_buf_type` (mem/ms, determined by instruction source/destination operand). Tasks from SQE/CLI paths with the same name do not have this information and output invalid value N/A.

> Cross-rank **data transfers** (WRITE/READ roles) are preserved. They are the main source of network bandwidth usage; only cross-rank **synchronization edges** (post-wait, see Sections 6.4/6.6) are ignored.

Key field sources:

- `size`: length register (A5 `lengthXnId` direct read / A6 `xlId` through `GetXnId`. When bit15=1, register number is offset by loop `xnIdOffset`) → look up register model.
- Remote location: A6 channel through **register value** (`Xn[GetXnId(xcId)]` value is channelId) → `channels[(dev,die,channelId)]` → `{dstDeviceId, dstDieId}`; A5 channel is instruction immediate field. A5 MS-type instructions `msId / 0x8000` = rank, `msId % 0x8000` = MS offset (MS page 4KB). This is **only addressing encoding**: MS is a local buffer directly accessible by rank (checker `GenSliceFromMs` + `AddLocalBatchReduce` treats all msId as local). The high bits of msId are **not used as** local/remote task type criteria; cross-rank data movement is expressed by Rmt* role instructions (through channels). A6 TransMem `msIdMode==1` marks the MS end (source for write / destination for read is local MS, `src/dst_buf_type=ms`).
- eid: channel leid/reid (hex string); local eid prioritizes the mission SQE `lcEid` (v2 new), defaults to same-die channel leid approximation.
- A6 `TransMem` `dmaOpCode` (implemented, checker `TransformTransMemInstr`): 3=write, 5=write with notify (notify value writes to remote Xn/CKE, target type unresolvable in single rank → approximate as remote record), 6=read, 0x1A=remote atomic count (no data plane, approximate as remote record), 0x19 (compressed write+atomic) and others unsupported (warning skip data plane, setCKE local post still executes to maintain synchronization skeleton); `udfEnable` (network reduction) charged as ordinary UB transfer (Section 5.7 approximation, remote reduction not modeled); `udfType != 0` warning skip data plane.

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

- LOAD/arithmetic-logic instructions "execute" on this model (read/write registers); uninitialized read → warning and treat as 0 (checker errors and exits, here lenient handling to accommodate incomplete data). A6 arithmetic-logic instructions (Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt) execute with 64-bit wraparound semantics (Mul takes lower 32-bit product, shift amount truncated by 0x3F, arithmetic shift interprets as signed); LoadX/StoreX indirect addressing target register number out of bounds (>16383) warning skip write (checker errors and aborts).
- **A6 in-loop register number offset** (checker `GetXnId`/`UpdateXnId`): when register number field bit15=1 in the payload, within the expansion region (`curLoopIdx >= expandOffset`), offset the register number by `xnIdOffset × curExpandCnt`; when bit15=0, use original number directly. CKE/MS id loop offset follows the same rule for both generations (`UpdateId`).
- **CKE pairing rules** (consistent with checker `GenWaitNode`, `ccu_task_transform_v3.cc:56-121`):
  ```
  post matches wait ⇔ post.waitDeviceId == wait.waitDeviceId
                   && post.dieId == wait.dieId
                   && post.ckeId == wait.ckeId
                   && (post.ckeMask & wait.ckeMask) != 0        # Bits have intersection
  After match: one wait instruction aggregates to at most 1 local wait node (mask = OR of all contributing
            post bits) + 1 remote wait node (bits depending on remote set)
            post.remaining_mask &= ~wait.ckeMask                  # One post's bits can be consumed by multiple waits in parts
            Emptied → remove from seen_post
  Node dependency: aggregated wait node builds edges to all contributing posts. The first contributing post
            uses notify pairing (add_match, Event semantics), the rest use extra edges
            (child=wait, parent=post, "wait for all parents to complete" pure timing constraint)
  ```
- **CKE accounting only registers posts from this rank** (single rank graph generation): cross-rank posts generate remote `Notify_Record` nodes (src=local, dst=INVALID → fixed remote latency) but do not register in `seen_post` (the set action happens on the remote rank's CKE); wait bits depending on remote sets generate remote `Notify_Wait` nodes (fixed latency, not paired), see Section 6.6.
- When bits are not satisfied, the queue pointer pauses (waiting for posts from other queues of this rank to unlock). Multi-queue joint advancement see Section 6.6 algorithm; when the same queue has multiple dispatch contexts (same mission multiple dispatches, Section 6.1), the later context must wait for the earlier context to complete before starting advancement.
- Final validation: `seen_post` residuals → warning and count in diagnostic statistics (checker reports GRAPH_UNMATCHED, here downgraded).

### 6.5 Queue Model and Loop Handling

- **queue**: `CcuSqeParam` groups by SQE stream (checker uses ccuParams[queueId]). Each queue forms one serial chain. The new task model `stream_id` is **int**: SQE streams use the raw streamId; CCU queue chains use allocator to get `alloc_stream("ccu_d{die}_q{queue}")` (base address `1<<20` sequential allocation, same name reuse merges multiple missions of the same queue). Logical names are preserved in `stream_labels` for dump/report display. Within the same queue, instructions are strictly ordered (checker semantics: `ccu_task_transform_v3.cc:189`, "a single queue advances strictly in instruction order; cross-queue coordination is only expressed through generated wait/post edges").
- **Assembly-phase queue parallelization splitting** (Section 7.4.1): after interpretation execution completes, each CCU queue chain is split into skeleton (wait / free set / remote sync nodes. Free set means record nodes not immediately following data tasks) and async data transfer units two categories: skeleton stays on the queue serial chain to express the timing framework. Async units are re-attached to virtual streams `ccu_d{die}_q{queue}#v{n}` (sequential allocation within the same base address space as the queue, reused across missions, not reused within a mission). They connect to the skeleton through fork extra edges. The serial chain only constrains skeleton and unit internals, eliminating "entire queue task-by-task serial" distortion.
- **loop expansion** (merge priority, aligned with checker `ccu_loop_merge_v3.cc`):
  - `Loop`/`LoopGroup` instruction iteration count comes from registers and is known during interpretation:
    - A5: LoopGroup parameters are in Xn (LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41]) and Xm (gsaOffset/MSOffset/ckeOffset). Loop iteration count is in Xn[xmId] lower 13 bits;
    - A6: LoopGroup parameters are carried by **Xn/Xm/Xp three registers** (checker `LoopGroupXnA6/XmA6/XpA6`). Xn: `loopInsCnt:10 | expandOffset:9 | expandCnt:9`, Xm: `ckOffset:10 | msOffset:11 | gsaOffset:32`, Xp: `xnIdOffset:32` (GetXnId offset, Section 6.4); Loop iteration count is in `Xn[GetXnId(xmId)]` lower 13 bits, GSA step is in `Xn[xnId]` lower 32 bits, context id is in `Xn[xpId]` lower 9 bits; `mode==1` is **conditional Loop**. It only expands when instruction `wishCKEBit == loopCtxId + 1024` (`A6_LOOP_BOARD_ENTRY`) matches. Otherwise this expansion is skipped (checker `ProcessLoopIns`). Conditional Loop always takes the full expansion path (checker `CollectLoopExpandA6` rejects merging).
    Each Loop within a LoopGroup first tries the **merge path** (checker `TryProcessMergedLoopA5/A6` → `LoopIterationMerge` / `LoopExpandMerge` semantics). If validation fails, fall back to full iteration expansion.
  - **Merge prerequisite validation** (any failure triggers WARN fallback, same semantics as checker "Skip merged-loop optimization"):
    1. Loop body only contains mergeable instructions:
       - A5: simple TRANS / SYNC TRANS (SyncCKE/GSA/Xn) / REDUCE / ClearCKE (`CollectLoopCkeOpsA5` rules; SetCKE, LOAD, LOOP, JMP, `setRmtCKEMask != 0` TransMem, `clearMask != 0` ClearCKE are all rejected);
       - A6 (`CollectLoopCkeOpsA6` + `Collect*LoopInstrA6` rules): local TRANS 4-code / TransMem (`udfType==0` and `dmaOpCode ∈ {3,6}`. 5 write with notify triggers remote post per iteration, rejected; channel must be resolvable, `IsExistRemoteDieInfo`) / REDUCE (`udfEnable` TransMem network reduction also folds by trans slot) / ClearCKE (`clearMask==0`); SetCKBit, LOAD, LOOP, JMP, Wait are all rejected;
    2. wait/set masks are **balanced** within the loop body (`ConsumeProducedMask`/`ProduceMask`: each wait's bits must be produced by an earlier set, all produced bits are exhausted at sequence end);
    3. Cross-expansion CKE/MS resources have no conflict (`CheckResourceConflict`: CKE/MS IDs after ckOffset/msOffset offset must not be reused across expansions; A6 MS conflict check covers local transfer/TransMem `msIdMode` MS end and reduction msId lists);
    4. Transfer/reduction length > 0 (`ValidateLoopLen`) and expansion count × iteration count > 0;
    5. **A6 additional**: wait CKE IDs within loop body must not be reused (`CheckA6LoopWaitCkeIdUnique` hardware constraint, checker errors and aborts, here downgraded to fallback full expansion); iteration count/length/channel values across expansions must be consistent (checker each expansion `loopCnt` equal + `InstrMerge`/`MemMerge` mergeability performance-side approximation. Register numbers after xnIdOffset offset may point to different registers).
  - **Merge emission** (`EmitMergedLoopInstrsV3` semantics): the loop body executes **once** by sample expansion (curExpandCnt=0). Per slot: wait → data task → set → clear wait bits; data task size = single-iteration length × (expansion count × iteration count) (timing-equivalent folding, consistent with checker AddBatchTransMem/AddBatchReduce batch tasks); CKE register bitmaps of other expansions replay by or/clear idempotency, guaranteeing post-loop register state is consistent with full expansion (only PostRecord task accounting is not duplicated). All tasks within the emission window (data tasks and folded notify tasks) carry `loop_cnt` (loop count after expansion = N+1 after N expansions) and `loop_iter_cnt` (self-loop iteration count per single loop). Total iteration count = `loop_cnt` × `loop_iter_cnt`, preserving pre-merge loop details for subsequent simulation use; straight-through instructions, full expansion fallback, and SQE/CLI path tasks have both fields as 0 (invalid, trace uniformly outputs N/A).
  - **Quantified effect** (REDUCE_SCATTER measured db, 128 expansions × 50 iterations × 13 instructions): task count 192,005 → 66 (10 data tasks, several local post/wait, 35 remote sync nodes. 21 remote post + 14 remote wait, Section 6.6). Graph generation + simulation end-to-end ~55s → ~6ms. makespan 70,113.650 us → 2,753.711 us (this merge eliminates 128K task-level notify overhead and 64K small transfer per-task latencies; final value after queue parallelization and remote sync node inclusion, Section 7.4.1).
  - A6 LoopGroup configuration validation (`loopInsCnt≤512`, `expandCnt≤511`, `expandOffset≤511` and `expandOffset≤loopInsCnt`, checker `TransformLoopGroupInstr`). When not satisfied, warning skip the entire LoopGroup (defense against uncontrolled expansion from abnormal register values); "waitCKE within loop must not duplicate CKE ID" validation see merge prerequisite validation #5 above (`ccu_task_transform_instruct_a6_v3.cc:1126-1146`).

### 6.5.1 Merged Loop Pipeline Concurrency Recovery (Loop Tag + Characteristic Chain Resolution)

**Problem: Folding pseudo-serialization (merge is not timing-equivalent for notify structure)**: Section 6.5 merge is timing-equivalent folding for **data tasks** (size × total iteration count, total duration unchanged in bandwidth-limited scenarios), but not for CKE wait/set handshakes within the loop body. During real microcode execution, the N expansions of a LoopGroup are dispatched concurrently. Each expansion internally advances by M inner iterations in pipeline (typically 4KB scale per iteration). The `TransMemToMs → setCKE → waitCKE → TransLocMsToRmtMem / TransLocMsToLocMem` within the loop body is **per-iteration fine-grained handshake**: the k+1 iteration HBM→MS prefetch (memcpy engine) overlaps with the k-th iteration network write + local copy-out (channel / memcpy engine). Steady-state throughput ≈ max(each phase total). After merge folding, per-iteration handshake degrades to a **single pair** of wait/record nodes. Semantics become "all front data (× N × M iterations) complete → back section can start" **full barrier**. Pipeline concurrency is folded into serialization.

Using the first loop template in `hccl_vm_install/data/hccl_vm_data.db` as example (microcode 0–10: `TransMemToLocMs(4KB, set 0:1) → waitCKE(0:1) → 7× TransLocMsToRmtMem(each 4KB, set 0:1..0x40) → TransLocMsToLocMem(4KB, set 0x80) → waitCKE(0:ff)`; LoopGroup 128 expansions × 256 inner iterations = 32768 times × 4KB = 128MB/task):

```
Current graph (after 7.4.1 parallelization;  AA = loopgroup predecessor task representative, meaning the last node
                            before the loop emission window in the queue skeleton):
  AA -> memcpy(mem->ms) -> record(0:1)                     ← Prefetch unit (128MB, 95.95us)
  record(0:1) →[paired]→ wait(0:1) → 7× ub_write + memcpy(ms->mem)
                                    ↑ Full serialization: back section 8 units fork from wait(0:1),
                                      wait for prefetch 128MB to fully complete through pairing edge
  Multiple { record(0:xx) / record(0x80) -> wait(0:ff) }   ← Loop end barrier (preserved)
```

- Real execution: steady state ≈ max(prefetch 95.95, 7 port writes 2526.48) us;
- Current graph: serial 95.95 + 2526.48 us.
- **Quantified** (sample db ALLGATHER, measured before and after this fix): makespan **2657.52 → 2561.54 us**. Eliminates 95.95 us serial prefetch phase; after prefetch and local copy-out run concurrently through HBM dynamic sharing, each 191.79 us (correct expression of pipeline steady state, Section 7.5). 7 port network writes 2526.48 us becomes the only critical path. Error upper bound: when two phases belong to different engines (HBM prefetch vs network write) and have comparable durations, loop body duration is overestimated by nearly **2×** (serial t1+t2 vs steady state max(t1,t2)); same-engine phases (for example pure HBM prefetch+copy-out) serialization error is absorbed by HBM dynamic sharing (total unchanged); error accumulates linearly with merged loop count.

**Correctness premise: merge validation invariant**: Section 6.5 merge prerequisite validation #2 (wait/set masks balanced within body) guarantees that **all wait bits within a merged loop body are produced by earlier sets within the body and exhausted at the end**. This means all CKE dependencies within the merge emission window are intra-loop per-iteration pipeline handshakes. No real data dependencies exist across loops (otherwise validation fails and falls back to full expansion). Therefore resolving "intermediate waits within the body and their consuming sets" does not break loop-to-external synchronization: **loop end wait** (the last wait within the emission window, after which no same-loop data tasks exist) and **mission completion set** (skeleton tail + all unit tails, Section 7.4.1) continue to carry post-loop instruction / SQE successor / same-queue next mission queue-level serial semantics.

#### Approach (Loop Tag + Characteristic Chain Identification + Deletion and Reconnection, Iterate to Convergence)

1. **Loop tag**: `_MissionContext` maintains the current emission window `_loop_tag`, named `loopgroup_{i}_loop_{j}` (i = LoopGroup instruction execution sequence number within dispatch context, starting from 0; j = cur_loop_idx within group, starting from 0; when LoopGroup is nested within loop body, each assigns by emission window, no hierarchy). `_emit_merged_loop` window-internal bits, `add_task` defaults inherit to new field `Task.loop_tag` (same pattern as `loop_cnt`). **Only merge path gets tagged**; full expansion fallback / SQE / CLI / straight-through instructions get empty string (trace outputs N/A). Tags also serve resolution scope limitation (no cross-loop / cross-mission accidental deletion) and trace/dump traceability.
2. **Characteristic chain identification** (recorded during emission + determined before assembly, avoiding graph result reverse lookup): when `_gen_waits` pairing succeeds, record (wait_task_id → all contributing post task_ids) into `ctx.wait_posts` (aggregated wait match + extra all contributors, Section 6.4). During the assembly phase (`expand_ccu_graph` assembly loop internals, **before** skeleton/unit splitting), scan the ctx.nodes sequence. A characteristic chain must satisfy all of:
   - W is a local paired `Notify_Wait` and `loop_tag == T` (no remote wait exists within merge window. Validation #2 guarantees all body wait bits are locally produced);
   - All contributing posts R* of W are `Notify_Record` with `loop_tag == T`, and each **immediately follows a same-tag data task M\*** in ctx.nodes (unit tail record, meaning the folded producer task, for example memcpy folded from TransMemToMs; data tasks ⊆ `_QUEUE_DATA_TYPES`, not limited to Ub_Write_Or_Read. Memcpy/Reduce_Inline also have this pseudo-serialization);
   - **After W (in ctx.nodes order) there exists a same-tag data task** (mid-body determination; loop end wait has no same-loop data task after it, naturally exempt; next loop data tasks have different tags, also not mis-identified);
   - R* do not immediately follow non-data nodes (free set / sync slot local set **does not match**. These wait for real dependencies such as remote synchronization. Resolution is forbidden).
3. **Deletion and reconnection** (prune ctx.nodes, not post-assembly graph surgery):
   - **W always deleted**: remove from ctx.nodes and dep task table, revoke its pairing edge (`remove_match`) and aggregated extra edges (`extra_edges[W]`). This is the essential operation to break the barrier;
   - **R\* depends on consumers**: if R does not appear in any other wait's `wait_posts` values → delete together (in the sample db, set24 is only consumed by wt25, delete directly); otherwise **preserve R and only delete W** (its preserved edges are not affected, loop end wait waits normally. Timing-equivalent, semantically lossless conservative branch; taken when post bits are partially consumed by multiple waits);
   - **Successor unit reconnection happens automatically**: after pruning, assembly logic rescans ctx.nodes. Units that originally forked from W automatically fork to W's preceding skeleton node (the loopgroup predecessor task AA); when W is the skeleton head (no preceding skeleton), the unit becomes the entry unit. Entry dependencies are expressed by SQE attachment edges (Section 7.4.1). **No graph surgery for edge reconnection is needed**. Fork parent / skeleton chain /
      The completion set is naturally generated by existing assembly rules based on the pruned sequence.
    - Register / CKE accounting **zero change**: pairing consumption completes during emission, pruning is only a graph structure
      operation; wait matching after the loop uses unchanged accounting results (bits of deleted posts
      have been consumed by W, no residual waiters remain).
 4. **Iteration and statistics**: Repeat steps 2-3 until no new matches remain (handle multi-level chains within the body, such as
    prefetch->wait->middle segment->wait->final segment: after removing the first layer, the next layer becomes exposed as a new chain head);
    `CcuBuildStats` adds `pipeline_chains_broken`, the summary line displays `pb=N`.

Fixed sample task graph (compared to the beginning of this section; deleted nodes are record(0:1)/wait(0:1), the original
"wait(0:1) -> ub_write" gate is taken over by AA, timing is equivalent):

```
AA -> wait(0:ff)                                        <- loop end barrier (skeleton)
AA -> memcpy(mem->ms)                                   <- prefetch unit (record deleted)
AA -> multiple { ub_write_or_read -> record(0:xx) }     <- 7-way write, concurrent with prefetch
AA -> memcpy(ms->mem) -> record(0x80)                   <- copy-out unit, concurrent with prefetch
multiple { record(0:xx) / record(0x80) -> wait(0:ff) }  <- loop end waits for all data completion
```

#### Implementation Details (Code Placement)

- `simulator/task_model.py`: `Task` adds `loop_tag` slot (default empty string),
  `to_trace_args` outputs `loop_tag` (empty string -> `N/A`, same scope as `loop_cnt`);
- `ccu_graph_builder.py`:
  - `_MissionContext.__init__`: `_loop_tag = ""`, `_lg_seq = 0`, `wait_posts = {}`;
  - `_exec_loop_group`: record `lg_tag = f"loopgroup_{self._lg_seq}_loop_{cur_loop_idx}"` on entry (pass through to `_emit_merged_loop`), after LoopGroup processing completes
    `self._lg_seq += 1`;
  - `_emit_merged_loop`: set `self._loop_tag = lg_tag` within window (reset in `finally`, same pattern as `_loop_cnt`);
  - `_gen_waits`: after successful pairing `self.wait_posts[wait_task.task_id] = [p.task_id for p in posts]`;
  - `expand_ccu_graph` assembly loop beginning calls `_break_pipeline_chains(ctx, dep)`
    (returns deleted chain count, accumulates into `stats.pipeline_chains_broken`);
- `graph_dumper.py` / `simulator/report.py`: dump `nodes[]` and trace args
  add `loop_tag`; DOT/HTML legend sync (optional, displayed with loop_cnt).

Trace presentation (`export_trace`, display layer only, does not affect simulation): tasks carrying
`loop_tag` on virtual slave streams have their display line reassigned to the loop tag line of the current stream (line name is the tag, such as
`loopgroup_0_loop_0`), indicating they are transformed from loop-internal instructions; remaining tasks on the same slave stream
keep the original display line--mixed slave streams reused across missions therefore split into original line + tag line.
Display lines are partitioned by `(stream_id, loop_tag)`, tasks from different virtual streams do not merge lines (preserve
inter-slave-stream task partitioning); main stream tasks (including loop-end wait) all keep original lines; task
`args.stream_id` retains the real stream number.


```
_break_pipeline_chains(ctx, dep):
    broken = 0; changed = True
    while changed:
        changed = False
        for W in [n for n in ctx.nodes if type == Notify_Wait and n.loop_tag]:
            posts = ctx.wait_posts.get(W.task_id);  T = W.loop_tag
            if not posts:                              continue   # unpaired/remote
            if any R in posts: R.loop_tag != T:        continue   # cross-loop source
            if any R in posts: R's preceding ctx.nodes node is not same-tag data task: continue
                                                                 # orphaned set/sync
            if W has no same-tag data task after it:   continue   # loop-end barrier
            for R in posts:                                        # --- hit ---
                if R is in other wait's wait_posts value: continue # has retained consumer
                dep removes R; ctx.nodes removes R                  # unit tail now taken over by M
            dep.remove_match(W); dep.extra_edges pops W
            dep removes W; ctx.nodes removes W
            broken += 1; changed = True; break                     # rescan
    return broken
```

Acyclicity guarantee: pruning only deletes nodes and edges, adds no edges, and deleted W/R are all nodes within the emission window,
fork/join edges are recomputed by assembly based on the pruned sequence (Section 7.4.1 acyclicity argument continues to hold);
`_drop_cyclic_matches` (Section 5.4) and simulator runtime deadlock detection remain unchanged as fallback.

#### Solution Evaluation (Pros and Cons) and Alternatives

**Pros**:

1. Does not modify Section 6.5 merge semantics or checker alignment logic--post-hoc resolution, risk isolation;
2. Prunes ctx.nodes before assembly (rather than surgery on the formed graph result): fork parent / skeleton chain /
   completion set is naturally generated by existing assembly rules, avoids the correctness burden of "edge reconnection after node deletion" graph surgery
   (post-assembly surgery requires manual reconnection of fork edges, skeleton chains and completion sets, error-prone);
3. Records wait->posts pairing during emission, identification does not depend on formed graph dependency structure--insensitive to Section 7.4.1
   assembly rule evolution (fork parent selection, unit partition adjustments);
4. Loop tags limit resolution scope (same loop same mission), prevent cross-loop / cross-mission
   false deletion; tags also feed back to trace/dump tracing;
5. Deletes nodes rather than only deleting paired edges: avoids retaining a semantically dangling unpaired local wait (although its
   measured behavior uses `local_wait_delay` = 30ns, timing has no difference, but trace statistics scope is clean).

**Cons / Risks**:

1. Feature chain is a structural heuristic, identification conditions have coupling with emission / assembly logic, require maintenance as both evolve
   (coupling surface already compressed to ctx.nodes sequence layer through "emission-time recording + pre-assembly pruning");
2. `Task` model field expansion, trace/dump/JSON presentation requires sync (Chapter 11 scope);
3. "Full concurrency + bandwidth manager arbitration" is a pipeline **steady-state** approximation: ignores pipeline fill/drain
   delay (single iteration 4KB magnitude, sub-microsecond, negligible) and per-iteration handshake **rate** overhead
   (if CKE engine throughput is the bottleneck, a dedicated rate model is needed, current folding / deletion both count as single operations,
   consistent with pre-fix scope, registered in Section 10.3 #21);
4. Full expansion fallback path (`mf > 0`) does not go through this resolution--its per-iteration wait/record is real 4KB
   granularity handshake, but inter-iteration overlap is likewise serialized by queue chain; this path has large node count, current sample does not
   trigger it, kept as known residual (Section 10.3 #21).

**Alternative Solutions Comparison**:

| Alternative | Approach | Pros | Cons | Conclusion |
|-------------|----------|------|------|------------|
| B. Direct judgment during emission (no tags / no identification) | `_emit_merged_loop` directly identifies mid-body wait by slot structure (data slots still follow it) and skips emitting its wait/pairing | Simplest implementation, no Task field expansion; judgment information (slot order) is at hand, no heuristic | Loses handshake node and loop tracing in trace; skipping emission changes register pairing accounting (loop-end wait absorbs prefetch set, timing harmless but checker behavior divergence increases); orphaned set / sync scenarios need additional exclusion conditions | Viable simplified variant; selectable if trace presentation is not needed |
| C. Segmented pipeline modeling | Fold data tasks into K segments (8~16), retain wait/record between segments, explicitly model fill/drain | Contains pipeline fill effect, more precise | Node count x K, K value has no physical basis; under steady state differs from full concurrency only by first and last segment | Over-design, rejected |
| D. Simulator soft dependency edges | New edge type "pipeline edge" (child starts after parent completes X%) | Graph structure unchanged | Simulator semantics become complex, X has no physical basis; and serial chain structure (unit forks from wait) still requires graph layer modification | Rejected |

Conclusion: Adopt the original solution (tags + feature chain + deletion), strengthen per the above step 2/3 implementation location and safety
conditions (emission-time recording + pre-assembly pruning + mid-body judgment + consumer conservative branch).

**Implementation Status (Deployed)**: `Task.loop_tag` field (task_model / trace args /
dump `nodes[]` / input_builder circuit) + `DependencyGraph.remove_task()`
(together with cleaning extra edges and notify pairing) + `_break_pipeline_chains()` (called at assembly loop
beginning, counts `pipeline_chains_broken`, summary displays `pb=N`) + trace
loop slave stream rename presentation (`export_trace`, see above). Verification:
sample ALLGATHER db cross-engine scenario makespan 2657.521 -> 2561.541 us (56 -> 54
tasks, pb=1, 9 units all fork reconnected to previous skeleton node rwt22, consistent with manual prototype item by item);
archive library regression--same-engine phase scenario (mv->mv both HBM) makespan unchanged (zero-error scenario,
theoretical expectation), SQE-only library pb=0 not affected. `loop_tag` is retained as a persistent task attribute,
for subsequent performance simulation consumption (such as per-loop bandwidth/rate modeling, Section 10.3 #21).

### 6.6 Graph Formation Algorithm (Pseudocode)

```
expand_ccu(device_id, die_id, sqe_params, instr_space, dep_graph, reg_model):
    SQE grouped by contiguous segments into dispatch contexts (Section 6.1: non-contiguous = new dispatch, independent window)
    Same queue contexts are serial by dispatch order; each queue takes first context into active set
    while active set is non-empty:
        progress = False
        for q in active set's each queue current context:        # cross-queue round-robin
            while ptr[q] has not reached window end:
                instr = decode(instr_space[ptr[q]])
                if instr is WAIT and reg_model.cke bits are insufficient:
                    break                              # this queue pauses, yields to other queues
                nodes = transform(instr, reg_model)   # Section 6.3 mapping; LOAD only changes registers
                for node in nodes: attach to queue q's chain tail
                if instr is SET_CKE: reg_model.seen_post registers
                if instr is WAIT:     pair with seen_post -> aggregate wait node (Section 6.4)
                if instr is LOOP:     merge path (Section 6.5 verification) or repeat interpret loop body by iteration
                ptr[q] += 1; progress = True
            if ptr[q] reaches window end: current context completes, next context of same queue enters active set
        if not progress:
            # Stalled = wait exists that depends on remote CKE (cross-rank, invisible in single rank graph)
            For stalled queues: generate remote Notify_Wait node (notify_id+mask, fixed delay),
                then replay the remaining part of this instruction (local set / data transfer), continue advancing
    SQE stream attachment (Section 7.4.1): placeholder removed from SQE stream (task graph introduces no placeholder nodes);
        entry edges (SQE predecessor -> queue skeleton head) / exit edges (mission completion set -> SQE successor)
        expressed via extra edges
    Assembly partitioning (Section 7.4.1): each mission queue sequence -> skeleton (retains queue chain) + async units
        (attach to #v virtual stream, stream_id changed to point to virtual stream), fork (predecessor skeleton->unit head) /
        completion set join (skeleton tail+all unit tails -> SQE successor / same queue next mission entry)
        extra edges connect
```

- **Cross-rank post-wait does not build edges, but retains synchronization task nodes** (single rank graph formation, the main difference from checker):
  checker's `GenerateRawCcuGraphsV3` (`ccu_task_transform_v3.cc:1275-1406`)
  lets multiple rank CCU graphs advance alternately, cross-rank posts mutually unlock; this plugin only has single rank instruction data:
  - Instructions that post to remote CKE / remote register notifications -> generate remote `Notify_Record` node
    (src=local, dst=INVALID -> fixed `remote_record_delay`, does not register local CKE accounting,
    does not participate in pairing):
    - A5: SyncCKE's `rmtCKEId/locCKEMask`, SyncGSA/SyncXn's
      `setRmtCKEId/setRmtCKEMask`, TransLocMSToRmtMS's `setRmtCKE` (consistent with checker
      `AddPost` trigger conditions);
    - A6: TransMem `dmaOpCode=5` write-then-notify and `0x1A` remote atomic count,
      SyncXnWtx's `notifyValid=1` notify, SyncAtx's atomic sync--remote target type
      (Xn/CKE) is resolved by peer address mapping (checker `GetXnAndTypeIdByAddr`), single rank
      cannot obtain, all approximated as remote record (mask takes notification value low 16 bits for trace display only).
    SyncGSA/SyncXn (A5) and SyncXnWtx main write (A6) value sync is only remote register write
    (checker builds node/write state on peer graph), local side does not separately generate tasks.
   - Wait-required bits cannot be satisfied by local rank posts (stall detection finds this) -> that wait depends on remote -> generate
     remote `Notify_Wait` node (fixed `remote_wait_delay`, unpaired), and replay remaining instruction
     parts; unsatisfied waits within loop bodies likewise generate remote wait nodes then continue. Partial satisfaction: satisfied
     bits pair normally with local posts (aggregate into 1 local wait node, Section 6.4), unsatisfied bits generate one
     remote wait node. A6 independent Wait instruction (Xn value comparison) when unsatisfied uses same mechanism: top-level stall ->
     stall resolution generates remote `Notify_Wait` (task id tag `xwt`) then pass through.
   - Task graph does not magically produce any nodes: all nodes come from original db data or microcode instruction transformation
     (attachment dependencies expressed via extra edges, see Section 7.4.1).
- Local rank post-wait pairs normally: multiple queues of the same device, CKE of multiple dies
  are all in the local graph, build edges per Section 6.4 rules.

### 6.7 Simplified Comparison with checker CCU Graph Formation

**CCU task rank/eid/direction semantics**:

- **Channel-type network tasks** (write/read): local rank takes opDetails.rankId; peer rank is looked up through
  channel peer eid (reid) via eid->rank index (Section 4.8), if not found fill INVALID +
  `rank_unresolved` count. Equivalent to checker `GetRemoteRankByChannel`
  (`g_allRankChannelInfo`: channelId -> peer), but uses RankEidMap's communicator rank
  to replace checker's physical dstDeviceId.
- **Reduce tasks** (REDUCE microcode): MS is rank-local buffer (checker for all msId always
  `GenSliceFromMs` + `AddLocalBatchReduce`), reduce is always local HBM operation, src/dst both fill
  local rank; cross-rank data is already charged in preceding Rmt* role transfers, reduce does not repeat network charging.
- **Local tasks** (local transfer Memcpy, local reduce Reduce_Inline): src/dst both fill local rank.
- **is_write**: CCU-produced UB tasks all explicitly carry (write=True / read=False),
  does not depend on simulator rank heuristic.
- **Local rank self-verification**: if channel leid <-> index-resolved rank is inconsistent with opDetails.rankId,
  use opDetails as authoritative and warn (eid numbering is offset protection, such as deviceId and rankId having offset).

| checker stage | light_performance | Rationale |
|---------------|-------------------|-----------|
| Two-phase (raw graph formation -> queue parallelization -> reconnect) | Single-phase interpretive execution + assembly-stage queue parallelization partitioning (Section 7.4.1: `ApplyCcuQueueParallelizationV3` simplified port, skeleton/unit partition + fork/join and attachment extra edges + `#v` virtual streams) | No verification needed, no need to retain modifiable raw intermediate form; parallelization completes in one pass during assembly (reuse granularity differences see Section 7.4.1) |
| Loop merge (BatchTransMem/BatchReduce, resource conflict check, producedMasks balancing) | **Equivalent implementation** (Section 6.5: pre-verification + fold emission + fallback full expansion; memory slice merge not needed--simulation only cares about data volume and timing) | Align with checker semantics to eliminate node bloat |
| SQE grouping into TaskCcuGraph (`IsContiguousCcuSqe` contiguous then merge + `CcuMissionKey` latest node append) | **Equivalent implementation** (Section 6.1: contiguous segment partition into dispatch contexts, each context independent window replay + sequential args consumption; register/CKE shared across contexts, same queue contexts serial by dispatch order) | Align with checker multiple dispatch (window wrap resend) semantics, previously lost second dispatch data transfer due to merging into single context (Section 10.3 #20 fixed) |
| GetRemoteRankByChannel (channelId -> dstDeviceId/dstDieId, map rank via Communicator table) | Peer eid (reid) -> RankEidMap/ranktable index -> communicator rank (Sections 4.8/6.7) | estimate mode Communicator table only has local rank one row; ccuChannels.rankId field is endpoint deviceId not usable |
| AllRankParamRecorder complete address space mapping (XN/CKE/PFE/CHANNEL/JETTY/MISSION/MS ...) + HBM data content + remote address->register reverse lookup (`GetXnAndTypeIdByAddr`) | Only retain xn/cke/gsa/channel four types; A6 Load/Store HBM data plane and remote target reverse lookup not modeled (Section 6.4, `_MissionContextA6` deviation list) | Simulation only needs length and peer; data content/remote register number does not affect timing |
| A6 checker error abort scenarios (unsupported dmaOpCode/dstType, clearType!=1, clearMask!=0, LoopGroup config illegal, LoadX/StoreX indirect number out of bounds, etc.) | Downgrade to warning + skip data plane (sync semantics still execute) or skip instruction | Performance tool tolerates incomplete/abnormal data, maintain queue skeleton integrity |
| CcuTraceInfo tracing, ccu_convert_config.json generation | Removed | Verification/rewrite tool specific |
| CheckAllPostMatch error report | Downgrade to warning + diagnostic statistics | Adapt to single rank data |
| Cross-rank post unlocks other's wait (multi-rank joint advancement) | **Do not build cross-rank edges**: cross-rank post/wait generate remote sync nodes (fixed delay), peer readiness approximated by fixed delay | Single rank graph formation (core difference) |

### 6.8 A5 and A6 Instruction Set Differences Overview

| Dimension | A5 (V1/Ascend950) | A6 (V2/Ascend960) |
|-----------|-------------------|-------------------|
| LOAD type | 7 instructions: includes GSA series | 18 instructions: unified X register + LoadX/StoreX/ClearX/Nop + arithmetic (Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt) |
| CTRL type | 5 instructions: Loop/LoopGroup/SetCKE/ClearCKE/Jmp | 7 instructions: + SetCKBit/Wait/Fence |
| TRANS type | 12 instructions: Mem<->MS all combinations + SyncCKE/GSA/Xn | 7 instructions: local 4 types + unified TransMem (dmaOpCode distinguishes) + SyncWtX/SyncAtX |
| REDUCE type | Add/Max/Min (msId list addressing, <=8 MS) | ReduceAdd/Max/Min (same structure, no embedded waitCKE) |
| Loop register | v1.loop{xnId, startInstrId, endInstrId}, loopCnt in Xn | v2.loop{xpId(context), xnId, xmId(iteration count), mode}, mode==1 conditional expansion (Section 6.5) |
| LoopGroup parameters | Xn: LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41]; Xm: gsaOffset/MSOffset/ckeOffset | Xn: loopInsCnt:10/expandOffset:9/expandCnt:9; Xm: ckOffset:10/msOffset:11/gsaOffset:32; Xp: xnIdOffset:32 |
| Register number addressing | Payload field direct use | `GetXnId`: when bit15=1 offset by loop xnIdOffset (Section 6.4) |
| Embedded sync fields | TRANS/REDUCE commonly carry waitCKE + setCKE + clearType | Local TRANS/REDUCE only setCKE (wait handled by independent SetCKBit/ClearCKE/Wait instructions) |
| Channel/length value source | Channel is instruction immediate field | Channel/length both are Xn register values (`Xn[xcId]`/`Xn[xlId]`) |

Both instruction generations have been deployed in graph formation (A6 interpreter `_MissionContextA6` reuses A5's task emission/
CKE pairing/advancement/merge emission infrastructure, only overrides instruction decoding and loop organization); A6 instruction ->
task type complete table see Section 11.3.3, established deviations from checker listed in `_MissionContextA6`
class docstring and Section 6.7 comparison table.

---

## 7. Performance Simulation Module (Sub-requirement 4, incorporated per HCCLSIM baseline)

> This chapter is the design description for the `simulator/` sub-package (ported from `docs/light_performance/HCCLSIM/hccl_sim/`),
> interface uses HCCLSIM as authoritative baseline. Differences from old version and residual items see Chapter 10.

### 7.1 Overview

```python
sim = HcclSimulator(config, dep_graph, topo_resolver)
result = sim.run(trace_filepath=None, max_trace_events=0)   # -> dict
```

The simulator accepts three parameters:

| Parameter | Type | Content | Builder |
|-----------|------|---------|---------|
| `config` | `SimConfig` | Delay, bandwidth, utilization configuration | Caller |
| `dep_graph` | `DependencyGraph` | Task nodes + dependencies | `graph_builder` (db path) / `input_builder` (JSON path) |
| `topo_resolver` | `TopoResolver` | eid -> physical port resolution (ranktable.json) | Caller |

`run()` returns a result dict (no longer returns raw makespan):

```python
{
    "makespan": float,        # end-to-end duration (seconds)
    "makespan_us": float,     # (microseconds)
    "trace_file": str|None,   # trace output path
    "trace_events": int,      # trace event count
    "sim_time_s": float,      # software execution time (seconds)
    "tasks": list[Task],      # all tasks (including start/end_time)
    "tasks_by_stream": dict,  # int stream id -> [Task] (chain order)
    "ports": dict,            # tuple(ports) -> NetworkPort
}
```

The simulator does not accept task sequence lists, does not infer dependencies--all obtained from `dep_graph`.

### 7.2 Parameter One: `config: SimConfig`

**Unit convention**: internal delay=seconds, bandwidth=B/s; **JSON configuration unit delay=us, bandwidth=GBps**
(`from_dict`/`to_dict` auto-convert). Port identifier is `"die_id/port_number"` string
(such as `"1/0"`), consistent with ranktable.json.

#### Field Definitions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `local_record_delay` | float | `1e-6` | Local record fixed duration; JSON value unit us |
| `remote_record_delay` | float | `1e-6` | Remote record fixed duration; JSON value unit us |
| `remote_wait_delay` | float | `1e-6` | Remote wait / unmatched local wait fixed duration; JSON value unit us |
| `port_bandwidth` | dict | `{}` | `"die/port"` -> bandwidth (B/s). One eid may correspond to multiple ports, configure per port individually, total bandwidth is sum |
| `default_port_bandwidth` | float | `53.125e9` | Default bandwidth for unconfigured ports (425 Gbps = 53.125 GBps); JSON value unit GBps |
| `hbm_bandwidth` | float | `1.4e12` | HBM bandwidth, used for Reduce_Inline/Memcpy; JSON value unit GBps |
| `bw_profile_set` | BandwidthProfileSet? | `None` | Bandwidth utilization configuration set, when None utilization is 100% |

#### Port Bandwidth Configuration Format

```json
{"port_bandwidth": {"1/0": 53.125, "0/7": 53.125}}
```

#### Bandwidth Utilization Configuration (`bw_profile_set`)

```
BandwidthProfileSet
+-- profile_low:   BandwidthProfile  <- used when shared port concurrency < 8
+-- profile_high:  BandwidthProfile  <- used when shared port concurrency >= 8
+-- THRESHOLD: 8
```

Each `BandwidthProfile` is a data volume (32B~128MB) -> utilization (0~1) mapping table, query rounds down to
the largest bin not exceeding that data volume (floor match).

JSON format:
```json
{
  "profile_low":  {"32": 0.30, "64": 0.35, "...": 0.90, "134217728": 0.95},
  "profile_high": {"32": 0.20, "64": 0.25, "...": 0.84, "134217728": 0.85}
}
```

### 7.3 Parameter Two: `dep_graph: DependencyGraph`

#### Four-layer Content

```
DependencyGraph
+-- tasks: dict[str, Task]                       <- task nodes (with all parameters)
+-- stream_chains: dict[int, list[str]]          <- serial dependencies (int stream id)
+-- notify_matches: dict[str, str]               <- notify dependencies (wait_task_id -> record_task_id)
+-- extra_edges: dict[str, list[str]]            <- additional dependency edges (child_task_id -> [parent_task_id, ...])
                                                  <- sources: CCU queue parallelization fork/join edges and SQE attachment edges (Section 7.4.1)
```

#### Layer One: Task Nodes (`Task`)

Each Task carries all parameters needed for execution (invalid values uniformly use `INVALID = 0xFFFFFFFF`,
eid is empty string, string attributes are `""`):

| Field | Type | Applicable Task Types | Simulator Purpose |
|-------|------|---------------------|-------------------|
| `task_id` | str | All | Dependency graph index |
| `task_type` | TaskType | All | Determines delay model (internal classification, Section 7.3) |
| `task_name` | str | Display use | trace/dump display name: profInfo.profTaskType op name (`_add_task` unified fill, Section 11.4.1); when empty `display_name` falls back to `task_type.value` (CCU subtasks/CLI input) |
| `stream_id` | **int** | All | Serial chain membership |
| `notify_id` | int? | Notify_Record/Wait/Write_With_Notify | Dependency pairing identifier (CCU path = cke id); Write_With_Notify display only (its notify faces peer, does not participate in pairing) |
| `mask` | int | Notify_Record/Wait | notify bitmask (CCU path): notify_id + mask uniquely identify one notify action; wait node carries aggregated mask (all contributing post consumption bits OR'd, Section 6.4); SQE path and 0=unfilled |
| `src_rank` / `dst_rank` | int? | notify/transfer | Local judgment (src==dst); read direction local judgment |
| `size` | int | Memcpy/Reduce_Inline/Ub_*/Write_With_Notify | Calculate delay + look up utilization |
| `port` | str/list? | (reserved field) | Port resolved by eid at runtime, db path does not fill |
| `src_eid` / `dst_eid` | str | Ub_*/Write_With_Notify | eid -> ranktable -> physical port |
| `transport_type` / `link_type` | str | Display use | msprof reconciliation dimension (profInfo transportType/linkType name, Sections 11.4.2/11.4.3, `_add_task` unified fill; CCU subtasks not filled) |
| `data_type` | str | Display use | trace/report annotation |
| `protocol` | str | Display use | Channel CommProtocol name (Section 11.4.2 note; empty when no channel info) |
| `jetty_id` | int? | Ub_*/Write_With_Notify | trace annotation |
| `is_write` | bool? | Network tasks | Charging direction and port query eid selection (Section 10.3 #1) |
| `start_time` / `end_time` | float | All | Simulator fills (-1=not executed) |

7 TaskTypes:

| TaskType | Delay Model | Description |
|----------|-------------|-------------|
| `NOTIFY_RECORD` | Fixed delay (local also triggers Event) | Local = `is_local_notify`(src==dst); unpaired local record only fixed delay |
| `NOTIFY_WAIT` | Wait for Event / fixed delay | matched->wait for record's Event; unmatched/remote->fixed `remote_wait_delay` |
| `REDUCE_INLINE` | HBM dynamic sharing | `size / hbm_bandwidth`, shares bandwidth with other HBM tasks |
| `MEMCPY` | HBM dynamic sharing | Same as above (HBM copy, not network transfer) |
| `UB_INLINE_WRITE` | Fixed delay (same as Notify_Record: local/remote record_delay) | notify inline write; SQE graph formation path does not produce, only old input path (task.txt) uses |
| `UB_WRITE_OR_READ` | Port bandwidth sharing (write/read two directions charged independently) | Network data send/receive; `is_write` explicitly carried by graph formation (SQE network tasks both directions uniformly map to this type) |
| `WRITE_WITH_NOTIFY` | Port write direction sharing + write-then-notify fixed delay | Network write + attached notify single task, one-to-one correspondence with db `WRITE_WITH_NOTIFY` (7); notify uses fixed remote record delay (peer wait not in this db) |

#### Layer Two: Serial Dependencies (`stream_chains`)

```
int stream id -> [task_id_0, task_id_1, ..., task_id_n]
```

- List order = execution order: task[i] must complete before task[i+1] can execute
- Each stream_id starts one simpy process, executes in chain order by `yield`
- Different streams are naturally parallel
- db path stream id addressing: SQE stream directly uses original streamId; CCU queue chain and virtual stream start from base `1<<20`
  allocation (Section 6.5, virtual stream logical name carries `#v{n}` suffix), logical name retained via `stream_labels`;
  `GraphBuilder.stream_meta` provides real/ccu/virtual three-category classification and virtual stream
  parent_sid (Section 7.4.1) on top of this, for report/trace/dump annotation

#### Layer Three: Notify Dependencies (`notify_matches`)

```
wait_task_id -> record_task_id
```

Explicit 1:1 pairing: which wait waits for which record. When multiple records exist for the same notify_id, only one
pairs with the wait, unpaired records use fixed delay.

#### Layer Four: Extra Dependency Edges (`extra_edges`)

```
child_task_id -> [parent_task_id, ...]
```

- Semantics: child must wait for **all** parents to complete before starting (pure timing constraint, does not carry
  record/wait event semantics--event pairing still expressed by `notify_matches`).
- Current sources: CCU queue parallelization fork edges (predecessor skeleton -> unit head), mission completion set
  join edges (skeleton tail + unit tail -> SQE successor / same queue next mission entry), CCU subgraph
  attachment edges on SQE stream (entry/exit, see Section 7.4.1), and aggregate wait node multi-parent edges
  (wait -> all contributing posts, Section 6.4--first contributing post uses `notify_matches`, remainder use this layer).
- Write side `add_edge` auto-deduplicates (same child-parent pair only records one edge); simulator queries via
  `get_extra_parents` / `extra_parent_ids`.
- Acyclicity: above edges always connect "forward" along original instruction order (skeleton before unit, unit before join point,
  placeholder position front and back attachment, post registered before wait instruction that consumes it), graph formation direction guarantees acyclicity,
  and included in `_drop_cyclic_matches` cycle detection fallback (Section 5.4).

### 7.4 Simulator Consumption Flow

```python
def run(self, trace_filepath=None, max_trace_events=0):
    # 1. Retrieve all task nodes
    tasks = list(dep_graph.tasks.values())

    # 2. Create sync primitives: only matched wait creates Event
    for wait_task_id in dep_graph.matched_wait_ids:
        notify_events[wait_task_id] = env.event()
    for parent_id in dep_graph.extra_parent_ids:  # all parent tasks of extra edges
        task_done_events[parent_id] = env.event() # must pre-build before stream process starts:
                                                  # if parent completes first, child builds Event later will wait forever (Section 7.4.1)

    # 3. HBM bandwidth manager (Reduce_Inline + Memcpy shared, utilization always 1.0)
    hbm_mgr = _BandwidthManager(env, config.hbm_bandwidth, empty utilization table)

    # 4. Start one simpy process per stream, execute serially in chain order
    for task in chain:
        for pid in dep_graph.get_extra_parents(task.task_id):
            yield task_done_events[pid]           # extra edges (Section 7.4.1), start_time does not include wait
        if NOTIFY_RECORD:
            yield timeout(calc_notify_record_delay(task, config))
            if local and matched: trigger matching wait's Event
        elif NOTIFY_WAIT:
            if local and matched: yield notify_events[task_id]   # wait for Event
            else:              yield timeout(remote_wait_delay)
        elif REDUCE_INLINE / MEMCPY:
            yield hbm_mgr.send(task_id, size)                 # HBM dynamic sharing
        elif UB_INLINE_WRITE / UB_WRITE_OR_READ / WRITE_WITH_NOTIFY:
            is_write = task.is_write  # explicit direction; old input path None falls back to heuristic
            eid = src_eid if is_write else dst_eid
            ports = topo_resolver.get_ports_by_eid(eid)       # missing then throw error
            port = lookup/create NetworkPort(ports, total bandwidth)
            yield port.send(task_id, size, is_write)          # port bandwidth sharing
            if WRITE_WITH_NOTIFY:
                yield timeout(calc_write_with_notify_delay(task, config))
                                                            # notify after write lands:
                                                            # fixed record delay
                                                            # (peer wait not in this db)
        task.end_time = env.now
        if task_id in task_done_events: task_done_events[task_id].succeed(env.now)
    # 5. makespan = max(end_time); assemble result dict (including trace export)
```

Delay formulas unified and maintained at single point in `task_timing.py`.

#### 7.4.1 CCU Queue Parallelization and Virtual Streams (extra edges)

**Problem**--CCU queue chains produced by Sections 6.5/6.6 are per-task serial streams: all
data transfers, waits, sets of the same queue execute in order. On real hardware, queue async transfer units execute concurrently,
serialization causes triple distortion:

| Distortion | Mechanism |
|------------|-----------|
| makespan overestimation | Async transfers separated by skeleton (wait/set) one by one, cannot overlap |
| Utilization bin distortion | Same queue concurrent transfers degrade to sequential execution, `_BandwidthManager` observed port concurrency is always 1, always queries profile_low bin (Section 7.5) |
| Port peak concurrency statistics distortion | `peak_concurrency` always 1, cannot reflect concurrent bandwidth sharing behavior (Section 8.2 port-level metrics) |

**checker reference**--`ApplyCcuQueueParallelizationV3` (`ccu_task_transform_v3.cc:1052`):
secondary traversal modification after raw graph formation, `CollectAsyncBoundRecordNodesV3` collects async transfer nodes,
`QueueIdReuseAllocatorV3` reuses virtual queue id by `[startNodeId, afterNode]` interval,
fork/join edges reconnect to skeleton.

**Simplified port** (`ccu_graph_builder.py` assembly loop, completes in one pass during assembly, no raw
intermediate form)--partition each mission's queue node sequence:

1. **Unit identification**: `_QUEUE_DATA_TYPES` (`Memcpy`/`Ub_Inline_Write`/
   `Ub_Write_Or_Read`/`Reduce_Inline`) data tasks + immediately following contiguous
   `NOTIFY_RECORD` chain = one **async unit**; remaining nodes (wait / orphaned set /
   remote sync nodes) are **skeleton**, remain in queue serial chain--queue chain is the skeleton itself,
   no head/tail placeholder nodes introduced.
2. **fork edges**: `add_edge(unit head, previous skeleton node)`--unit must wait for queue to advance
   to its predecessor skeleton before starting (adjacent units share the same fork parent node); units appearing before skeleton
   (mission first node is data task) have no skeleton predecessor, their entry dependency is expressed by attachment edges instead.
3. **Completion set join**: mission completion set = skeleton tail + all unit tails, joins via extra edges to
   --same queue next mission's entry (skeleton head + pre-skeleton unit head, queue serial semantics;
   skeleton tail order already covered by queue chain serialization) and SQE stream successor tasks (exit edges,
   added during graph_builder attachment stage).
4. **Virtual stream reassignment**: per queue one set of counters/reuse pools (`vstream_seq`/
   `vstream_free`/`vstream_busy`), unit task `stream_id` reassigned to point to
   `ccu_d{die}_q{queue}#v{n}` (same `1<<20` base address space), unit chain registered into
   `result.virtual_chains` and `add_stream_chain`--serial chain only constrains skeleton and
   unit-internal task order.
5. **SQE stream attachment** (`graph_builder._expand_ccu` consumes `result.attach`):
   CCU placeholder removed from SQE stream chain (task graph does not produce any placeholder/phantom nodes),
   entry edge = SQE predecessor task (including immediately preceding CCU mission completion set) -> skeleton head and
   pre-skeleton unit head; exit edge = mission completion set -> SQE successor task. All expressed via extra edges,
   timing semantics: SQE predecessor (and preceding mission completion set) must complete before subgraph can start,
   subgraph mission completion set must all finish before SQE successor task can start.
   Remove empty chains for SQE streams where all tasks are placeholders.

**CKE timing preservation**: partitioning does not modify `add_match` pairing--which unit record falls into, wait
remaining in skeleton does not affect event pairing (wait waits across streams via notify_events). In measured sample,
trailing memcpy unit's fork parent wt28 happens to be on the accounting wait of the preceding read unit's tail set,
cross-stream event + fork edge together guarantee "read complete -> trailing copy" original timing. **Exception**: merged loop
emission window mid-body wait/record pairing is folded pseudo-serial barrier (per-iteration pipeline handshake is
folded into full barrier), resolved by Section 6.5.1 feature chain and pruned before assembly--this section's partition is based on pruned
sequence, skeleton/unit/fork structure is not affected.

**Differences from checker reuse rules**:

| Dimension | checker (V3) | light_performance |
|-----------|--------------|-------------------|
| Reuse granularity | afterNode interval (join successor node) level | mission level (all preceding released when entering new mission) |
| Intra-mission reuse | Allowed (DAG explicit edges, no pseudo dependencies) | No reuse (same mission units are concurrent with each other, reuse on same stream in chain model would introduce pseudo serialization) |
| Residual strictness | GRAPH_UNMATCHED error exit | Tolerated (performance tool, Section 6.4 downgrade to warning) |

Cross-mission reuse safety: units reusing the same virtual stream are serial within that stream chain (next mission
unit chain follows previous mission unit chain), no time axis overlap.

**Acyclicity guarantee**: fork (skeleton -> unit head) and completion set join edges (unit tail -> SQE successor /
next mission entry) always connect "forward" along original instruction order, partitioning does not change original order;
`_drop_cyclic_matches` cycle detection also includes extra edges (Section 5.4) as fallback; simulator
also has runtime deadlock detection (error exit when task `end_time<0`).

**Statistics and tagging**: `CcuBuildStats` carries parallelization (`parallel_units`/`virtual_streams`/
`fork_edges`/`join_edges`--join_edges now counts completion set join and SQE attachment edges), remote sync
(`local_posts`/`remote_posts`/`wait_matched`/`remote_waits`) and other counts,
summary line displays `pu/vs/lp/rp/wm/rw`;
`GraphBuilder.stream_meta` classifies by logical stream name--`s{n}` -> real, contains `#` ->
virtual (with parent_sid), remainder -> ccu, for three consumption points: report (`Stream count:
N (including virtual M)`, virtual stream annotated with `vir_stream {n} (label)`, n numbered from 0 by allocation order,
detail folded when exceeding 64 streams), trace (virtual stream thread_name `vir_stream
{n} ({label})` + task args `virtual_stream: true`), dump (streams[] with
kind/virtual/parent_sid,
extra_edges list; DOT purple dashed cluster + blue dotted extra edges, HTML badge
+ dotted arcs). stream_meta default (simulator standalone CLI old input) not annotated, backward compatible.

**Quantified effect** (sample db REDUCE_SCATTER iter0, 66 tasks): queue parallelization makes makespan
20,404.20 -> 5,149.36 us (~3.96x); subsequent two corrections further reduce to **2,753.71 us**:
(1) Remote MS reduce correction (Section 6.3.2, reduce always local HBM)--original second phase 128MB reduce charged as
network read ~2526 us, corrected to HBM ~96 us; (2) Cross-rank sync node completion (Section 6.6)--
14 waits depending on remote set were previously skipped (read unit emitted early at t~0), corrected to use fixed
remote delay simulation. 10 async units reassigned to 10 virtual streams (fork edges 10, attachment/join edges generated per
SQE neighbors and queue serial conditions)--first phase 8 units concurrent (7 network read units each
~2526 us, belonging to 7 independent ports `1/0`..`1/6` + 1 local memcpy ~96 us), then
reduce ~96 us and trailing memcpy ~96 us serial connection. Note single port read peak concurrency remains
1--benefit comes from **cross-port overlap**, not single port concurrency improvement.

### 7.5 Bandwidth Sharing Mechanism (`_BandwidthManager`)

#### General Manager (port and HBM reuse same model)

- **Single manager process model**: when new transfer joins, `interrupt` the manager process, manager process evenly divides bandwidth by current concurrency N:
  `bw_per = total bandwidth / N`.
- **Utilization query baseline is `min_remaining`**: same group of concurrent tasks uses "minimum remaining data volume" to query
  utilization table (concurrency <8 uses profile_low, >=8 uses profile_high), same baseline within group.
- Effective bandwidth = `(total bandwidth / N) x utilization(min_remaining, N)`; calculate earliest completion time,
  timeout settle progress; recalculate when join/complete occurs.
- **HBM dynamic sharing**: `Reduce_Inline` and `Memcpy` share one HBM manager
  (utilization always 1.0)--when multiple HBM tasks are concurrent, evenly divide HBM bandwidth (differs from old version per-task independent
  `size / hbm_bandwidth`, see Section 10.3).

#### Bidirectional Independence

Each `NetworkPort` contains write and read two independent managers, bandwidth independent and non-interfering:
when configuring port bandwidth B, write direction has B available, while read direction also has B available.

#### Multi-port Aggregation

When one eid corresponds to multiple physical ports, `port_key = tuple(port_ids)` as logical port,
total bandwidth = sum of each port's bandwidth (`get_ports_total_bandwidth`).

### 7.6 Parameter Three: `topo_resolver: TopoResolver`

- Input `ranktable.json` (`TopoResolver(ranktable_path=...)`), provides
  **eid (16B hex string) -> physical port list (`["die/port", ...]`)** resolution.
- Transfer task routing: write direction queries `task.src_eid`, read direction queries `task.dst_eid`.
- **Throws error directly when eid cannot resolve to ports** (no jettyId fallback, no default port)--therefore db path
  entirely skips simulation when "network tasks exist but no ranktable" (Section 3.3), db graph formation counts and warns on tasks with unresolvable
  eid (Section 5.5).
- Port bandwidth queries `SimConfig.port_bandwidth` per resolved port and sums.

### 7.7 Build Methods and CLI

All input methods uniformly build `(config, dep_graph)`:

| Input Source | Build Function | Returns | Status |
|--------------|---------------|---------|--------|
| **hccl-vm db (newly added)** | `graph_builder.build(group)` | `(config, dep_graph)` | Chapters 5, 6, main input |
| JSON task file | `load_dep_graph(filepath)` + `load_config(...)` | `(config, dep_graph)` | Retained (hccl_sim compatible) |
| task.txt / algorithm generated and other old input sources | -- | -- | **Removed with HCCLSIM version** (see Section 10.3) |

Build flow unified as:
1. Create `Task` nodes, call `dep_graph.add_task(task)`
2. Call `dep_graph.add_stream_chain(stream_id, task_ids)` in task order within stream
3. Call `dep_graph.add_match(wait_task_id, record_task_id)` per notify pairing result

CLI entry points:

```bash
# db input (light_performance main entry, see Section 3.3 for details)
python3 -m light_performance --db data/hccl_vm_data.db \
    --config sim_config.json --bw-profile bw_profile.json --ranktable ranktable.json -v

# One-click task graph dump (repository root)
python3 dump_graph.py data/hccl_vm_data.db --html

# hccl_sim compatible entry (JSON task file)
python -m simulator --tasks tasks.json --config sim_config.json \
    --bw-profile bw_profile.json --ranktable ranktable.json -v --trace trace.json
```

`--config` overrides default delay/bandwidth per mode (units us/GBps); `--bw-profile` independently loads
utilization configuration; report and trace generated by `report.py`'s `generate_report` / `export_trace`
from `run()` result dict.

Trace file is Chrome Trace Event format: `ts`/`dur` value units are **us** (format specification convention),
file top level carries `"displayTimeUnit": "ns"` specifying viewer (chrome://tracing etc.)
Start/Wall Duration **display** unit as ns (default ms).

`export_trace` display adaptation conventions:

- **tid sequential numbering**: thread rows mapped by stream id ascending to tid `0,1,2,...` (original
  streamId 2/6..13 used directly as tid, string-sorting viewers would place 10..13 before 2);
  real stream id retained in thread_name label and `args.stream_id`.
- **Endpoint rounding**: `ts`/`end` each rounded then subtracted to get `dur`--`ts` and `dur` each independently rounded
  would produce 0.001us pseudo overlap, same tid overlap slices would be nested by viewer as sub-rows (displayed on new line).
- **Virtual stream annotation** (Section 7.4.1): db path via `export_trace(..., stream_meta=...)`
  passes stream metadata, virtual stream thread_name changed to `vir_stream {n} ({label})` (n numbered from 0 by
  sid ascending i.e. allocation order, eliminating 2^20 base large-value stream numbers; real stream number see
  in-line task `args.stream_id`), its task args append `virtual_stream: true`;
  real/ccu stream names unchanged (ccu queue chain logical name only appears in text report lines). stream_meta
  default (simulator standalone CLI old input) not annotated, backward compatible.
- **notify task args carry mask**: CCU path `Notify_Record`/`Notify_Wait`
  task args append `mask: "0x..."` (notify_id + mask uniquely identify one notify action,
  Section 7.3); SQE path and mask=0 does not output this field.
- **Operator summary info first line (op_summary)**: pipeline per-operator call
  `export_trace(..., op_summary=...)`, each operator (when multiple operators merged export, per
  pid process group) retains `op_summary` first line with tid 1 + `thread_sort_index` 0, task row
  tid and sort_index continue sequentially (presentation layer only; tid avoids 0--some viewers like Perfetto treat
  tid 0 as idle thread, `thread_sort_index` is Chrome Trace standard row order control),
  summary events placed at event array front, triple guarantee op_summary is always first row. First line places one
  `X` summary slice spanning operator start to end (start time and duration are the slice's own ts/dur, taking
  earliest task start to latest end), row label `"<op_name> | <algorithm_name>"`,
  args carry: `algorithm_name` (algorithm selection result, opDetails.algorithmName log
  interception backfill (DFX main path, Section 4.2 note)), `data_type`/`data_count` (element count, V2 layout operator is
  send/receive both sides element count and type, others are count and single-side type), `task_count` (post-graph-formation
  task total, CCU mode is microcode instruction translated task graph count), and op_type/
  comm_id/op_iter/rank/rank_size/input_bytes/output_bytes (opMemInfo
  inputSize/outputSize, calculated from business API parameters during collection) context; summary events not counted in
  `max_events`.

### 7.8 Alignment Adaptation Completion Status

Old version (pre-modification in this repository) differences from HCCLSIM baseline have been resolved in this alignment:

| # | Item | Treatment |
|---|------|-----------|
| 1 | Task model 6 types -> new 6 types (local/remote record/wait merged into Notify_*, LOCAL_COPY split into Memcpy/Reduce_Inline, network transfer changed to Ub_*; later with db added WRITE_WITH_NOTIFY expanded to 7 types, Section 10.3 #19) | `task_model.py` replaced per HCCLSIM |
| 2 | stream_id str -> **int**; notify_id str -> **int** (INVALID=0xFFFFFFFF) | `dependency_graph.py`/graph formation side synced |
| 3 | Port specified by `port` int field -> **eid runtime resolution** (write queries src_eid / read queries dst_eid) | `topo_resolver.py` + graph formation side fills eid |
| 4 | Unit system: JSON delay s -> **us**, bandwidth B/s -> **GBps**, port key int -> **"die/port"** | `config.py` from_dict conversion |
| 5 | HBM delay `size/bw` independent -> **dynamic sharing** (Reduce_Inline+Memcpy share manager) | `_BandwidthManager` reuse |
| 6 | `run()` returns makespan -> **result dict**; simulator two params -> **three params** (+topo_resolver) | `simulator.py` |
| 7 | Delay formulas scattered -> `task_timing.py` single point | Incorporated |

Residual unmatched items (including `src_rank==0` hardcode, eid no fallback, broadcast pairing only triggers first wait, etc.)
uniformly moved to Chapter 10 for tracking.

---

## 8. Performance Profiling Output (Sub-requirement 5, Summary)

### 8.1 Output Location and Files

```
<installRoot>/plugin/light_performance/data/<timestamp>/
+-- perf_profiling.json      # main artifact: performance metrics + task graph summary
+-- trace_<commId>_<opIter>.json   # (optional) Chrome tracing, chrome://tracing visualization
+-- light_performance.log    # run log
```

- Directory naming: `hccl-vm plugin run @estimate` one-time execution path does not pass data_id (Section 3.2),
  uses run timestamp as substitute; standalone debug mode (run_perf_sim.py) specifies output path via `-o`.
- Output fixed in plugin data directory rather than `data/`: host machine clears `data/` on every start (`src/main.cc:32`),
  artifacts under plugin directory can persist across sessions; `archive.sh` archive only covers logs/ and data/, plugin artifacts
  do not participate in auto-archive (Section 10.3 #15).

### 8.2 Metric Definitions (Draft)

| Level | Metric | Source |
|-------|--------|--------|
| Operator level | opType/opIter/rankSize/dataSize, makespan, software simulation time | opDetails + `sim.run()` result dict |
| Operator level | Operator summary info first line (algorithm_name/data_type/data_count, start/end and duration carried by slice ts/dur, Section 7.5 op_summary) | opDetails.algorithmName (DFX/log dual-path interception) + simulation task timeline |
| Operator level | Total transfer volume, effective bandwidth utilization (total transfer volume / (makespan x port bandwidth)) | Summarize result["tasks"] |
| Stream level | Each stream start/end time, duration, task count + virtual stream annotation (Section 7.4.1) | result["tasks_by_stream"] + stream_meta |
| Graph level | Async unit count, virtual stream count, fork edge count, completion set/attachment edge count (Section 7.4.1); local/remote post count (lp/rp), local wait matched count (wm), remote wait count (rw) | CcuBuildStats (parallel_units / virtual_streams / fork_edges / join_edges / local_posts / remote_posts / wait_matched / remote_waits, summary line `lp/rp/wm/rw`) |
| Port level | Each port peak concurrency, write/read transfer volume and average bandwidth | result["ports"] (NetworkPort) |
| Task level | Per-task start/end/duration, transfer task average bandwidth | result["tasks"] (existing trace logic) |

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

> `cross_rank_notify` is the retained cross-rank notify node count (diagnostic information): these nodes are formed normally per stream order,
> use fixed remote delay, but do not participate in local pairing (no cross-rank record-wait edges); this count is used to troubleshoot
> data/classification anomalies (such as local notify being misclassified as cross-rank). `kind`/`virtual_stream_count`/
> `extra_edge_count` are CCU queue parallelization (Section 7.4.1) newly added fields: stream classification real/ccu/virtual,
> virtual stream count and extra edge (fork/join/attachment) count.

---

## 9. Appendix

### 9.1 checker Graph Formation Module -> light_performance Graph Formation Module Complete Comparison

| Aspect | checker (C++) | light_performance (Python) |
|--------|---------------|---------------------------|
| Goal | Correctness verification DAG (mainStart reachable, supports various checks) | Simulation DependencyGraph (three layers: tasks / stream_chains / notify_matches) |
| Input | db (via StorageManager/sim::Query) | Same db (sqlite3 direct read + struct decode) |
| Graph formation scope | **Multi-rank joint graph formation** (all rank tasks enter same DAG) | **Single rank graph formation** (cross-rank does not build pairing edges, remote sync tasks retained) |
| SQE translation | TaskMetaTranslatorV3 (includes MemSlice address resolution) | Section 5.2 mapping table (only takes len/jettyId/notify fields) |
| WRITE_WITH_NOTIFY consumption | Decode boundary split: `DecodeAndExpandTaskMeta` expands into [write, NOTIFY_RECORD] two subtasks | **No split**: single task mapping `Write_With_Notify` (Section 5.2), one-to-one correspondence with db tasks, trace/dump/statistics granularity consistent |
| Dependency edges | stream chain + local/cross-rank notify edges (queue simulation) | stream chain + only local notify first-come-first-served (two-pass scan) + extra edges (CCU fork/join/attachment, Section 7.4.1) |
| CCU | Two-phase expansion + queue parallelization + loop merge + CKE accounting | Single-phase interpretive execution + assembly-stage queue parallelization (Section 7.4.1) + loop merge (Section 6.5) + CKE accounting (Chapter 6) |
| Post-verification processing | Memory conflict/semantics/sync conflict/reachability/single-task check | None |
| Output | Verification conclusion + dump | (SimConfig, DependencyGraph) -> makespan -> profiling json |

### 9.2 Glossary

| Term | Description |
|------|-------------|
| SQE | Submit Queue Element, task unit submitted by AICPU/SDMA side, i.e. one row of opTask table |
| WQE | Work Queue Element (UB/DB mode), not involved in this work |
| notify | AICPU side event sync primitive, record sets bit / wait waits, paired by (recordDev, waitDev, notifyId) |
| CKE | CCU side event sync register (bitmap), post sets bit / wait waits, paired by (waitDev, dieId, ckeId, mask) |
| jetty | Network card (communication engine) port number, determines memcpy bandwidth attribution |
| die | Chip die, CCU microcode space and CKE both organized by (device, die) |
| mission | CCU one task dispatch (instStartId/instCnt define instruction window, args[13] passes parameters) |
| makespan | Task graph end-to-end duration (latest task end time) |
| Main stream/Slave stream | Operator main stream marked by opDetails.streamId; others are slave streams (first WAIT last RECORD structure) |

---

## 10. HCCLSIM Alignment Residual Items and To-Match List

> This chapter records **field gaps,
> task type mismatches and other conflict points** confirmed during this "db graph formation path + design document" alignment to HCCLSIM baseline.
> Current implementations all deployed per "default fill invalid values / approximate mapping",
> documented here for subsequent joint resolution with collection side and simulation module item by item.

### 10.1 Task Field Availability Table (db Graph Formation Path)

New model Task field sources and defaults in db graph formation path (invalid value convention: int fields `INVALID=0xFFFFFFFF`,
eid empty string, other strings `""`):

| Task Field | db Source | Availability | Default/Approximate Value | Subsequent Match Path |
|------------|-----------|--------------|--------------------------|----------------------|
| `task_id` / `stream_id` | Global sequence / streamId+synthetic base | Available | -- | -- |
| `notify_id` | opTaskMeta `NotifyTask.notifyId` | Available | -- | -- |
| `size` | MEM_CPY `len` / REDUCE `dataCount` / CCU length register | Available | -- | -- |
| `jetty_id` | SQE table column / ccuChannels.jettyId array | Available | INVALID (non-network tasks) | -- |
| `link_type` | opTaskMeta `HcclTaskProfInfo.linkType` (v3, enum name mapping Section 11.4.2) | Available (v3) | "" (v2 old BLOB without profInfo displays `Unknown`) | -- (Section 11.4 consumed) |
| `protocol` | ccuChannels.protocol (CommProtocol enum name mapping) | Available | "" (when no channel info, trace displays `N/A`) | -- (migrated from `link_type`, Section 11.1 plan a) |
| `src_eid` (local, write) | SQE `lcEid` (v2 collection newly added, Section 4.4.1) priority; default jettyId -> local device outbound channel `leid` derivation | Available (v2)/approximate (fallback) | empty->any local channel leid fallback | -- (v2 already supplemented explicit field) |
| `dst_eid` (peer) | Channel `reid` priority, fallback to SQE `rmEid` | Approximate | "" (tasks without channel info) | Collection side supplement peer explicit eid |
| `src_rank` / `dst_rank` | taskData endpoint rankId (carried by deviceId field, Section 4.4.1 rank field fallback) directly consumed: write=src local/dst peer, read=src peer/dst local; CCU/unresolved fallback via eid->rank index (Section 4.8) | Available/index fallback | INVALID + `rank_unresolved` count when both explicit value and index are unresolved | -- (Sections 5.2/5.5/6.7) |
| `transport_type` | opTaskMeta `HcclTaskProfInfo.transportType` (v3, enum name mapping Section 11.4.3) | Available (v3) | "" (v2 old BLOB without profInfo displays `Unknown`) | -- (Section 11.4 consumed) |
| `data_type` | REDUCE task-level `dataType` (v2 newly added, enum name mapping int8..bfp16); MEM_CPY task-level has no source | Partially available (v2) | REDUCE fills enum name; MEM_CPY "" | MEM_CPY collection side supplement |
| `port` | (reserved field) | Not filled | None | Simulator resolves port via eid at runtime (Section 7.6) |

**Instruction-level gaps (CCU microcode expansion)**: channel-type network task peer rank already resolved by eid->rank index
(Sections 4.8/6.7); reduce tasks always local (MS is rank-local buffer, Section 6.3.2), no peer eid
needed. UB tasks produced by CCU all explicitly carry `is_write` (Section 6.7), direction judgment does not depend on rank
value.

### 10.2 Task Type Mismatch Table

| db taskType / Old Type | New Model TaskType | Mismatch Treatment |
|----------------------|-------------------|-------------------|
| SYNC_STREAM (taskType=6) | **No corresponding type** | Skip, do not generate node (host-side stream sync point, not device-side task, Section 5.6); count `skipped_sync` |
| AIV_GRAPH (taskType=5) | None | Skip + warn (old and new models consistent, both unsupported) |
| WRITE_WITH_NOTIFY (taskType=7, db newly added) | `WRITE_WITH_NOTIFY` (newly added, 7th type) | Single task mapping (**not split** into write + NOTIFY_RECORD): write always write direction uses port bandwidth sharing, attached notify uses fixed record delay (Section 5.2, Section 10.3 #19 supported) |
| Old `REMOTE_RECORD` / `REMOTE_WAIT` | Already merged into `NOTIFY_RECORD` / `NOTIFY_WAIT` (src!=dst expresses remote) | db single rank path cross-rank notify generates remote node with `src!=dst` (Section 5.2, retain node without pairing) |
| Old `MEMCPY` (network transfer) | Semantics changed: new `Memcpy` = HBM copy | Network transfer changed to `Ub_Write_Or_Read` (write/read direction distinguished); `Ub_Inline_Write` only retains notify inline write semantics (fixed delay) |
| Old `LOCAL_COPY` | Split into `Memcpy` (pure copy) / `Reduce_Inline` (reduce) | REDUCE not network -> Reduce_Inline |
| Old `MemcpyDirection` enum (direction field) | Deleted | Direction expressed by task type + rank/eid values (Section 5.5) |
| Old `sync_stream_delay` config item | Cancelled with type gap | Incorporated into local_record_delay reuse |

### 10.3 Other Conflict Points (Subsequent Matching)

1. ~~**`src_rank == 0` local judgment hardcode** (`simulator.py` TRANSFER branch): if read direction task
   peer happens to be rank 0 and explicitly filled, would be misclassified as write.~~ **Fixed**: Task adds explicit
   `is_write` field (db graph formation path fills, task.txt old path is None falls back to heuristic), simulator
   consumes with priority; taskData peer rankId (carried by deviceId field) also directly filled into src/dst_rank (Sections 5.2/5.5).
2. **TopoResolver has no fallback**: eid not found in ranktable throws error directly (no jettyId->port mapping,
   no default port). db path avoids by "network tasks exist and no ranktable -> skip simulation only dump"
   (Section 3.3); subsequent may consider integrating ccuChannels/JettyMaps to build fallback mapping.
3. **Broadcast pairing only triggers first wait** (matched record's `break`): when multiple waits pair with same
   record, subsequent waits' Events are not triggered, end_time stays -1 (task silently lost).
4. **`is_matched_record` linear scan**: `dependency_graph.py` does O(R x M)
   traversal on matches, large-scale db (tens of thousands of tasks) has performance risk; recommend maintaining record->wait reverse index (also
   solves #3).
5. **SimConfig unit system and port key change**: JSON delay s->**us**, bandwidth B/s->**GBps**,
    port key int->**"die/port"** string; old config files incompatible (sample already synced this time
    `src/plugin/light_performance/sim_config.json`).
6. **HBM dynamic sharing semantics change**: old version per-task independent `size / hbm_bandwidth`, new version
   Reduce_Inline+Memcpy concurrent evenly divide HBM bandwidth (manager with utilization always 1.0); result scope
   has difference.
7. **Utilization query baseline `min_remaining`**: same group concurrent tasks use minimum remaining data volume to query utilization bin,
   differs from old version "each task queries by own data_size"--small tasks in group will drag down large tasks' utilization bin.
8. **Old input sources removed**: task.txt parsing, AllToAll generator, built-in demo removed from input_builder with HCCLSIM version;
   if log replay debug needed must restore separately.
9. **TopoResolver has no net_type filter**: when same eid has multiple network type entries, no priority rule exists.
10. **profiling json not implemented** (sub-requirement 5): current output is task graph dump (DOT/JSON/HTML)
    + text report + Chrome trace; perf_profiling.json format see Chapter 8 draft.
11. ~~**local_rank fallback convention**: when opDetails.rankId is missing, fall back by `deviceId-1`
    (follow task.txt path convention); deviceId and rank numbering have no necessary relationship, ranktable can
    be precise.~~ Peer rank already resolved by eid->rank index (RankEidMap main path + ranktable
    fallback, Section 4.8), no longer depends on numbering convention; local still uses opDetails.rankId as authoritative
    (channel leid cross-verification, Section 6.7).
12. ~~**CCU queue parallelism not modeled**: CCU queue chain is independent synthetic stream executing in parallel, real hardware
    queue concurrent resource constraints not modeled (checker side has queue parallelization modification, this plugin Section 6.7 already simplified).~~
    **Queue parallelization implemented** (Section 7.4.1: assembly-stage skeleton/unit partition + fork/join/attachment extra edges +
    `#v` virtual stream cross-mission reuse). Still not modeled: real hardware queue concurrent resource upper limit--
    virtual stream count unlimited (checker V3 likewise unlimited), subsequent may cap by chip generation.
13. **@estimate one-time path does not pass data_id**: `plugin run @estimate` special-case branch directly executes
    entry (`cmd_base_utils.cc:626-663`), does not go through generic message path's `DumpData` (which generates
    `data_id` currently fixed value "1234"), therefore output directory named by run timestamp (Section 8.1); if subsequent
    switch to resident model may align data_id semantics.
14. **`plugin install @estimate` semantics degraded**: install goes through `RegisterPlugin` resident startup
    (fork+pipe), main.py one-time execution completes then exits after stdin EOF, MonitorThread reclaims--
    behavior harmless but does not match install's resident semantics; if truly resident needed must implement stdin command loop
    (Section 3.2 evolution direction).
15. **Plugin artifacts do not participate in auto-archive**: `archive.sh` only archives `logs/` and `data/`,
    `plugin/light_performance/data/` perf_profiling.json and other artifacts do not enter archive directory
    (benefit is not affected by start-time `remove_all("data")`); if archive needed must extend archive.sh.
16. **simpy deployment dependency**: target environment needs pre-installed simpy (not distributed with hccl_vm_install offline wheel),
    main.py startup import failure should give clear error and installation prompt (Section 3.4).
17. ~~**db_reader.py not synced to v3 collection layout**: C side `HcclTaskMetaData` already expanded to 188B (after taskType
    newly added `HcclTaskProfInfo` 3 bytes, Section 4.4.1), `db_reader.py` still per 185B
    (`TASK_META_LEN=185`, `<BQIQQI16s16s`) decode--length gate `len(blob) >= 185` cannot block
    188B BLOB, reading v3-collected db would **fields entirely misaligned** (from commId). Need to sync to
    188B/`<BBBBQIQQI16s16s` and decode profInfo three fields (value domain/names see Section 11.4).~~
    **Synced**: `db_reader.py` auto-identifies v3 (188B/`<BBBBQIQQI16s16s`,
    profInfo three fields into `HcclTaskMetaData`) vs old v2 (185B) layout by BLOB length, see Section 4.4.1 implementation status.
18. ~~**JSON task type not consuming profInfo** (residual narrowed): trace args
    `transport_type`/`link_type` already switched to profInfo-driven, CommProtocol migrated to new field
    `protocol` (Sections 11.4.2/11.4.3, Section 11.1 plan a deployed); **residual** is `task_type`
    field--currently still derived by TaskType enum + jettyId/direction conditions (Section 11.2), with collection side
    profInfo hcomm op name has **semantic misalignment** (such as `Ub_Inline_Write` bidirectional reference to different
    tasks, Section 11.4.4); after switching to `profInfo.profTaskType` driven can correspond one-to-one with msprof timeline
    op names, target state comparison (including each value's correspondence to simulation TaskType) see Section 11.4.1.~~
    **Deployed (all residuals concluded)**: `Task` adds `task_name` display name (`task_model.py`,
    `display_name` = `task_name or task_type.value`), `graph_builder._add_task`
    unified inject `prof_task_type_name(profInfo.profTaskType)`; trace event name/`cat`
    (`report.py export_trace`), `args.task_type` (`to_trace_args`), verbose
    detail and dump (DOT/JSON/HTML legend) all switched to op name. Internal `TaskType` retired as
    delay model classification (Section 11.4.5 #3), delay/pairing/charging zero change. v2 old BLOB
    (profInfo=0xFF) displays `Unknown` (same scope as link/transport); CCU microcode subtasks
    have no profInfo, fall back to internal names (Section 11.3). Misalignment elimination comparison see Section 11.4.4 implementation record.
19. ~~**WRITE_WITH_NOTIFY (taskType=7) decode missing**: `db_reader.py`'s
    `HccLTaskMetaType` IntEnum only 0~6, `decode_op_task_meta` encountering 7 directly raises and discards
    (counted in warning).~~ **Supported (single task mapping)**: `db_reader.py` decodes
    `WriteWithNotifyTask` (Section 4.4.1) and single-task pass-through, `graph_builder` maps as
    `WRITE_WITH_NOTIFY` TaskType (trace display name `Write_With_Notify`, Section 5.2),
    **one-to-one correspondence** with db tasks; delay = port write direction bandwidth sharing + write-then-notify
    fixed record delay (Sections 7.3/7.4). **Does not adopt split strategy** (checker/runner side's
    "split into [write, NOTIFY_RECORD] two subtasks", and this plugin's early decode boundary split both
    deprecated): split would break task granularity, make trace/dump/statistics unable to correspond one-to-one with db tasks.
    Collection side profInfo=TASK_WRITE_(REDUCE_)WITH_NOTIFY (op name `Write_With_Notify` /
    `Reduce_With_Notify`, hcomm source code latter is misspelled `Reuce_With_Notify`, see 11.4.5),
    both profInfo values map to same `Write_With_Notify` task (isReduce only affects
    data_type display).
20. ~~**Same mission multiple dispatch SQE merged into single context**: CCU expansion merges all SQE of same
    `(dieId, missionId, queue)` into one dispatch context (single instruction window
    [first instStartId, end), single SQE args consumption)--does not satisfy checker
    `IsContiguousCcuSqe`'s "contiguous then merge" constraint (`task_meta_translator_v3.cc:213-231`).
    Typical affected scenario: instruction window wrap resend (head SQE 13 Load SqeArgs + body SQE sent twice consecutively,
    second pass uses new args), under old implementation second dispatch: (1) window start calculated by head SQE, end segment
    JMP jumps over entire data transfer instruction segment; (2) SQE args already consumed by first pass, subsequent Load takes 0 value
    (GSA address all 0); (3) loop count/CKE residual from first pass state--second dispatch's
    Ub/Memcpy/local post entirely lost, network data volume only 3/10 of checker.~~
    **Fixed (contiguous segment partition)**: partition by `instStartId`/`instCnt` continuity into multiple dispatch
    contexts (Section 6.1), each context independent window replay (including Load SqeArgs header) + independent sequential
    args consumption (`GetSqe` semantics); register/CKE bitmap shared across contexts (`AllRankParamRecorder`
    semantics), same queue contexts serial by dispatch order, multi-segment task id disambiguated with `_s{sequence}` suffix
    (`ccu_graph_builder.expand_ccu_graph`). After fix aligns with checker single-pass graph stream by stream:
    record local/remote 54/54, wait local/remote 27/36, Ub 39/39, Memcpy 6/6,
    network+local data volume 1088MB (2333 verification environment ALLGATHER dev1).
    Same root cause **wait node emission deviation** also fixed: old implementation per "each contributing post one wait node"
    split (one wait instruction can produce N nodes, inconsistent with checker `GenWaitNode`'s "per instruction at most
    1 local + 1 remote node", in this environment manifests as local wait 48 vs 27); now aggregated into single node
    (mask = all contributing post bits OR'd, first contributing post uses notify pairing, remainder via extra edges,
    Section 6.4), after fix local wait 27/27 aligned.
21. **Merged loop fold pseudo-serial residual gaps**: Section 6.5.1 feature chain resolution covers merged path's
    mid-body wait/record (folded pseudo-serial barrier), but two residuals: (1) full expansion fallback path
    (`mf > 0`) per-iteration wait/record is real 4KB granularity handshake, its inter-iteration overlap likewise
    serialized by queue chain (node count large, current sample does not trigger, not processed for now); (2) per-iteration handshake
    **rate** overhead not modeled--fold and resolution both count as single operation (CKE engine throughput if bottleneck needs dedicated
    rate model, such as ns/operation x 32768 operation magnitude evaluation).

---

## 11. db Task Type and JSON Artifact Task Type Mapping Complete Table

> This chapter is a type mapping quick reference table, answers three questions:
> (1) Tasks in db database (`opTask_P_<pid>` table taskType column) ultimately convert to which
> task types in JSON artifacts (Sections 11.1, 11.2); (2) CCU mode microcode instruction to JSON task type correspondence
> (Section 11.3); (3) Collection side profiling reconciliation information (`HcclTaskProfInfo`) to JSON
> display type correspondence and target state (Section 11.4). Implementation basis: `src/plugin/light_performance/
> graph_builder.py` (SQE mapping, `_map_tasks`/`_map_transfer`),
> `src/plugin/light_performance/ccu_graph_builder.py` (CCU instruction mapping, `transform`
> series), `src/plugin/light_performance/simulator/task_model.py` (TaskType enum).
> Same source as Section 5.2 (SQE mapping + delay model), Section 6.3 (instruction category mapping), here reorganized per
> "db -> JSON" correspondence, with reverse index supplemented.
>
> **profInfo background**: from v3 collection, db task opTaskMeta BLOB carries `HcclTaskProfInfo`
> sub-structure (Section 4.4.1, single source defined in `include/sim_profiling_defs.h`, checker/runner as
> opaque whole pass-through), three fields `profTaskType`/`linkType`/`transportType` value domain and
> names one-to-one aligned with hcomm `TaskParamType`/`DfxLinkType`/`DfxTransportType`--i.e.
> msprof timeline op name and L1 detail linkType/transportType field. This chapter Sections 11.1~11.3 describe
> **current TaskType implementation** (not yet consuming profInfo), 11.4 gives three-field complete comparison and switch
> target state (to-do see Section 10.3 #17~#18).

### 11.1 Task Type Carriers in JSON Artifacts

Current implementation JSON artifacts total three types, task types all taken from `TaskType` enum values
(string form, `task_model.py:20-26`):

| JSON Artifact | Format | Task Type Field | Generation Module |
|---------------|--------|----------------|-------------------|
| `perf_profiling.json` (main artifact, Section 8.1) | Chrome Trace Event | `traceEvents[].name` / `.cat` / `.args.task_type` | `simulator/report.py` (`export_trace`, event name is task display name: profInfo op name, Section 11.4) |
| `trace_comm<commId>_iter<opIter>.json` (optional single-operator trace) | Chrome Trace Event | Same as above | Same as above |
| Task graph dump `graph_comm<commId>_iter<opIter>_dev<dev>.json` (optional, `--dump-dir`) | `light_perf.graph_dump/2` | `nodes[].type` | `graph_dumper.py` |

> dump JSON notify nodes carry `nodes[].mask` when mask is non-zero (Section 7.3 Task.mask),
> HTML view tooltip syncs display; `perf_profiling.json`/trace task args also contain
> `mask` (Section 7.7).

Complete set of task types appearing in JSON (7 types):

| TaskType Value (JSON Presentation) | Delay Model | Semantics |
|-----------------------------------|-------------|-----------|
| `Notify_Record` | Fixed delay (local also triggers Event) | notify set bit (record / CKE post) |
| `Notify_Wait` | Wait for Event / fixed delay | notify wait (wait / CKE wait) |
| `Reduce_Inline` | HBM dynamic sharing | Local reduce (computation approximated as transfer, Section 5.7) |
| `Memcpy` | HBM dynamic sharing | Local copy (HBM, not network) |
| `Ub_Inline_Write` | Fixed delay (same as Notify_Record) | notify inline write (only old input path) |
| `Ub_Write_Or_Read` | Port bandwidth sharing (write/read directions charged independently) | Network data send/receive (SQE network tasks both directions uniformly mapped) |
| `Write_With_Notify` | Port write direction bandwidth sharing + write-then-notify fixed delay | Network write + attached notify (one-to-one with db `WRITE_WITH_NOTIFY`, Section 5.2) |

> Section 8.3 `perf_profiling.json` structured format (meta/ops array) is draft, current main artifact
> is actually Chrome tracing format (Section 10.3 #10); regardless of format, task types are the above
> 7 TaskType values, this chapter mapping table uniformly applies to them.

**Type-related fields in trace args (current state and profInfo consumption)**: Task's `to_trace_args`
(`task_model.py`) outputs four type-related fields; all four fields already switched to profInfo-driven
(Section 10.3 #18 all deployed, Section 11.4):

| trace args Field | Current Source | Status |
|-----------------|----------------|--------------------------|
| `task_type` | `profInfo.profTaskType` op name (`prof_task_type_name`, `graph_builder._add_task` unified fill as `Task.task_name`; `display_name` empty falls back to internal TaskType name) | **Deployed** (one-to-one with msprof timeline op name, Section 11.4.1; v2 old BLOB is `Unknown`) |
| `transport_type` | `profInfo.transportType` name (`ProfTransportTypeName`, Section 11.4.3, `graph_builder._add_task` unified fill) | **Deployed** |
| `link_type` | `profInfo.linkType` name (`ProfLinkTypeName`, Section 11.4.2, `graph_builder._add_task` unified fill)--**note this is a different concept from `protocol` (CommProtocol)** | **Deployed** (plan a: overwrite with DfxLinkType name) |
| `protocol` (new field) | Channel `CommProtocol` enum name (HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO, filled via `graph_builder._resolve_eids` / CCU `_channel_eids`) | **Deployed** (CommProtocol migrated from `link_type`, `N/A` when no channel info) |

> `link_type` overwrite (plan a) already implemented: semantics aligned with msprof L1 detail field name, avoid same-name-different-meaning;
> existing CommProtocol information not lost, moved to `protocol` field. CCU microcode subtasks are simulator
> semantic artifacts, have no profInfo (Section 11.3), their `link_type`/`transport_type` is `N/A`,
> `protocol` retains channel CommProtocol name.

### 11.2 SQE Path: db taskType -> JSON Task Type Mapping Complete Table

Forward mapping (db task -> JSON task display name). "JSON task type" column is trace/dump displayed
op name (already switched to `profInfo.profTaskType` driven, Section 10.3 #18 deployed); "Internal TaskType"
(delay model classification, determines charging/pairing) maintains Section 11.2 original judgment conditions (jettyId/direction/recordDev),
correspondence with op names see Section 11.4.1 "Simulation TaskType" column:

| db taskType (opTask.taskType column) | Judgment Condition | JSON Task Type (op name) | Collection side profInfo.profTaskType (op name) | Description |
|-------------------------------------|-------------------|-------------------------|------------------------------------------------|-------------|
| `NOTIFY_WAIT` (0) | recordDev == local device | `Notify_Wait` | `TASK_NOTIFY_WAIT` -> `Notify_Wait` check | Local wait: participates in first-come-first-served pairing (Section 5.4), waits for Event |
| `NOTIFY_WAIT` (0) | recordDev != local device | `Notify_Wait` | `TASK_NOTIFY_WAIT` -> `Notify_Wait` check | Cross-rank wait: node retained without pairing, src!=dst -> fixed `remote_wait_delay` |
| `NOTIFY_RECORD` (1) | waitDev == local device (SDMA/local notify) | `Notify_Record` | `TASK_NOTIFY_RECORD` -> `Notify_Record` check | Local record: participates in pairing, triggers Event |
| `NOTIFY_RECORD` (1) | waitDev != local device (SDMA/local notify) | `Notify_Record` | `TASK_NOTIFY_RECORD` -> `Notify_Record` check | Cross-rank post: node retained without pairing, src!=dst -> fixed `remote_record_delay` |
| `NOTIFY_RECORD` (1) | UBDMA WQE `inlineEn==1` (notify inline write) | `Ub_Inline_Write` | `TASK_UB_INLINE_WRITE` -> `Ub_Inline_Write` check | notify inline write (name already returned, Section 11.4.4); internal TaskType still Notify_Record (fixed record delay) |
| `REDUCE` (2) | jettyId == INVALID | `Reduce_Inline` | `TASK_REDUCE_INLINE` -> `Reduce_Inline` check | Local reduce: HBM dynamic sharing charging |
| `REDUCE` (2) | jettyId valid + write direction | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` -> `Ub_Reduce_Inline` check | Network reduce approximation (Section 5.7), internal TaskType=Ub_Inline_Write, port write direction charging |
| `REDUCE` (2) | jettyId valid + read direction | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` -> `Ub_Reduce_Inline` check | Same as above, internal TaskType=Ub_Write_Or_Read, read direction charging |
| `MEM_CPY` (3) | jettyId == INVALID (same device) | `Memcpy` | `TASK_SDMA` -> `Memcpy` check | Local copy: HBM dynamic sharing charging; profInfo.linkType=ONCHIP |
| `MEM_CPY` (3) | jettyId == INVALID (cross-device SDMA SQE) | `Memcpy` | `TASK_SDMA` -> `Memcpy` check | Cross-die SDMA transfer; same as Memcpy, difference expressed by profInfo.linkType=PCIE / transportType=SDMA (Section 11.4) |
| `MEM_CPY` (3) | jettyId valid + write direction | `Ub_Write_Or_Read` | `TASK_UB` -> `Ub_Write_Or_Read` check | Network send (direction not encoded in type name, expressed by direction/is_write); internal TaskType=Ub_Inline_Write |
| `MEM_CPY` (3) | jettyId valid + read direction | `Ub_Write_Or_Read` | `TASK_UB` -> `Ub_Write_Or_Read` check | Network receive; internal TaskType=Ub_Write_Or_Read |
| `WRITE_WITH_NOTIFY` (7) | Always write direction (no direction inference, Section 5.5) | `Write_With_Notify` (single task, no split); when `udfFlag==1` `Reduce_With_Notify` | `TASK_WRITE_WITH_NOTIFY` -> `Write_With_Notify`; when `udfFlag==1` `TASK_WRITE_REDUCE_WITH_NOTIFY` -> `Reduce_With_Notify` (hcomm source is misspelled `Reuce_With_Notify`, Section 11.4.5) | Write+notify combined task (`WriteWithNotifyTask`, Section 4.4.1): port write direction bandwidth sharing + write-then-notify fixed record delay; both profInfo values map to same internal WRITE_WITH_NOTIFY (delay model consistent, display name and isReduce distinguished) |
| `CCU_GRAPH` (4) | -- | (placeholder, not directly produced) | `TASK_CCU` -> `Ccu` | Expanded into CCU subgraph, final artifacts see Section 11.3 |
| `AIV_GRAPH` (5) | -- | None (skip) | `TASK_AIV` -> `AivKernel` | Warning + `skipped_aiv` count (not supported for now) |
| `SYNC_STREAM` (6) | -- | None (skip) | `UNKNOWN` -> `Unknown` | Host-side stream sync point, not device-side task (Section 5.6), hcomm profiling has no corresponding type |

> Cross-rank judgment (waitDev/recordDev) see Section 4.6; network direction judgment (write/read) see Section 5.5.

Reverse index (JSON task display name <- all sources, including CCU path and profInfo dimension; SQE path
display name = profInfo op name, CCU path display name = internal TaskType name, Section 11.3):

| JSON Task Display Name | SQE Source (db taskType) | Collection side profInfo Source (op name) | CCU Source (microcode instruction, Section 11.3) |
|------------------------|-------------------------|------------------------------------------|--------------------------------------------------|
| `Notify_Record` | `NOTIFY_RECORD` (1), local/cross-rank (SDMA/local notify) | `Notify_Record` (TASK_NOTIFY_RECORD check) | SetCKE/SetCKBit local post; TRANS/REDUCE/LOAD type instruction embedded `setCKE` field (A6 LOAD type also carries); cross-rank set/notify (A5 Sync series setRmtCKE; A6 TransMem notify/0x1A atomic, SyncWtX notify, SyncAtx--remote approximation); subgraph attachment launch-record / completion-record |
| `Notify_Wait` | `NOTIFY_WAIT` (0), local/cross-rank | `Notify_Wait` check | Various instruction embedded `waitCKE` field; ClearCKE; A6 independent Wait instruction (Xn value comparison, unsatisfied -> remote wait); subgraph attachment completion-wait / launch-wait |
| `Reduce_Inline` | `REDUCE` (2), jettyId invalid | `Reduce_Inline` check | REDUCE type instructions (msId list all local) |
| `Memcpy` | `MEM_CPY` (3), jettyId invalid (including cross-die SDMA) | `Memcpy` check | TRANS local transfer instructions (A5 4 opcodes: 0x0/0x2/0x5/0xA; A6 4 opcodes: 0x0/0x2/0x5/0x6) |
| `Ub_Inline_Write` | `NOTIFY_RECORD` (1) + `inlineEn==1` (notify inline write) | `Ub_Inline_Write` (TASK_UB_INLINE_WRITE check, name already returned) | -- (CCU network transfer uniformly mapped `Ub_Write_Or_Read`, 11.3.1) |
| `Ub_Write_Or_Read` | `MEM_CPY` (3) / `REDUCE` (2), jettyId valid (write/read direction) | `Ub_Write_Or_Read` (TASK_UB check, direction expressed by direction/is_write) | TRANS network transfer instructions (A5 write role 3 opcodes: 0x3/0x7/0x9, read role 4 opcodes: 0x1/0x4/0x6/0x8; A6 unified TransMem 0x10: dmaOpCode 3/5=write, 6=read, including udfEnable network reduce approximation); REDUCE type containing remote MS |
| `Ub_Reduce_Inline` | `REDUCE` (2), jettyId valid (network reduce) | `Ub_Reduce_Inline` (TASK_UB_REDUCE_INLINE check) | -- |
| `Write_With_Notify` | `WRITE_WITH_NOTIFY` (7), isReduce=0 | `Write_With_Notify` (TASK_WRITE_WITH_NOTIFY check) | -- |
| `Reduce_With_Notify` | `WRITE_WITH_NOTIFY` (7), udfFlag==1 | `Reduce_With_Notify` (TASK_WRITE_REDUCE_WITH_NOTIFY check) | -- |

### 11.3 CCU Path: Microcode Instruction -> JSON Task Type Mapping

The SQE of `CCU_GRAPH` (4) in db is only a placeholder (`CcuPlaceholder`, Section 6.1), final
JSON tasks are produced by microcode interpretation from `ccuInstrRes` instruction space (Chapter 6). **A5 (V1,
Ascend950, devType=6) and A6 (V2, Ascend960, devType=8) both instruction sets are implemented**
(selected by `opDetails.devType`, other devType falls back to A5): A5 complete table see Section 11.3.1,
A6 complete table see Section 11.3.3.

> profInfo correspondence: CCU placeholder SQE's `HcclTaskProfInfo` as whole is
> `TASK_CCU`->`Ccu` (linkType/transportType is UNKNOWN, microcode has no link dimension). Expanded
> microcode subtasks (`Memcpy`/`Ub_Write_Or_Read`/`Notify_*` etc.) are
> simulator semantic artifacts, **have no corresponding ProfTaskParamType**, display name falls back to internal TaskType name
> (Section 10.3 #18 implementation established trade-off, Section 11.4.4); msprof side CCU tasks via
> microcode-level detail (`CcuProfilingInfo`: CCU_TASK/WAITCKE/LOOPGROUP three types, reported with TASK_CCU)
> express equivalent granularity information, reconciliation uses whole SQE's `Ccu` as alignment anchor.

#### 11.3.1 A5 Instruction Complete Table (Instruction -> JSON Task Type)

Microcode instruction = 2B header (type:4bit category + code:11bit instruction code, Section 6.2) + 30B payload:

| Instruction Category (type) | Instruction (code) | JSON Task Type | Description |
|----------------------------|-------------------|----------------|-------------|
| LOAD (0x0) | LoadSqeArgsToGSA (0x0), LoadSqeArgsToXn (0x1), LoadImdToGSA (0x2), LoadImdToXn (0x3), LoadGsaXn (0x4), LoadGsaGsa (0x5), LoadXx (0x6) | **None** (does not generate node) | Only reads/writes register model (xn/gsa), for subsequent instructions to get length, loop count etc. |
| CTRL (0x1) | SetCKE (0x2) | `Notify_Record` | Local CKE post (ckeId as notify_id, ckeMask as mask, src=dst=local); embedded `waitCKE` field can additionally produce `Notify_Wait` |
| CTRL (0x1) | ClearCKE (0x4) | `Notify_Wait` | wait semantics (embedded `waitCKE` field); clear action itself does not produce node |
| CTRL (0x1) | Loop (0x0) | **None** | Must be triggered by LoopGroup; appearing alone is abnormal, warn and skip |
| CTRL (0x1) | LoopGroup (0x1) | (no direct node) | Loop control: merge/expand loop body (Section 6.5), body instructions mapped per this table; merge path data task size x total iteration count, and emitted task carries loop_cnt (expanded loop count) / loop_iter_cnt (single loop self-loop count), total iteration count = product of two fields |
| CTRL (0x1) | Jmp (0x5) | **None** | Conditional jump: when `Xn[conditionXnId] != expectData` pc jumps to instruction pointed by `Xn[dstInstrXnId]` (checker `TransformJmpInstrA5` semantics, target limited to current instruction window, out-of-bounds warn pass-through); pass-through within loop body (merge pre-verification already rejected loop bodies containing JMP); step budget defense against backward jump infinite loop |
| TRANS (0x2) | LocMemToLocMs (0x0), LocMsToLocMem (0x2), LocMsToLocMs (0x5), LocMemToLocMem (0xA) | `Memcpy` (src/dst_buf_type annotated mem/ms) | Local transfer (LOCAL_COPY role, Section 6.3.2), HBM dynamic sharing |
| TRANS (0x2) | LocMsToRmtMem (0x3), LocMemToRmtMem (0x9), LocMsToRmtMs (0x7), RmtMemToLocMs (0x1), RmtMsToLocMem (0x4), RmtMsToLocMs (0x6), RmtMemToLocMem (0x8) | `Ub_Write_Or_Read` (is_write distinguishes direction, src/dst_buf_type annotated mem/ms) | Cross die/card transfer (write/read role unified type, Section 6.3.2); peer rank resolved via channel reid -> eid->rank index (Sections 4.8/6.7) |
| TRANS (0x2) | SyncCKE (0xB), SyncGSA (0xC), SyncXN (0xD) | Local post part -> `Notify_Record`; remote post part -> `Notify_Record` (remote, unpaired) | Remote set fields (SyncCKE's `rmtCKEId/locCKEMask`, SyncGSA/SyncXN's `setRmtCKEId/setRmtCKEMask`) generate remote record node (src=local, dst=INVALID -> fixed `remote_record_delay`, does not register local accounting), counted in `remote_posts` (Section 6.6); SyncGSA/SyncXN carrying second remote post likewise generates; embedded `waitCKE` can produce `Notify_Wait` |
| REDUCE (0x3) | Add (0x0), Max (0x1), Min (0x2) | `Reduce_Inline` | Reduce computation approximated as transfer (Section 5.7); MS always rank-local buffer (msId high bits only addressing encoding, Section 6.3.2), local HBM charging |
| (subgraph attachment) | -- (synthesized during graph formation, not microcode) | None (does not produce task node) | Attachment dependencies all expressed via extra edges (Section 7.4.1): entry edge SQE predecessor->skeleton head/pre-skeleton unit head, exit edge mission completion set->SQE successor; task graph does not magically produce any nodes |

#### 11.3.2 Embedded Sync Fields (Cross-Instruction Common)

TRANS / REDUCE / SetCKE / Sync type instruction payloads commonly carry sync fields, **superimposed to produce** JSON tasks with main semantics:

| Payload Field | JSON Task Type | Behavior |
|---------------|---------------|----------|
| `waitCKEId` / `waitCKEMask` | `Notify_Wait` | When bits satisfied, pairs with local post accounting (aggregated into single node: mask=all contributing post consumption bits OR'd, first post add_match, remainder extra edges, Section 10.3 #20); bits insufficient: top-level instruction blocks this queue (stall detection finds then generates remote wait node and replays remaining instruction parts, `remote_waits`); within loop body cannot block, directly generates remote wait node and continues |
| `setCKEId` / `setCKEMask` | `Notify_Record` | Local CKE post, registers seen_post accounting (Section 6.4) |
| `clearType` bits / SetCKE clear flag | None (only register clear) | Clears bits after wait completes, does not produce node |

> Therefore one TRANS instruction can produce 1 data transfer task (`Memcpy`/`Ub_Write_Or_Read`),
> and additionally produce several `Notify_Wait` (paired by bits,
> 0..n) and 1 `Notify_Record` per embedded fields. Under merged loop (Section 6.5) each slot likewise retains
> wait/set tasks, only data tasks folded to size x total iteration count.
>
> **A6 difference**: embedded `waitCKE` only exists in SetCKBit/ClearCKE (CTRL type); local
> TRANS/REDUCE and TransMem/Sync type only carry tail `setCKEId/setCKEMask` (@26/28,
> additionally produce `Notify_Record`); wait semantics handled by independent Wait instruction (Xn value comparison).

#### 11.3.3 A6 (Ascend960) Instruction Complete Table (Instruction -> JSON Task Type)

devType=960 (`DEV_TYPE_960`) corresponds to V2/A6 instruction set, **implemented** (ported from checker
`ccu_task_transform_instruct_a6_v3.cc`, interpreter `_MissionContextA6`). Microcode instruction
= 2B header (isomorphic with A5) + 30B V2 payload (`ccu_microcode_v1.h` CcuV2 struct,
pack(1)); register numbers commonly via `GetXnId` (bit15 addressing mode, Section 6.4).

| Instruction Category (type) | Instruction (code) | JSON Task Type | Description |
|----------------------------|-------------------|----------------|-------------|
| LOAD (0x0) | LoadSqeArgsToX (0x1), LoadImdToX (0x2), LoadX (0x6), StoreX (0x7), ClearX (0x8), Nop (0x9), arithmetic Add (0xD)/Sub (0xE)/Mul (0xF)/And (0x10)/Or (0x11)/Not (0x12)/Xor (0x13)/Shl (0x14)/Shr (0x15)/Popcnt (0x16) | Data plane **None**; when `setCKEMask != 0` additionally `Notify_Record` | Unified Xn register read/write (no GSA): SQE args sequential consumption (Section 6.1), immediate load, indirect read/write (LoadX/StoreX, oMode 0=immediate offset/1=register offset), range clear (ClearX), 64-bit wrap arithmetic (Mul takes low 32-bit product, shift amount 0x3F truncated, Shl/Shr support logical/arithmetic/rotate three types); all except Nop carry tail setCKE field (checker `ProcessSetMask`, mask non-zero -> local post node) |
| LOAD (0x0) | Load (0xA), Store (0xB) | Data plane **None**; setCKE additionally `Notify_Record` | HBM <-> Xn data plane: HBM content not modeled (checker reads/writes data via AllRankParamRecorder), registers not backfilled, warn and leave trace; only retain sync semantics |
| CTRL (0x1) | SetCKBit (0x2) | `Notify_Record` (+ embedded `waitCKE` additionally `Notify_Wait`) | wait -> local post -> when clearType==1 clear wait bits (checker only supports clearType=1, others warn and skip clear) |
| CTRL (0x1) | ClearCKE (0x4) | `Notify_Wait` (embedded `waitCKE`) | After wait, per checker `ClearWaitMask` semantics clear consumed wait bits; `clearMask != 0` is checker unsupported scenario (warn + tolerant clear target bits) |
| CTRL (0x1) | Loop (0x0) | **None** | Must be triggered by LoopGroup; appearing alone warn and skip. `mode==1` conditional Loop: `wishCKEBit == Xp.loopCtxId + 1024` hit then expand (Section 6.5) |
| CTRL (0x1) | LoopGroup (0x1) | (no direct node) | Loop control: parameters from Xn/Xm/Xp three registers (loopInsCnt:10/expandOffset:9/expandCnt:9 etc., Section 6.5); merge/expand same as A5 (merge verification includes A6 additional rules: wait CKE ID not reused, cross-expand iteration count/length/channel consistent); illegal config (loopInsCnt>512 etc.) warn and skip |
| CTRL (0x1) | Jmp (0x5) | **None** | Conditional jump: conditionType 0~5 (==/!=/>/>=/</<=), >=6 unconditional; jumpMode 0=relative pc (mod 0x10000) /1=absolute instruction number, target takes `Xn[relTarInstrXnId]`; limited to current instruction window, out-of-bounds warn pass-through; pass-through within loop body; step budget defense against infinite loop |
| CTRL (0x1) | Wait (0x7) | When unsatisfied remote `Notify_Wait` | Xn value comparison spin (conditionType same as Jmp, default per equal): satisfied pass-through no node; unsatisfied = depends on peer remote register write -> top-level stall resolution/within loop body directly generates remote wait node (fixed `remote_wait_delay`, task id tag `xwt`, counted in `remote_waits`) |
| CTRL (0x1) | Fence (0x8) | **None** | checker does not support (error abort); here warn pass-through (queue serial model order naturally guaranteed) |
| TRANS (0x2) | TransLocMemToLocMS (0x0), TransLocMSToLocMem (0x2), TransLocMSToLocMS (0x5), TransLocMemToLocMem (0x6) | `Memcpy` (src/dst_buf_type annotated mem/ms) | Local transfer (length `Xn[GetXnId(xlId)]`); no embedded waitCKE, tail setCKE additionally `Notify_Record` |
| TRANS (0x2) | TransMem (0x10) | `Ub_Write_Or_Read` (dmaOpCode 3/5=write, 6=read); notify/atomic -> remote `Notify_Record` | Unified network transfer: channel = `Xn[GetXnId(xcId)]` register value -> peer rank/eid resolution same as A5 (Sections 4.8/6.7); when `msIdMode==1` write source / read destination is local MS (buf_type=ms); dmaOpCode 5 write-then-notify and 0x1A remote atomic count (no data plane) approximated as remote record (mask=notification value low 16 bits); `udfEnable` network reduce charged as normal UB transfer (Section 5.7 approximation); 0x19 and other dmaOpCode / `udfType != 0` warn skip data plane (setCKE post still executes) |
| TRANS (0x2) | SyncXnWtx (0xD), SyncAtx (0xE) | Remote `Notify_Record` (approximation); SyncAtx additionally has setCKE -> local `Notify_Record` | Cross-rank register sync: SyncWtX's `notifyValid==1` notify and SyncAtx's atomic write approximated as remote record (remote target type single rank cannot resolve, Section 6.6); main value sync is remote register write does not build node (A5 SyncXn same scope); SyncWtX tail setCKE field checker does not consume (local does not post); channel unresolvable warn skip (`eid_unresolved`) |
| REDUCE (0x3) | ReduceAdd (0x0), ReduceMax (0x1), ReduceMin (0x2) | `Reduce_Inline` | Local MS batch reduce (msId[8] list + `Xn[XnIdLength]` length); no embedded waitCKE, tail setCKE additionally `Notify_Record`; MS always local HBM charging (Section 6.3.2) |
| (subgraph attachment) | -- (synthesized during graph formation, not microcode) | None (does not produce task node) | Same as A5 (Section 7.4.1 extra edge attachment) |

> A6 established deviations from checker (HBM data plane, remote target type resolution, checker error scenario
> tolerant downgrade etc.) summarized in `_MissionContextA6` class docstring and Section 6.7 comparison table;
> loop merge/pipeline concurrency recovery (Sections 6.5/6.5.1), queue parallelization (Section 7.4.1), SQE merge
> and stall resolution (Sections 6.1/6.6) and other framework mechanisms shared by both generations.

### 11.4 HcclTaskProfInfo Three-Field Comparison (profInfo -> JSON Display)

From v3 collection db tasks carry `HcclTaskProfInfo` (Section 4.4.1, single source definition
`include/sim_profiling_defs.h`). This section gives the three fields "value -> enum -> JSON display string"
complete comparison and each value's correspondence to simulation TaskType (delay model classification, Section 7.3);
C++ side name functions (`ProfTaskParamTypeName`/`ProfLinkTypeName`/
`ProfTransportTypeName`) are the single authoritative source for display strings.

> **Implementation status**: `task_type`/`transport_type`/`link_type`/`protocol` four fields all per
> this section comparison deployed (Python side display name mapping in `db_reader.py`'s `PROF_*_NAMES`, with C++
> name functions character-by-character consistent, Section 11.4.5; `graph_builder._add_task` unified fill--`task_type`
> via `Task.task_name`/`display_name`, Section 10.3 #18 concluded).

#### 11.4.1 profTaskType (ProfTaskParamType, 0~22 -> msprof timeline op name)

This table supplements each value's correspondence to simulation TaskType (`task_model.py`, delay model classification)--previously
db task and simulation task type matched via db taskType column (Section 11.2), after profInfo consumption this table
`profTaskType -> simulation TaskType` is second match dimension (judgment conditions unchanged, same db taskType
per jettyId/direction routed to different TaskType rows noted inline):

| Value | Enum | JSON Display String (op name) | Simulation TaskType (delay model classification) | light_perf Occurrence Scenario |
|-------|------|------------------------------|------------------------------------------------|------------------------------|
| 0 | TASK_SDMA | `Memcpy` | `Memcpy` (jettyId invalid, HBM charging) / `Ub_Inline_Write`/`Ub_Write_Or_Read` (jettyId valid by direction, port charging) | Local copy, cross-die SDMA SQE (linkType distinguishes, 11.4.2) check |
| 1 | TASK_RDMA | `RDMASend` | `Memcpy` (jettyId invalid, HBM charging approximation) | DPU host machine ibv normal read/write (`RDMASend`, hcomm UbMemTransport::Write/Read same scope) check |
| 2 | TASK_REDUCE_INLINE | `Reduce_Inline` | `Reduce_Inline` (jettyId invalid, HBM charging) / `Ub_Inline_Write`/`Ub_Write_Or_Read` (network approximation, Section 5.7, port charging) | Local reduce check |
| 3 | TASK_REDUCE_TBE | `Reduce_TBE` | -- (does not appear) | 910 generation reserved, does not appear |
| 4 | TASK_NOTIFY_RECORD | `Notify_Record` | `Notify_Record` | SDMA/local notify record check |
| 5 | TASK_NOTIFY_WAIT | `Notify_Wait` | `Notify_Wait` | notify wait check |
| 6/7 | TASK_SEND_NOTIFY / TASK_SEND_PAYLOAD | `Send_Notify` / `Send_Payload` | -- (does not appear) | hcomm reserved, does not appear |
| 8 | TASK_WRITE_WITH_NOTIFY | `Write_With_Notify` | `Write_With_Notify` | WRITE_WITH_NOTIFY (7) whole task (single task mapping, Section 10.3 #19 supported) check |
| 9 | TASK_WRITE_REDUCE_WITH_NOTIFY | `Reduce_With_Notify` (tool side corrected spelling; hcomm source/msprof is misspelled `Reuce_With_Notify`, Section 11.4.5) | `Write_With_Notify` (isReduce=1 reduce write payload, data_type distinguishes display) | WRITE_WITH_NOTIFY (7) isReduce=1 |
| 10 | TASK_CCU | `Ccu` | (CCU placeholder whole SQE, expanded into Chapter 6 subgraph, Section 11.3) | CCU placeholder whole SQE check |
| 11/12 | TASK_AICPU_KERNEL / TASK_AICPU_REDUCE | `AicpuKernel` / `Aicpu_Reduce` | -- (not consumed) | AICPU kernel dispatch (goes through device process, current no collection landing point/not consumed) |
| 13 | TASK_AIV | `AivKernel` | -- (skip) | AIV kernel task (skip) |
| 14 | TASK_UB_INLINE_WRITE | `Ub_Inline_Write` | `Notify_Record` (notify fixed record delay; display name already returned check) | UBDMA `inlineEn==1` notify inline write |
| 15 | TASK_UB_REDUCE_INLINE | `Ub_Reduce_Inline` | `Ub_Inline_Write` (write) / `Ub_Write_Or_Read` (read) (network reduce approximation, port charging) | UBDMA `udfFlag==1` network reduce |
| 16 | TASK_UB | `Ub_Write_Or_Read` | `Ub_Inline_Write` (write direction) / `Ub_Write_Or_Read` (read direction) (direction expressed by direction/is_write) | UBDMA normal read/write WQE (write/read both directions same value) |
| 17~22 | TASK_DPU_KERNEL / THREAD_FENCE / CHANNEL_FENCE / INLINE_WRITE / NOTIFY_WAIT / WRITE_WITH_NOTIFY | `DpuKernel` / `Dpu_ThreadFence` / `Dpu_ChannelFence` / `Dpu_Notify_Record` / `Dpu_Notify_Wait` / `Dpu_Write_With_Notify` | `Dpu_Notify_Record`->`Notify_Record`, `Dpu_Notify_Wait`->`Notify_Wait` (cross-rank fixed delay), `Dpu_Write_With_Notify`->`Write_With_Notify` (single task port charging, Section 11.4.6); `DpuKernel`/Fence type current collection path does not produce | Host machine CPU RoCE (ibv_dpu_stub) tasks check (Section 11.4.6) |
| 0xFF | UNKNOWN | `Unknown` | -- (SYNC_STREAM etc. already skipped, does not produce node) | SYNC_STREAM, level1 channel API tasks, v2 old BLOB and other unlabeled sources |

> `task_type` already switched to op name (Section 10.3 #18 concluded), this table "Simulation TaskType" column is now
> internal delay model classification index (Section 11.4.5 #3); check mark rows op name consistent with TaskType value.
> After SQE batch merge (Section 5.1.1), merged task op name and link/transport take **last** data task's profInfo in batch
> (description follows last task, length takes whole batch sum), v3 and earlier BLOB without
> sqeGroupId do not merge, per-task display.

#### 11.4.2 linkType (ProfLinkType, 0~9 -> hcomm DfxLinkType enum name)

| Value | Enum | JSON Display String | light_perf Occurrence Scenario |
|-------|------|---------------------|------------------------------|
| 0 | LINK_ONCHIP | `ONCHIP` | Local Memcpy/Reduce_Inline/Notify_* |
| 1 | LINK_HCCS | `HCCS` | Reserved (current collection path does not produce) |
| 2 | LINK_PCIE | `PCIE` | Cross-die SDMA SQE (same device judged as local ONCHIP, Section 4.4.1 collection rules) |
| 3 | LINK_ROCE | `ROCE` | Host machine CPU RoCE (DPU, ibv_dpu_stub) tasks check (Section 11.4.6) |
| 4/5/6 | LINK_SIO / HCCS_SW / STANDARD_ROCE | `SIO` / `HCCS_SW` / `STANDARD_ROCE` | Reserved |
| 7 | LINK_UB | `UB` | Network tasks (UBDMA WQE all types) |
| 8 | LINK_UBoE | `UBoE` | UBoE channel (collection side does not yet distinguish UB/UBoE, does not yet appear) |
| 9 | LINK_RESERVED | `RESERVED` | Reserved |
| 0xFF | UNKNOWN | `Unknown` | CCU/AIV/SYNC_STREAM, level1 channel API tasks |

> **Implementation status**: trace args `link_type` field already filled per this table (`prof_link_type_name`,
> `graph_builder._add_task` unified injection, Section 11.1 plan a); this table and original `link_type` carried
> `CommProtocol` channel protocol (HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO) are **two different
> concepts**: CommProtocol describes channel establishment protocol, DfxLinkType describes task execution link (profiling
> dimension). CommProtocol name already migrated to `protocol` field (`N/A` when no channel info).

#### 11.4.3 transportType (ProfTransportType, 0~4 -> msprof L1 detail transportType value domain)

| Value | Enum | JSON Display String | light_perf Occurrence Scenario |
|-------|------|---------------------|------------------------------|
| 0 | TRANSPORT_SDMA | `SDMA` | Cross-die SDMA SQE |
| 1 | TRANSPORT_RDMA | `RDMA` | Reserved (910 generation RDMA semantics) |
| 2 | TRANSPORT_LOCAL | `LOCAL` | Local tasks (Memcpy/Reduce_Inline/Notify_*) |
| 3 | TRANSPORT_UB | `UB` | Network tasks (**delay model anchor**: port bandwidth sharing charging, Section 5.5) |
| 4 | TRANSPORT_ROCE | `ROCE` | Host machine CPU RoCE (DPU, ibv_dpu_stub) tasks check (Section 11.4.6) |
| 0xFF | UNKNOWN | `Unknown` | Unlabeled tasks |

> **Implementation status**: trace args `transport_type` field already filled per this table
> (`prof_transport_type_name`, `graph_builder._add_task` unified injection).
> Same-named `Memcpy` task local/cross-die difference expressed by linkType+transportType:
> ONCHIP+LOCAL (local) vs PCIE+SDMA (cross-die)--trace args can now losslessly distinguish these two types
> of previously indistinguishable tasks in JSON.

#### 11.4.4 Ub_Inline_Write Bidirectional Semantic Misalignment and Switch Implementation Record

**Historical issue**: before switch, JSON `Ub_Inline_Write` (network write direction send) and hcomm
profiling `Ub_Inline_Write` (TASK_UB_INLINE_WRITE, notify inline write) **refer to
different tasks**, and are inversely misaligned:

| Task (Collection Semantics) | db taskType | ProfTaskParamType (op name) | Pre-switch JSON Type |
|-----------------------------|-------------|----------------------------|---------------------|
| UBDMA normal write WQE (network send, write direction) | MEM_CPY (3) | `TASK_UB` -> `Ub_Write_Or_Read` | `Ub_Inline_Write` <- name misalignment |
| UBDMA network reduce (write/read direction) | REDUCE (2) | `TASK_UB_REDUCE_INLINE` -> `Ub_Reduce_Inline` | `Ub_Inline_Write` / `Ub_Write_Or_Read` <- name misalignment |
| UBDMA inline write notify (`inlineEn==1`) | NOTIFY_RECORD (1) | `TASK_UB_INLINE_WRITE` -> `Ub_Inline_Write` | `Notify_Record` <- name misalignment |

**Implemented (Section 10.3 #18 concluded)**--trace/dump display name switched to `profInfo.profTaskType`
op name (`Task.task_name`, `graph_builder._add_task` unified injection), all three misalignments eliminated:
1. Network data task (TASK_UB) write/read uniformly displays `Ub_Write_Or_Read`; **direction no longer encoded
   in type name**, expressed by trace args existing `direction` (write/read) and `is_write` field
   --delay model unchanged, still per port write/read direction bandwidth sharing separately charged (Section 5.5).
2. `Ub_Inline_Write` name **returned** to notify inline write task (db taskType=
   NOTIFY_RECORD + inlineEn==1), one-to-one with msprof timeline.
3. Network reduce displays `Ub_Reduce_Inline` (TASK_UB_REDUCE_INLINE), retains reduce semantics distinguishable.
4. `Notify_Record`/`Notify_Wait`/`Reduce_Inline`/`Memcpy` four types already consistent with op name,
   switch has no change (`Memcpy` corresponds to TASK_SDMA).

**Implementation trade-offs (established decisions)**:
- v2 old BLOB (no profInfo, profTaskType=0xFF) displays `Unknown`--same scope as
  `link_type`/`transport_type` v2 behavior (strict semantics, no internal name fallback).
- CCU microcode subtasks (Section 11.3) have no profInfo, display name falls back to internal TaskType name--simulator
  semantic artifacts have no reconciliation need, maintain status quo (network transfer uniformly `Ub_Write_Or_Read`, direction by
  `direction`/`is_write` expressed, Section 11.3.1).
- Internal `TaskType` enum retired as delay model classification (Section 11.4.5 #3), simulation/pairing/charging zero change.

#### 11.4.5 String Consistency Convention (Reconciliation Basis)

1. **Python side display strings must be character-by-character consistent with C++ side three name function output**
   (`sim_profiling_defs.h` single source), including hcomm source code existing spellings
   (`Ub_Write_Or_Read`, `Dpu_Notify_Record` etc.)--this is the basis for field-by-field reconciliation with msprof timeline /
   L1 detail.
   **Single exception**: `TASK_WRITE_REDUCE_WITH_NOTIFY` op name tool side unified as corrected spelling
   `Reduce_With_Notify` (hcomm source code two places both misspelled `Reuce_With_Notify`:
   `profiling_common.h:42` and `dfx_profiling_handler_lite.cc:67`, msprof raw
   data displays the misspelled version); when cross-system reconciliation encounters this string need to do
   `Reuce_With_Notify <-> Reduce_With_Notify` difference mapping.
2. trace args field-level switch (**all deployed**, Section 10.3 #18 concluded):
   `transport_type` <- transportType name, `link_type` <- linkType name
   (`graph_builder._add_task` unified injection, value domain tables see Sections 11.4.2/11.4.3), CommProtocol
   migrated to new field `protocol` (plan a, Section 11.1); `task_type` <- profTaskType op name
   (`Task.task_name`/`display_name`, Section 11.4.4 implementation record).
3. `TaskType` enum already retired as internal delay model classification (`task_name` handles display duty),
   trace event name (`traceEvents[].name`) and `args.task_type` both switched to op name;
   CCU microcode subtasks/CLI input have no profInfo, `display_name` falls back to internal name.

#### 11.4.6 DPU (Host machine CPU RoCE) Task Collection and Merge

Host machine DPU transfer (hcomm `HostCpuRoceChannel`, pod networking cross-node path) ibv hijack tasks
(`ibv_dpu_stub.cc`) previously did not tag profInfo (display `Unknown`), WriteWithNotify split into two
tasks, peer resolution loops back to local. This section is the alignment implementation record for this path.

**profInfo tagging** (align with hcomm `host_cpu_roce_channel.cc` DFX scope, link/transport
always ROCE/ROCE):

| ibv Operation | db taskType | profTaskType (op name) |
|---------------|-------------|----------------------|
| Pure imm notify (len=0 WRITE_WITH_IMM / hybrid) | NOTIFY_RECORD | `TASK_DPU_INLINE_WRITE` -> `Dpu_Notify_Record` |
| Write+imm (len>0 WRITE_WITH_IMM) | **WRITE_WITH_NOTIFY (7) single task** | `TASK_DPU_WRITE_WITH_NOTIFY` -> `Dpu_Write_With_Notify` |
| Pure RDMA write/read (no imm) | MEM_CPY | `TASK_RDMA` -> `RDMASend` (align with hcomm `UbMemTransport::Write/Read` precedent) |
| poll_cq pairing | NOTIFY_WAIT | `TASK_DPU_NOTIFY_WAIT` -> `Dpu_Notify_Wait` |

**WriteWithNotify merge single task**: whole operation only produces one taskType=7 task
(`WriteWithNotifyTaskData`: src/dst rank, offsets, len, notifyId, protocol),
aligns with hcomm whole operation only reports one `Dpu_Write_With_Notify` scope; `DpuPendingNotify`
pairing mechanism unchanged. Simulation side `_map_write_with_notify` uses port write direction bandwidth sharing
(local RoCE eid -> ranktable `d2h` port; when `d2h` has no configured port bandwidth falls back to
`default_port_bandwidth`).

**Real peer resolution (phantom echo problem, socket timing binding)**: estimate mode socket layer constructs
phantom peer (local Send mirror replay as Recv, `hccp_ra_socket_stub.cc`), QP handshake returns
"remote gid" which is actually local self, `RaTypicalQpModify` gid->IP resolution returns to local. Fix
(zero-byte read, only depends on API call timing):
- `RaGetSockets` resolves real peer EndPoint from `SocketInfoT.remoteIp` (hcomm link configuration, client/server role
  both carry) and stores in FdInfo (peerDeviceId/peerEid/localEid);
- `RaGetQpAttr` anchor (hccp_stub, only executes within estimate branch) records QP pending bind
  (`DpuRecordPendingBindQp`, `bindAttempted` prevents Modify repeated attr query then misbind);
- Same thread next `RaSocketSend` (i.e. hcomm ExchangeData attr pack->Send timing) via
  `BindDpuPendingQps` binds QP to that socket real peer: `DpuQpInfo` adds
  `peerRankId`/`lcEid` (peer rank by FdInfo peerEid queries `RankEidMap`, estimate
  communicator table only has local rank; lcEid for WRITE_WITH_NOTIFY port charging).
`RaTypicalQpModify` maintains `remoteDeviceId` loopback plane semantics
(`DpuPendingNotify` pairing depends on this, under estimate = local), task metadata uniformly takes binding fields.

**NotifyWait precise synthesis (hcomm primitive interception)**: correct semantics is "only pending
inserted by peer post generates NotifyWait for me", under phantom peer does not run, this notification has no real source, and hcomm
`NotifyWait` does imm hard verification on poll result (`wc.imm_data == dpuNotifyId`, mismatch immediately
error). Implementation (`hcomm_api_stub.cc` intercepts 5 hcomm northbound primitives, real pointer static
one-time dlsym resolution, **normal mode first line short-circuit pure forwarding, zero overhead**):
- post type primitives (`HcommChannelNotifyRecordOnThread` / `HcommWriteWithNotify(+-Nbi)
  OnThread`, input carries `remoteNotifyIdx`) -> `InsertDpuPendingNotifyEstimate`
  records `channelSlots[channel][idx] = (imm, notifyId)` slot mapping--under echo
  `remoteDpuNotifyIds_ == localDpuNotifyIds_`, post side idx->imm mapping is wait side
  expected mapping (same idx same slot value);
- wait type primitives (`HcommChannelNotifyWaitOnThread(+-WithDefaultTimeout)`, input carries
  `localNotifyIdx`) -> `ConsumeDpuPendingNotifyAndEmit` precise path: expected
  imm = `channelSlots[channel][idx]` **precisely known**, directly synthesize "virtual peer post" to construct
  wc return, imm must pass hcomm hard verification; NotifyWait node records local real wait event,
  only virtualized is "notification arrived" this external fact (isomorphic with socket echo virtualizing "peer response");
- Design constraint: estimate mode wait consumption **only goes through precise path**--all wait entries must be intercepted
  (single-card phantom FIFO loopback consumption unreliable, send/receive asymmetric imm mismatch/timeout, old mechanism removed);
  unintercepted entry -> `pollCq::miss` **ERROR** (per qpn+idx combination deduplicated once, prevent busy-loop
  flooding log) -> hcomm timeout exposed, accordingly supplement interception in `hcomm_api_stub.cc`.

**taskSrc source tag (BLOB v5)**: `HcclTaskMetaData` adds `taskSrc`
(0=default, 1=DPU), DPU stub emitted tasks set to 1, for Python side precise split by source
(does not depend on transport==ROCE etc. link semantic criteria, AICPU RoCE etc. future sources not affected).

**DPU sequence merge pass (Section 5.1.2, `sqe_group_merge.py::merge_dpu_write_sequences`)**:
hcomm `WriteWithNotify` for len > maxMsgSize large messages splits into N-1 normal RDMA_WRITE +
1 trailing WRITE_WITH_IMM (whole operation still only reports one); collection side per WR stored as N-1 MEM_CPY +
1 WRITE_WITH_NOTIFY. pass scans src/dst dual-address contiguous MEM_CPY chain by DPU stream order,
terminated by address-contiguous WRITE_WITH_NOTIFY merged into single entry (len whole chain accumulated, description follows trailing task containing
notifyId, occupies trailing slot); chains without trailing not merged (independent pure write remains per-entry `RDMASend`);
only `src_type==1` data tasks participate, other tasks retained as-is without cutting chain.

> **Verification** (222 networking 8MB all_reduce): first round (20260921) 15 `Unknown` -> 0
> (`Dpu_Notify_Record` x5 / `Dpu_Notify_Wait` x9 / `Dpu_Write_With_Notify` x4,
> link=ROCE), peer rank=2/4/6 (real cross-pod peer), `eid_unresolved=0`,
> WRITE_WITH_NOTIFY enters `d2h` port charging. Final version (20260922, after primitive interception):
> estimate (-np 1) makespan 166.596us, primitive interception 16 times, precise synthesis 11 times, miss=0;
> normal (-np 8) checker All Success (primitive pure forwarding, interception log 0 entries); single machine 8p
> (118) SQE path regression no change.

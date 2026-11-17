# light_performance Plugin Design Document (Including Simulator Design)

> Document location: `docs/light_performance/simulator_design.md`
> Requirement source: `docs/light_performance/HCCLSIM/需求描述`
> Reference implementation: `src/plugin/checker/src/framework/task_graph_generator_v3/` (graph building principles),
> `docs/light_performance/HCCLSIM/hccl_sim/` (simulator prototype, **authoritative baseline for the simulation module**)
> Alignment backlog and matching checklist: Chapter 10
> db task / CCU instruction → JSON task type mapping table: Chapter 11

---

## 1. Requirement Background and Document Description

### 1.1 End-to-End Requirement

Integrate light_performance (lightweight performance simulation tool) as an independent plugin into the current project, completing the following end-to-end flow:

```
hccl-vm flow execution ──generate──> task sequence / instruction db files
        │
        ▼
light_performance plugin reads db files ──generate──> task graph for performance simulation
        │
        ▼
Performance simulation module runs simulation based on the task graph
        │
        ▼
Output performance profiling JSON file
```

### 1.2 Sub-Requirement Breakdown and Chapter Mapping

| Sub-Requirement | Content | Status | Chapter |
|----------------|---------|--------|---------|
| Sub-Req 1 | Plugin engineering: CLI, build and installation, complete plugin architecture | CLI and installation design detailed (@estimate one-shot execution channel) | Chapter 3 |
| Sub-Req 2 | Integration with hccl-vm: db file reading, task graph generation | **Detailed design in this iteration** | Chapters 4, 5 |
| Sub-Req 3 | CCU mode task graph generation (SQE queue + microcode instruction sequence → task graph) | **Detailed design in this iteration** | Chapter 6 |
| Sub-Req 4 | Performance simulation module (task graph parsing, performance simulation) | **Aligned and incorporated per HCCLSIM baseline** | Chapter 7 |
| Sub-Req 5 | Performance profiling JSON output (performance metrics, task graph) | High-level design | Chapter 8 |

> Chapter 10 records the **field gaps, type mismatches, and other conflict points** identified during this alignment (differences between the db graph building path and the HCCLSIM simulation model) for future matching reference. Chapter 11, Section 11.4 provides the mapping between v3 collected profInfo (`HcclTaskProfInfo` three fields) and JSON display types, along with the switchover target state.

### 1.3 Delivery Scope for This Iteration

This iteration focuses exclusively on **db file reading, parsing, and the graph building module**:

- **Input**: db files under the hccl_vm_install/data directory (format and data reference: `hccl_vm_install/data/hccl_vm_data.db`).
- **Output**: Task graph `DependencyGraph` and `SimConfig` required by the performance module (hccl_sim).
- **Reference code**:
  - Basic graph building logic: `src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_generator_v3.cc/h`;
  - CCU microcode instruction to task graph: `src/plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/`;
  - AIV mode graph building: not considered at this time.
- **Graph building scope (core design premise)**: The db contains **single-rank task data**, and this plugin performs **single-rank graph building** —
  this is one of the primary differences from checker (multi-rank joint graph building). All **cross-rank task edges
  (post-wait relationships) are ignored**: only post-wait edges within the single rank are considered; cross-rank edges are excluded
  (see Sections 5.1, 5.4, and 6.6).
- **Relationship with checker**: The output structure of the light_performance graph building module is not identical to checker, but the basic graph building and
  instruction translation principles are the same; the former does not require checker's subsequent memory conflict verification / semantic verification operations, so appropriate simplification is possible
  (difference comparison in Sections 5.8 and 6.7).

---

## 2. Plugin Overall Design

### 2.1 End-to-End Data Flow and Module Division

```
┌────────────────────────── light_performance plugin ──────────────────────────┐
│                                                                           │
│  ┌───────────┐   ┌───────────────┐   ┌───────────┐   ┌────────────────┐  │
│  │ db_reader  │──>│ graph_builder │──>│ simulator │──>│ profiling      │  │
│  │ (Ch. 4)    │   │ (Ch. 5, 6)    │   │ (Ch. 7)   │   │ (Ch. 8)        │  │
│  └───────────┘   └───────────────┘   └───────────┘   └────────────────┘  │
│   sqlite reading    SQE→Task mapping    simpy discrete     json + trace    │
│   BLOB decoding     notify pairing      event port bandwidth sharing       │
│   operator grouping CCU microcode       on-chip memory dynamic sharing     │
│                     expansion           eid→port resolution                │
│  Input: data/hccl_vm_data.db (+ data/hccl_sim.db, ranktable.json)          │
│  Output: (SimConfig, DependencyGraph, TopoResolver) → makespan →          │
│        perf_profiling.json (+ graph dump: DOT/JSON/HTML)                  │
└───────────────────────────────────────────────────────────────────────────┘
```

Data contracts between modules:

```
db_reader     ──OpExecutionGroup(intermediate data model, Sec. 4.5)──────────>  graph_builder
graph_builder ──(SimConfig, DependencyGraph)─────────────────>  simulator   [output of this requirement]
simulator     ──result dict(tasks with start/end + makespan + ports)──>  profiling/report
```

### 2.2 Plugin Directory Structure (Source Location: `src/plugin/light_performance`)

```
src/plugin/light_performance/
├── __init__.py
├── __main__.py             # Standalone CLI entry point (Sec. 3.3)
├── main.py                 # @estimate plugin one-shot execution entry (Sec. 3.1/3.2)
├── manifest.json           # Plugin manifest (name=estimate; installed with package)
├── pipeline.py             # Shared pipeline: db → graph → simulation → perf_profiling.json
│                           #   (shared by run_perf_sim.py and main.py, Sec. 3.1)
├── db_reader.py            # Sub-Req 2: db reading and BLOB decoding (Chapter 4)
├── sqe_group_merge.py      # SQE batch merge preprocessing: merge UDMA data tasks within same (stream, sqeGroupId) batch into 1 (Sec. 5.1.1)
├── graph_builder.py        # Sub-Req 2: SQE → DependencyGraph (Chapter 5)
├── graph_dumper.py         # Task graph dump: DOT + JSON + HTML
├── ccu_graph_builder.py    # Sub-Req 3: CCU microcode → task graph (Chapter 6)
├── register_model.py       # CCU register/Event execution model (Sec. 6.4)
├── sim_config.json         # Default latency/bandwidth configuration (Sec. 7.2, units us/GBps; installed with package)
├── bw_profile.json         # Default bandwidth utilization configuration (installed with package)
└── simulator/              # Sub-Req 4: simulation module (transferred from HCCLSIM, Chapter 7)
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

> The `simulator/` sub-package is transferred from `docs/light_performance/HCCLSIM/hccl_sim/`, with interfaces based on the HCCLSIM baseline (Chapter 7). The `dump_graph.py` at the repository root serves as the one-click dump entry point
> (db → DOT/JSON/HTML, optionally with simulation timestamps).
> `manifest.json` and `main.py` (one-shot execution entry, Chapter 3) are placed directly within this package directory.
> During installation, the entire directory becomes the `plugin/light_performance` plugin directory
> (installation layout in Sec. 3.1, CMake rules in Sec. 3.4).

### 2.3 Relationship with Existing Plugins

| Plugin | Responsibility | Relationship with light_performance |
|--------|---------------|-------------------------------------|
| checker | Communication correctness verification (deadlock/memory conflict/semantics) | Reuses its graph building principles; key difference: checker performs **multi-rank joint graph building**, light_performance performs **single-rank graph building** (cross-rank pairing edges are not built; remote synchronization tasks are retained and approximated with fixed remote latency), and no verification post-processing is performed |
| runner | Virtual runtime (task-by-task simulated execution of the data plane) | Also consumes opTask/ccuInstr; light_performance is read-only and does not advance isDone |
| insight | Visualization | light_performance profiling JSON / Chrome trace can be consumed by it (future) |

> **Plugin tag versus directory/module name**: This plugin uses the installation tag `estimate` (the host machine `@estimate` command channel
> and mode gating are hardcoded by tag, see Sec. 3.1), while the plugin directory and Python package name are both
> `light_performance` (`plugin/light_performance`) — the tag corresponds to the "host machine routing identity,"
> and the directory/package name corresponds to the "code organization identity."

---

## 3. Plugin Engineering Design (Sub-Requirement 1)

### 3.1 Plugin Identification and manifest.json

**The plugin tag is fixed as `estimate`** (rather than the module name light_performance): the host machine CLI has
hardcoded gating and path conventions for this plugin —

- The execution channel for `hccl-vm plugin run @estimate` is in `src/cmd/cmd_base_utils.cc:599-680`
  (`RunUserPlugin`) with special-case handling for `estimate`: only available in estimate mode (`sim::IsEstimateMode()`),
  the plugin directory is fixed as `<installRoot>/plugin/light_performance` (tag decoupled from directory name),
  directly reads its `manifest.json` and synchronously executes the `entry` command (see Sec. 3.2);
- Mode exclusion: `@checker` is unavailable in estimate mode (`cmd_base_utils.cc:618-624`),
  `@estimate` is unavailable in non-estimate mode — correctness verification and performance simulation belong to the closing actions of two separate running modes;
- The plugin scan/daemon model (`src/plugin/hccl_plugin_manager.cc:94-169`, `src/plugin/hccl_plugin.cc`)
  matches the tag by the manifest `name` field (scans `<installRoot>/plugin`, depth ≤ 2).

manifest.json (source and installation both within the package directory: `src/plugin/light_performance/manifest.json`
→ `plugin/light_performance/manifest.json`):

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

- `name`: Plugin tag; the host machine routes by it (`hccl-vm plugin run @estimate`).
- `entry`: Startup command within the plugin directory. @estimate uses the one-shot synchronous execution channel (Sec. 3.2);
  during execution, cwd is the plugin directory, and relative paths are resolved from the plugin directory.
- `setting`: Plugin-private configuration key-value pairs, read by main.py itself (following checker's `SettingManager`
  self-managed configuration approach; the Python side parses JSON directly). Path-type configurations are resolved relative to the plugin directory (the package directory):
  `db` defaults to `<installRoot>/data/hccl_vm_data.db` (same source as checker's
  `Loader::LoadOpTaskFile`); `sim_config`/`bw_profile` are Sec. 7.2 format
  latency/bandwidth and utilization configurations, defaulting to the same directory as the manifest, installed with the package as
  `sim_config.json`/`bw_profile.json`;
  `ranktable` defaults to `ranktable.json` in the same directory as the db;
  `device`/`iter` correspond to the CLI `--device`/`--iter`; `out_dir` is the output root directory
  (sub-directories are created under it by run timestamp); `per_op_mode` currently supports only per-operator (when false, a warning is issued
  and per-operator execution proceeds); `enable_trace`/`dump_graph`/`max_trace_events` control supplementary artifacts.

Source and installation layout (the plugin directory is the package directory, with no nesting):

```
source = installation artifact structure (src/plugin/light_performance → hccl_vm_install/plugin/light_performance):
plugin(light_performance)/            # Plugin directory = light_performance package directory
├── manifest.json                     # Plugin manifest (name=estimate)
├── main.py                           # @estimate one-shot execution entry (reuses pipeline.py)
├── __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
├── sim_config.json                   # Default latency/bandwidth (Sec. 7.2)
├── bw_profile.json                   # Bandwidth utilization
├── simulator/                        # Simulation sub-package (Chapter 7)
└── data/                             # Runtime output (perf_profiling.json etc., Ch. 8;
                                      #   produced only in installation tree; not present in source tree)
```

> main.py resides within the package directory but is executed as a script (`python3 main.py`): it adds the package's **parent directory**
> (`plugin/`, source tree is `src/plugin/`) to `sys.path` before performing
> `from light_performance.pipeline import ...` — intra-package relative imports are unaffected,
> and the source tree and installation tree layouts are consistent.
> The `run_perf_sim.py` at the repository root (manual entry point) shares the same pipeline implementation with `main.py`:
> main.py adds "installation tree path conventions + manifest setting reading" on top of it.
> Source tree debugging still uses run_perf_sim.py (Sec. 3.3); within the installation tree, `hccl-vm plugin run @estimate`
> triggers main.py.

### 3.2 Invocation Chain and Process Model (@estimate One-Shot Execution)

The host machine side requires only one path adaptation — the one-shot execution channel for `@estimate` was already implemented
(`cmd_base_utils.cc:599-680`); this iteration changes the plugin directory in the special-case branch from `plugin/estimate`
to `plugin/light_performance` (tag remains `estimate`, directory is the package directory):

```
hccl-vm plugin run @estimate
    │
    ▼
RunUserPlugin("@estimate")                                  # subcmd_plugin.cc:55-57 routing
    ├─ Gate: must be in estimate mode, otherwise error exit  # cmd_base_utils.cc:611-617
    ├─ Read <installRoot>/plugin/light_performance/manifest.json  # Missing file or missing entry field → error
    └─ system("cd <installRoot>/plugin/light_performance && python3 main.py")   # Synchronous blocking execution
       └─ Non-zero exit code → HCCL_VM_ERROR, command returns failure overall
```

**Client wait strategy for long-duration commands** (`StartHostClient`/`ServerListen` in `cmd_base_utils.cc`):
within a session, `hccl-vm` sends to the host machine through a POSIX message queue; **the host machine replies only after `ParseCommand` returns (the command has finished executing)**; after CCU CcuBuffer mode microcode expansion, the task count can reach 10^5 level (measured 192,005 tasks, software simulation approximately 54s),
far exceeding the 5s reply limit for normal commands. To address this:

- The client identifies `plugin run @estimate` and adopts a long wait upper limit (default 30 minutes, environment variable
  `HVM_CLIENT_RESP_TIMEOUT_MS` can override); during segmented waiting, it prints
  "command still executing on host machine" to the terminal every 30s (first at 5s), and prints a clear warning upon timeout; other commands maintain the 5s limit.
- The reply carries the original command string (`ACK:<cmd>`); the client drains the reply queue before sending a command and only accepts replies that strictly match the current command — preventing misreading of late replies from previously timed-out commands that would cause premature return (especially important during command interruption retries within the same session).
- The host machine prints `[HVM] @estimate performance simulation started/finished` markers to the terminal at simulation start and end (stdout, along with report
  output); this mechanism ensures **the prompt returns only after simulation truly finishes**, eliminating the misconception that "simulation has finished."

main.py (plugin side) execution flow:

```
RunEstimate():
    setting   ← Parse manifest.json setting section (paths resolved relative to plugin directory)
    db        ← setting.db (default <installRoot>/data/hccl_vm_data.db, derived as ../../data from __file__)
    config    ← setting.sim_config / bw_profile (use built-in defaults when missing, Sec. 7.2)
    ranktable ← ranktable.json in same directory as db (_find_ranktable default rule; consistent with host machine InitHvmCommEnv
                writing data/ranktable.json, cmd_base_utils.cc:437-450, no configuration needed)
    groups    ← db_reader.load_op_executions(db)            # Chapter 4
    topo      ← TopoResolver(ranktable)                     # Sec. 7.6
    for group in groups:                                    # Per-operator (Sec. 5.1)
        config, dep_graph = graph_builder.build(group)      # Chapters 5, 6
        result = HcclSimulator(config, dep_graph, topo).run()   # Chapter 7
    profiling.dump(<plugin>/light_performance/data/<timestamp>/perf_profiling.json)   # Chapter 8
```

Comparison with the checker daemon process model (stdin single-line JSON command loop) and rationale for the choice:

| Dimension | checker (daemon model) | estimate (one-shot model) |
|-----------|----------------------|--------------------------|
| Startup method | fork+pipe daemon at `plugin install` (`hccl_plugin.cc:74-156`) | `cd && entry` synchronous execution at `plugin run` (`cmd_base_utils.cc:655-660`) |
| Trigger protocol | stdin JSON: `{"action":"status","payload":{"status":"finish","data_id":...}}` | No protocol: entry is the startup; input paths are all through conventions + manifest setting |
| data_id | Carried in message (generated by host machine `DumpData`) | Not passed; output directory uses run timestamp as substitute (Sec. 10.3 #13) |
| Lifecycle | Reclaimed on stop command/pipe EOF/host machine exit | Process exits naturally; exit code is the execution result |
| Adaptation rationale | Verification needs to be triggered by the host machine at flow end and may run multiple times | Simulation is offline batch processing: consumes already-persisted db, runs once and exits |

> **Evolution direction** (not implemented in this iteration): if "automatic simulation trigger by the host machine at flow end" (daemon model) is needed later,
> main.py can add a stdin command loop to reuse the same pipeline — the host machine's message channel
> (`SendMessageToPlugin`) and daemon startup (`RegisterPlugin`) are both ready; at that point, only the
> @estimate special-case branch in `RunUserPlugin` needs to be switched back to the general message path.

Data contracts (fixed conventions within the installation tree, all aligned with existing mechanisms):

| Path | Source | Description |
|------|--------|-------------|
| `data/hccl_vm_data.db` | Written during host machine flow execution | Primary simulation input; estimate mode produces **single-rank task data** (phantom peer), naturally aligned with the Sec. 4.6/5.1 single-rank graph building premise |
| `data/ranktable.json` | Written by `InitHvmCommEnv` | eid→port resolution (Sec. 7.6); `_find_ranktable` defaults to the same directory as the db; no explicit configuration needed |
| `plugin/light_performance/sim_config.json`, `bw_profile.json` | Installed with the package directory by CMake (Sec. 3.4) | Sec. 7.2 format (units us/GBps) |
| `plugin/light_performance/data/<timestamp>/` | Created at runtime by main.py | Output directory. **Must not be placed under `data/`**: the host machine runs `remove_all("data")` at each start (`src/main.cc:32`); contents of the data directory are not preserved across sessions |

### 3.3 Command Line Design

Host machine integration mode (`plugin run` is the simulation entry point, replacing manual execution of run_perf_sim.py):

```bash
# ① Start environment in estimate mode (choose one)
hccl-vm start <topo> --estimate             # Interactive mode, PS1 displays (estimate)$>
hccl-vm run <topo> --estimate <sample command>     # One-shot mode
# (Within an already started session, also switch via hccl-vm mode estimate, subcmd_mode.cc)

# ② Run workload → produce data/hccl_vm_data.db (single-rank task list)

# ③ One-click performance simulation (only available in estimate mode, Sec. 3.2 gating)
hccl-vm plugin run @estimate
# Artifacts: plugin/light_performance/data/<timestamp>/perf_profiling.json (+ trace / graph dump)
```

- The one-shot execution channel does not pass through command-line arguments; simulation parameters (target device, opIter filter, trace/dump switches, etc.)
  are configured through the manifest `setting` section (Sec. 3.1); re-run `plugin run @estimate` after changing the configuration for changes to take effect.
- `plugin install @estimate` is also executable (through the `RegisterPlugin` daemon startup channel); main.py completing in one shot and exiting is acceptable degraded behavior (Sec. 10.3 #14); routine use does not require install.

Independent debugging mode (does not depend on the host machine, convenient for offline db analysis; already implemented):

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
| `--config` | Latency/bandwidth configuration (Sec. 7.2 format, units us/GBps; uses built-in defaults by default) |
| `--bw-profile` | Bandwidth utilization configuration (default utilization is 100%) |
| `--jetty-num` | Override the UB multi-channel (jetty) count parsed from the db, for multi-jetty utilization scaling (default follows "db actual link establishment value (`RaJetty`/`ccuChannels`) preferred, `opDetails.ubMultiChannelNum` as fallback" per-operator parsing, see Sec. 7.2) |
| `--ranktable` | Topology file for eid → physical port resolution (defaults to `ranktable.json` in the same directory as the db; when network tasks exist but no topology is available, **skips simulation and only dumps**) |
| `--device` | Target device (deviceId) for single-rank graph building; defaults to the first device in opDetails; only needed when the db contains multi-device data |
| `--iter` | Simulate only the operator with the specified opIter (default: all) |
| `--trace` | Export Chrome tracing JSON (for multiple operators, files are split by `_<commId>_<opIter>`) |
| `--dump-dir` | Task graph dump directory (per-operator output of .dot/.json/.html, including simulation timestamps) |
| `-v` / `-s` | Task-level detail / summary only |

### 3.4 Build and Installation Design (hccl_vm_install/plugin/light_performance)

Add an install rule in `src/plugin/CMakeLists.txt` (following the precedent of insight's directory installation;
Python plugins require no build targets. The plugin directory is the package directory; manifest.json/main.py/default configurations are all
within the package directory, so a single DIRECTORY rule carries everything):

```cmake
# estimate performance simulation plugin: plugin directory is the light_performance package directory, install entire directory
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/light_performance"
    DESTINATION "plugin"
    FILE_PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
    PATTERN "__pycache__" EXCLUDE
    PATTERN "data" EXCLUDE)      # Runtime output directory not included in installation package
```

Comparison with existing plugin installation methods:

| Plugin | Installation Method | Artifact |
|--------|-------------------|----------|
| checker | `install(TARGETS checker DESTINATION plugin/checker)` + `install(FILES manifest.json)` (`src/plugin/checker/CMakeLists.txt:67-82`) | Build artifact + manifest |
| insight | `install(DIRECTORY insight DESTINATION plugin/visualization PATTERN frontend EXCLUDE)` (`src/plugin/CMakeLists.txt:26-39`) | Python directory + manifest |
| **estimate (this iteration)** | `install(DIRECTORY light_performance DESTINATION plugin)` (single rule; manifest/main.py/default configurations all within the package directory, installed with the package) | `plugin/light_performance/`: Python package + manifest (plugin directory and package directory unified, no nesting) |

Build and verify: after `bash build.sh` (internally `cmake && make && make install`, installation prefix defaults to
`hccl_vm_install/` within the repository, `CMakeLists.txt:89-92`), check the installation artifacts:

```
hccl_vm_install/plugin/light_performance/
├── manifest.json
├── main.py
├── __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
├── sim_config.json / bw_profile.json
└── simulator/ ...
```

- Packaging path: `build.sh --pkg` packages the entire `hccl_vm_install` into a tar archive; the estimate plugin is distributed along with it.
- Dependencies: python3 (≥3.8, including sqlite3 standard library) and **simpy** (discrete event simulation engine, current environment
  4.1.2, imported by `simulator/simulator.py`); the deployment environment must have these pre-installed (the insight plugin also depends on python3,
  with consistent conventions); see Sec. 10.3 #16.

---

## 4. db File Reading and Parsing (Sub-Requirement 2 - First Half, Focus of This Iteration)

### 4.1 db Location and Opening

- The path is fixed as `<installRoot>/data/hccl_vm_data.db` (consistent with checker's `Loader::LoadOpTaskFile`).
- Read-only opening: Python `sqlite3` standard library, `mode=ro` URI, to avoid write conflicts with the running hccl-vm:
  ```python
  conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
  ```
- Auxiliary database `hccl_sim.db` (same directory): resolves `opDetails.commId` to communicator identity
  (`Communicator` table, `comm_id[128] + comm_hash` unique identifier); eid→rank index
  (`RankEidMap` table, Sec. 4.7), `Server.port_type` / `Device.soc_version`
  (port bandwidth table selection), **actual jetty count** (`RaJetty`/`RaContext`/`EndPoint` tables,
  primary data source for multi-jetty scaling, see Sec. 7.2). All are optional; each falls back independently when missing.

### 4.2 Table Inventory and Association Model

hccl_vm_data.db is created by `OpDbOps::InitSchema()` (`src/runnerdb/modeldb/db_hccl_op_db_ops.cc:76-184`);
the tables consumed by light_performance are as follows:

| Table | Primary Key/Association | Content | Purpose |
|-------|------------------------|---------|---------|
| `opDetails` | `id`; `pid`→opTask table name | One row per operator per rank: pid/deviceId/rankId/commId/opIter/syncIter/streamId(main stream)/root/rankSize/srcRank/dstRank + `opDetail` BLOB (operator parameters) + `algorithmName` (algorithm selection result, captured and backfilled through DFX hijack/logging dual paths, see note below) + `ubMultiChannelNum` (UB multi-channel/jetty count; proxy level0 stub reads environment variable `HCCL_UB_MULTI_CHANNEL_NUM` and stores in db; defaults to 1 when unconfigured/invalid; represents **business configuration intent**; during scaling, db actual link establishment value is preferred and this column serves as fallback, see Sec. 7.2 multi-jetty scaling) | Operator grouping, main stream identification, operator parameters, algorithm selection results, bandwidth utilization scaling |
| `opMemInfo` | `opDetailId` → opDetails.id | Addresses and sizes of input/output/ccl three memory segments | Direction determination (Sec. 5.5), data volume validation |
| `opTask_P_<pid>` | `opDetailId` → opDetails.id | SQE task queue: deviceId/streamId/taskType/isDone + `opTaskMeta` BLOB | **Primary graph building input** |
| `ccuChannels` | `(deviceId, dieId, channelId)` | CCU channels: src/dst die/device/rank + leid/reid(16B EID) + protocol + jettyNum + jettyId array | CCU channel → peer/jetty mapping (Chapter 6) |
| `ccuInstrRes` | `(deviceId, dieId)` | Microcode instruction space per die: instrCount + instrSpace BLOB (32K×32B) | CCU microcode source (Chapter 6) |
| `ccuInstr` | `ccuInstrResId` → ccuInstrRes.id | One row per microcode dispatch (startId/instrInfoSize) | Used only to detect "new microcode dispatched"; decoding does not depend on it |
| `syncRecords` | — | Host-side stream synchronization records | Not used in this iteration |
| `JettyMaps` | `opDetailId` | eid mapping recorded per operator | Backup (not used when ccuChannels is available) |

> Table structure definitions are in `include/runnerdb/sim_op_db_types.h`; writer-side semantics are in `src/runnerdb/tracedb/db_sim_op_db_ops.cc`.

> **Source of the algorithmName column (algorithm selection result capture)**: Dual-path capture, both last-wins backfilling
> `sim::UpdateOpDetailAlgName`, serving as mutual backups and writing the same value.
>
> 1. **DFX hijack stub (primary path)**: After algorithm selection completes, the business calls
>    `HcclDfxRegOpInfoByCommId(commName, &HcclDfxOpInfoCompat)` per operator (hccl repository
>    op_common.cc); its `algTag` carries the selected algorithm name (format
>    `{opType_commName}_{algName}_{host|device}[_{dataType}...]`, graph mode
>    `Graph_{algName}_{...}`). Level=2 mode (default) only hooks level0+level2;
>    the level1 dfx_stub is not in the process; `src/proxy/level2/hcomm_api_stub.cc` uses
>    level2 strong symbols to preempt resolution (preload order precedes the real
>    libhcomm.so loaded by the business through the Python plugin chain), parses the algorithm name from algTag and backfills it, then
>    **forwards to the real implementation through `DlsymRealWithFallback` (RTLD_NEXT + libhcomm.so SONAME fallback + dladdr
>    self-call prevention), passes through the return value, and does not alter call semantics**. The struct layout is replicated from
>    hccl_res_dl.h (`static_assert` locks algTag offset at 68); content validation
>    (commId prefix + host/device suffix) filters out op-begin registrations without algorithm names and
>    provides fallback for business header drift. Under Level=1 mode, this stub is shadowed by the level1 dfx_stub (behavior
>    same as the old version), and capture falls back to the logging path.
> 2. **Logging capture (backup path)**: The tool is the implementer of the business logging underlying interface (the business
>    `DlogRecord` weak symbol resolves to `src/proxy/level2/log_stub.cc`). The level0 stub
>    `RecordOpDbInfo` first inserts the opDetails row (setting the runnerdb global `g_currOpDetailId`) then calls
>    the real business function; the INFO-level algorithm selection log output by the selector within the business (`[HcclGetAlgRes] algName=` /
>    `SelectMinCost: selected algName=` / old selector `the selected algo type is ` three types of markers,
>    matched multiple times per operator with last-write-wins) is matched and extracted through the `DlogPrintStub` entry point. The capture executes before the message enters the tool log output
>    (immune to `log_config.yaml` level); the proxy layer also provides an `acllogCheckDebugLevel`
>    always-pass stub (immune to business-side log level thresholds).
>
> Old databases are automatically supplemented with `ALTER TABLE` by `OpDbOps::InitSchema()`;
> the plugin reader falls back to empty values for old databases without the column.

**Operator grouping model**: One operator instance = rankSize opDetails rows under the same `(commId, opIter)` (one per rank),
each with one opMemInfo and a set of tasks (`WHERE opDetailId=?`, table name derived from the row's `pid` column as `opTask_P_<pid>`,
referencing checker `QueryCompositeOpDetailByOpIter`).

### 4.3 opTask Dynamic Tables and Multi-pid Handling

- Each task-writing process creates its own table `"opTask_P_" + getpid()`; AICPU device-side processes use `getppid()` to share the same table as the host machine.
- **Discover all task tables** (enumeration is mandatory; multi-pid is the norm; early sample db already had `opTask_P_477776`/`opTask_P_477780` two tables; current sample has a single table):
  ```sql
  SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'opTask_P_%'
  ```
- Consumption order: within the same operator, by `id` ascending order (= SQE dispatch order; this is the basis for notify first-come-first-served pairing).

### 4.4 BLOB Decoding Specification (pack(1), Little-Endian)

All BLOBs are raw C struct memcpy (no serialization framework); the Python side uses the `struct` module for decoding.
The following layouts have all been verified with `hccl_vm_install/data/hccl_vm_data.db` (from v5 collection onward, sample
opTaskMeta length is 190B; historical samples were 189B/188B).

#### 4.4.1 opTaskMeta (190 Bytes)

The struct definition is shared with the writer side: `include/sim_profiling_defs.h` (single-source definition of profiling reconciliation information)
and `src/plugin/checker/header/internal/task_meta_defs.h` (HcclTaskMetaData).

v5 collection structure (compared to v4's 189B version): `taskSrc` (1 byte) added after `sqeGroupId` —
task source marker (`HcclTaskSrc`: 0=default SQE/aclrt native path, 1=DPU i.e. host CPU RoCE
`ibv_dpu_stub` collection path), enabling the performance simulation DPU sequence merge pass to precisely split by source
(Sec. 5.1.2); taskData start offset 69 → 70.

v4 collection structure (compared to v3's 188B version): `sqeGroupId` (1 byte) added after `jettyId` — UDMA
SQE batch grouping marker (all WQEs within one doorbell share the same group), enabling the performance simulation entry to merge by batch
(Sec. 5.1.1); taskData start offset 68 → 69. Group number semantics: **cyclically incrementing doorbell batch number 1..255**
(wraps at 255; the collection side `device_sqe_parse_stub.cc` maintains global `g_currSqeGroupId`,
`% 255 + 1`) — batch numbers are unique within the 255 window; doorbells from other operators are not visible in this operator's task list
but equally consume batch numbers, so the same number within the same stream must belong to the same doorbell; **0 is reserved as "unmarked"**
(HcclTaskMetaData construction default); the counter never takes 0; the merge entry treats `sqeGroupId <= 0`
as non-participating (Sec. 5.1.1).

v3 collection structure (compared to v2's 185B version): `HcclTaskProfInfo` sub-structure (3 consecutive bytes,
checker/runner pass through as opaque) added after `taskType`: `profTaskType` (aligned with hcomm TaskParamType value range 0~22) and
`linkType`/`transportType` (aligned with hcomm DfxLinkType value range 0~9 / DfxTransportType value range 0~4,
used for reconciliation with profiling data); taskData start offset 65 → 68.

> **Implementation status**: `db_reader.py` auto-identifies the layout by BLOB length — `>= 190B` uses v5
> (`<BBBBQIQQIBB16s16s`, taskSrc decoded into `HcclTaskMetaData.src_type`; this branch
> **must precede the v4 check**, otherwise 190B would first match the `>= 189` v4 branch causing eid/taskData
> offset misalignment), `>= 189B` uses v4
> (`<BBBBQIQQIB16s16s`, sqeGroupId decoded into `RawTask.sqe_group_id`, src_type=0),
> `188B` uses v3 (`<BBBBQIQQI16s16s`, sqe_group_id filled with -1, non-participating in merge),
> `185~187B` uses old v2 (`<BQIQQI16s16s`, profInfo filled with 0xFF=UNKNOWN) for backward compatibility,
> `< 185B` is discarded with a warning. profInfo consumption (`task_type` display name + link/transport)
> **has been implemented** (#18, Sec. 11.4).

v2 collection structure (compared to earlier 169B version): `lcEid[16]` (explicit local EID) added in the header;
taskData members previously had `srcRankId/dstRankId` fields supplemented.

> **rank field rollback: srcRankId/dstRankId are deprecated (current convention)**: the `srcRankId/dstRankId` fields
> previously supplemented in taskData members have been rolled back and deleted, uniformly replaced by `srcDeviceId/dstDeviceId` — the collection side
> (estimate mode, `device_sqe_parse_stub.cc` etc.) writes the communicator rankId directly into
> the `srcDeviceId/dstDeviceId` fields, i.e. **the deviceId field carries rankId** (peer rank unresolved
> is `0xFFFFFFFF`). `db_reader.py` consumes the unpacked values directly under the names `src_rank_id/dst_rank_id`
> (reads `srcDeviceId` as `srcRankId`). This is a hard switch: old BLOBs collected before the field rollback
> (dual rank/device field layout) are no longer supported.

```
Header 70B:  struct.unpack_from("<BBBBQIQQIBB16s16s", blob, 0)
┌──────┬──────┬──────────────┬────────┬───────────┬──────────────────────┐
│Offset│Length│Field         │Type    │Sentinel   │Description           │
├──────┼──────┼──────────────┼────────┼───────────┼──────────────────────┤
│ 0    │ 1    │ taskType     │ u8 enum│ —         │ Redundant with table column taskType │
│ 1~3  │ 3    │ profInfo     │ sub-struct│ 0xFF   │ HcclTaskProfInfo:      │
│      │  1   │ .profTaskType│ u8 enum│ 0xFF      │ Aligned with hcomm TaskParamType │
│      │  1   │ .linkType    │ u8 enum│ 0xFF      │ Aligned with hcomm DfxLinkType │
│      │  1   │ .transportType│u8 enum│ 0xFF      │ Aligned with hcomm DfxTransportType │
│ 4    │ 8    │ commId       │ u64    │ —         │ Communicator ID         │
│ 12   │ 4    │ rankId       │ u32    │ 0xFFFFFFFF│ Backfilled via device→rank when default │
│ 16   │ 8    │ deviceId     │ u64    │ —         │ Physical device ID      │
│ 24   │ 8    │ streamId     │ u64    │ —         │ Stream ID              │
│ 32   │ 4    │ jettyId      │ u32    │ 0xFFFFFFFF│ Invalid = not network   │
│ 36   │ 1    │ sqeGroupId   │ u8     │ 0         │ Doorbell batch number 1..255 cyclic increment (v4 new, Sec. 5.1.1); 0 = unmarked│
│ 37   │ 1    │ taskSrc      │ u8 enum│ 0         │ Task source (v5 new, Sec. 11.4.6): 0=default, 1=DPU(host ibv)│
│ 38   │ 16   │ rmEid[16]    │ bytes  │ all 0     │ Remote EID             │
│ 54   │ 16   │ lcEid[16]    │ bytes  │ all 0     │ Local EID (v2 new)     │
│ 70   │ 120  │ taskData     │ union  │ Interpreted by taskType, see below │
└──────┴──────┴──────────────┴────────┴───────────┴──────────────────────┘
```

taskData union internal layout (relative offset 69):

| taskType | Member | Format String | Fields (in order) |
|----------|--------|---------------|-------------------|
| 0/1 (notify) | `NotifyTask` | `<IQIBB` | srcDeviceId(carrying rankId), notifyId, dstDeviceId(carrying rankId), notifyCount, protocol |
| 3 (transMem) | `TransMemTask` | `<IQIQQB` | srcDeviceId(carrying rankId), srcOffset, dstDeviceId(carrying rankId), dstOffset, len, protocol |
| 2 (reduce) | `ReduceTask` | `<IQIQQBBBB` | srcDeviceId(carrying rankId), srcOffset, dstDeviceId(carrying rankId), dstOffset, dataCount(actually byte size), dataType, reduceOp, protocol, resv |
| 4 (ccu) | `CcuTask` | `<BBHHHII` + `<13Q` | dieId, missionId, timeout, instStartId, instCnt, key, argSize, args[13] |
| 5 (aiv) | `AivTask` | `<Q` | launchIdx |
| 6 (sync) | `SyncStreamTask` | `<Q` | syncIdx |
| 7 (writeWithNotify) | `WriteWithNotifyTask` | `<IQIQQQBBBBB` | srcDeviceId(carrying rankId), srcOffset, dstDeviceId(carrying rankId), dstOffset, len, notifyId, isReduce, dataType, reduceOp, protocol, notifyCount |

> Both `len` and `dataCount` are **byte sizes** (the reduce field name has historical reasons; it is actually used as size).
> `WriteWithNotifyTask.len` is also byte size; `isReduce=1` indicates the write payload is reduction data
> (`dataType`/`reduceOp` are valid in this case), mapped as a single `Write_With_Notify` task (Sec. 5.2).

#### 4.4.2 taskType Enumeration Full Table

`HccLTaskMetaType` (`include/sim_common_defs.h:184-193`):

| Value | Enum | light_performance Handling |
|-------|------|---------------------------|
| 0 | `NOTIFY_WAIT` | → local_wait / remote_wait (Sec. 5.2) |
| 1 | `NOTIFY_RECORD` | → local_record / remote_record |
| 2 | `REDUCE` | → memcpy / local_copy (reduction approximated, Sec. 5.7) |
| 3 | `MEM_CPY` | → memcpy / local_copy (determined by jettyId) |
| 4 | `CCU_GRAPH` | → CCU sub-graph expansion (Chapter 6) |
| 5 | `AIV_GRAPH` | Not supported; skipped with warning |
| 6 | `SYNC_STREAM` | → Skipped, no node generated (Sec. 5.6) |
| 7 | `WRITE_WITH_NOTIFY` | → Single-task mapping `Write_With_Notify` (network write + attached notify, not split, Sec. 5.2) |

#### 4.4.3 opDetails.opDetail (28 Bytes)

`OpDetails` (`task_meta_defs.h:141-165`): `<HHH` header + union:

```
opType u16@0 | dataType u16@2 | reduceType u16@4
opV1: count u64@6 | dataType u16@14 | reduceOpType u32@16          (total 28B)
opV2: sendCount u64@6 | sendDataType u16@14 | recvCount u64@16
      | recvDataType u16@24 | extInfo u16@26                       (AllToallV type)
```

opType is `HcclCMDType` (ALLREDUCE=2, ALLGATHER=6, REDUCE_SCATTER=7, ALLTOALL=10 etc.;
complete enumeration in `src/plugin/checker/header/external/hccl_types.h:162-189`), used for operator annotation in profiling reports.
`opExtInfo` BLOB (optional) stores V operator counts/displs, AllToallV send matrix etc.; this iteration only passes it through.

V1/V2 layouts are distinguished by opType (level0 stub writer-side convention, `hccl_op_stub.cc`):
**V2 is only for AllToAll(10)/AllToAllV(8)/AllToAllVC(9)** (`BuildOpDetailsV2`; sendCount/recvCount
are this rank's total send/receive element counts); **AllGatherV(16)/ReduceScatterV(17) use V1 + opExtInfo**
(count = this rank's side element count). When decoding, db_reader selects the layout by opType; V2 operators decode send/receive
element counts and types for both sides (`send_count/recv_count/send_data_type/recv_data_type`);
`data_size/data_type` follow the send side to maintain backward compatibility with old consumers; `reduce_op` is set to 0.

#### 4.4.4 ccuChannels BLOB

| Field | Format | Description |
|-------|--------|-------------|
| `leid` / `reid` | 16B raw bytes each | Source/destination endpoint EID (`URMA_EID_LEN=16`) |
| `jettyId` | `jettyNum × u32` little-endian | Contiguous jetty number array (C struct reserves `u32[64]`; sample measured jettyNum=1, jettyId=(1024,)) |
| `protocol` | int column | CommProtocol: HCCS=0/TCP=1/ROCE=2/UB_CTP=3/UB_TP=4/PCIE=5/SIO=6 |

Note: The `srcRankId/dstRankId` columns actually contain device_id (channel establishment precedes the communicator; see writer side
`hccp_ccu_stub.cc:401-405` comment); rank semantics require correction through opDetails.

#### 4.4.5 ccuInstrRes instrSpace

- `instrSpace` = **complete microcode instruction space snapshot for one die**: `32K entries × 32B = 1 MiB`
  (`CCU_INSTRUCTION_NUM=32*1024`, `CCU_INSTRUCTION_SIZE=32`, `include/sim_common_defs.h:35-36`).
- The i-th instruction is at byte offset `i*32`; `instrCount = max(historical value, startId+instrCnt)`, **allowing gaps**.
- When reading, only the first `instrCount` entries are taken; CCU task `instStartId/instCnt` is used directly as subscript range.
- The `ccuInstr` table is only for runtime awareness (runner polls `COUNT(ccuInstr)+COUNT(ccuInstrRes)` changes);
  light_performance decoding does not depend on this table.

### 4.5 Intermediate Data Model (db_reader Output)

```python
@dataclass
class OpExecutionGroup:          # One operator instance (single-rank perspective: task data for the target device only)
    comm_id: int
    op_iter: int
    op_type: int                 # HcclCMDType
    rank_size: int
    main_stream_ids: dict[int, int]      # deviceId → main stream streamId (opDetails.streamId)
    tasks: list[RawTask]         # All SQEs for this operator, globally ordered
    mem_info: dict[int, MemInfo]         # deviceId → input/output/ccl ranges
    channels: dict[tuple, ChannelInfo]   # (srcDev, srcDie, channelId) → channel info
    instr_space: dict[tuple, list[bytes]]# (deviceId, dieId) → 32B instruction list
    algorithm_name: str          # Algorithm selection result (opDetails.algorithmName, captured and backfilled through DFX/logging dual paths; empty string for old databases)
    ub_multi_channel_num: int    # UB multi-channel (jetty) count env stored value (opDetails.ubMultiChannelNum; 1 for old databases/zero values); configuration intent only; scaling uses db actual link establishment value first (Sec. 7.2)
    send_count / recv_count: int # Send/receive element counts for both sides (V2 layout operators; same as count for V1)
    send_data_type / recv_data_type: int # Send/receive data types for both sides (same as above)

@dataclass
class RawTask:
    global_id: int               # Global sequence across opTask_P_* tables (sorted by table pid value + in-table id)
    op_detail_id: int
    device_id: int
    stream_id: int
    meta: HcclTaskMetaData       # 4.4.1 decoding result (dataclass form)
```

`meta` contains the v2 new field: header `lc_eid` (explicit local EID, used with priority for Sec. 5.5 local eid resolution);
NotifyTaskData/TransMemTaskData both carry `src_rank_id/dst_rank_id`
(unpacked from the rankId value carried in the deviceId field under the estimate collection convention; see Sec. 4.4.1 rank field rollback).

### 4.6 Data Integrity Validation

For each `OpExecutionGroup`:

1. **Single-rank filter (design premise)**: The performance plugin consumes only **single-rank task data** (the sample db contains only device 1;
   this is a design norm, not data missingness). If the db contains multi-device data, filter opDetails and their tasks by target deviceId (`--rank`, defaults to first device);
   `rankSize` is passed through as metadata only; there is no requirement that `len(opDetails rows) == rankSize`.
2. **Notify peer** (no backfill):
    - Estimate collection convention (`device_sqe_parse_stub.cc`, consistent for both task types): the notify's
      `srcDeviceId` (carrying rankId) = record side, `dstDeviceId` (carrying rankId) = wait side,
      both explicitly filled by the sending end; peer rank is `0xFFFFFFFF` when unresolved.
    - The early `dstDeviceId == 0` backfill fallback was removed along with the rank field rollback (the 0 sentinel conflicts with valid
      rank 0, and backfill is unnecessary after the sending end always explicitly fills the value).
    - The notify carried by `WRITE_WITH_NOTIFY` (7) targets the **write peer** (the wait side is not in this db; single-rank
      simulation premise); it does not participate in local pairing; its `dstDeviceId` (carrying rankId) is the write destination.
    - Note: The runner/checker consumer side (`DecodeAndExpandTaskMeta`) splits 7 into
      [write, NOTIFY_RECORD] two sub-tasks at the decoding boundary; light_performance **does not split** — single-task mapping
      `Write_With_Notify` (Sec. 5.2), ensuring simulation tasks correspond one-to-one with db tasks.
3. **Decoding length validation**: `len(opTaskMeta) >= 185` (auto-adapts to v4/v3/v2 layout by length, Sec. 4.4.1),
   `len(opDetail) >= 28`,
   `len(instrSpace) >= instrCount*32`; rows that do not satisfy these are discarded with a warning
   (checker uses direct memcpy; this is more lenient).

### 4.7 Decoding Example (Measured hccl_vm_data.db)

Sample db (REDUCE_SCATTER, opType=7, dataType=4/FP32, count=8, rankSize=2, devType=6/950)
measured decoding results:

- `opTaskMeta` length is consistently 188B (v3 collection sample; from v4 collection onward it is 189B, Sec. 4.4.1); taskType distribution: 4 (CCU_GRAPH) and 6 (SYNC_STREAM) alternating.
- opDetails contains only device 1 (rank 0) → single-rank data (design norm); main stream is stream 2.
- Task sequence (main stream per opIter):

```
stream 2 (main stream):  CCU_GRAPH(die=1, mission=0, instStartId=14, instCnt=100, argSize=13)
                          SYNC_STREAM(syncIdx=n)          # Microcode dispatch + stream sync
```

- CCU task args match the data directory `sqe_info_device_1_die_1_mission_0_startId_14.txt`
  item by item (args[0]=281475405602881, args[1]=281475405602945 ...);
  `args[0]/args[1]` measured as opMemInfo's inputAddr/outputAddr.
- `ccuInstrRes` (dev 1, die 1) instrCount=114 = instStartId(14) + instCnt(100).
- Early 169B samples containing MEM_CPY/NOTIFY measured: MEM_CPY's srcOffset/dstOffset can both fall within
  opMemInfo's input/output/ccl ranges (`input ⊃ output`; ReduceScatter's output is the first
  1/rankSize segment of input); network copies with jettyId also exhibit "local range addresses" for their offsets
  (communicator memory symmetric layout; peer buffer has the same VA as the local side), therefore **direction determination cannot rely on addresses alone**
  (see Sec. 5.5). This conclusion also applies to v2 data.

#### 4.7.1 v2 AICPU Mode Sample (opTask_P_118800)

REDUCE_SCATTER, rankSize=8, count=32768 (FP32, output=128KB), 76 tasks,
stream 2 (main)/6 (dispatch)/7..13 (one stream per peer). Key differences from the CCU mode sample:

- taskType distribution: NOTIFY_WAIT×30, NOTIFY_RECORD×30, MEM_CPY×8 (7 network write
  jet=1..7 + 1 local), REDUCE×7 (local), SYNC_STREAM×1; no CCU_GRAPH, `ccuChannels` is empty.
- **Direction entirely determined by rank dimension** (Sec. 5.5 rule 1): all 7 network MEM_CPY are
  `srcRank=0/dstRank=1..7` (write); task offsets do not intersect with opMemInfo ranges (range inference fails),
  `direction_inferred=0`.
- **notifyId reuse across ranks**: for example, `RECORD(nid=30, dst=dev2)` (wait side is on the peer, skipped) and
  local `WAIT(nid=30)` (record side is on the peer, not in this db) appear in pairs, 7×2 each for two rounds;
  16 locally paired pairs (nid 3/13 main↔dispatch stream, 15..27 dispatch stream→peer stream, 6..12 peer stream→dispatch stream),
  14 unpaired local WAITs → charged at `remote_wait_delay` (Sec. 5.4).
- lcEid/rmEid match ranktable.json verbatim (`…0040…01010001` → port `1/0` etc.);
  all network task ports are resolvable.

### 4.8 eid→rank Index (`build_rank_index`, Peer Rank Resolution)

Peer rank resolution (Sec. 5.5 SQE fallback, Sec. 6.7 CCU primary path) shares a single eid(hex) → rank_id
index, constructed with the following priority and basis:

1. **RankEidMap table (primary path)**: The `RankEidMap` table in runnerdb `hccl_sim.db` (same directory as hccl_vm_data.db).
   In estimate mode, the Communicator table has only one row for the local rank; the peer rank
   follows this table (`sim_models.h` comment; host machine precedent `GetRmtRankIdByEid`,
   `device_sqe_parse_stub.cc:889`). The table is `sim::RankEidMap` split-column storage
   (`id INTEGER PK + rank_id INTEGER + eid BLOB(16B)`); eid is "hex string byte order,"
   same source and order as `ccuChannels.leid/reid`; the Python side matches directly by `bytes.hex()` equality;
   read as `SELECT rank_id, eid FROM RankEidMap`.
   The table is populated from ranktable.json parsing (`topo_ascend_cluster_parser.cc:229`);
   in mock-comm scenarios, it is entirely rewritten with communicator initialization, hence authoritative.
2. **ranktable.json (fallback)**: When RankEidMap is empty/`hccl_sim.db` is missing/the table does not exist, expand
   `rank_list[].rank_id ↔ level_list[].rank_addr_list[].addr` (addr_type=EID)
   to build an isomorphic index. Measured: both paths produce identical indexes (8 ranks × 9 eids = 72 entries).

Read-only opening (`file:…?mode=ro` URI); read failures (including table non-existence) degrade to fallback rather than error.
Note that **`ccuChannels.srcRankId/dstRankId` cannot be directly used as communicator rank**: the writer side
(`hccp_ccu_stub.cc:401-405`) explicitly fills global endpoint deviceId (channel establishment precedes
communicator; the same channel can be reused by multiple domains), which has an offset from the opDetails.rankId numbering system;
checker has deprecated this field for this reason (`g_allRankChannelInfo` stores only dstDeviceId/dstDieId).

---

## 5. SQE Task Graph Generation (Sub-Requirement 2 - Second Half, Focus of This Iteration)

### 5.1 Overall Pipeline

```
OpExecutionGroup
   │
   ▼
⓪ SQE batch merge (sqe_group_merge.py, Sec. 5.1.1): merge UDMA data tasks within same (stream, sqeGroupId) batch into 1
   │
   ▼
⓪' DPU sequence merge (sqe_group_merge.py, Sec. 5.1.2): DPU large message N-1 front blocks + tail block WRITE_WITH_NOTIFY merged into 1
   │
   ▼
① Per-SQE → Task node (Sec. 5.2 mapping table; CCU_GRAPH forwarded to Chapter 6 for expansion)
   │     stream_id: int (SQE stream uses original streamId; CCU skeleton tasks also belong to their SQE original stream, Sec. 6.5)
   ▼
② Stream serial chain: group by (deviceId, streamId), global_id ascending → dep_graph.add_stream_chain
   ▼
③ Notify pairing: cross-rank tasks do not participate in pairing (nodes retained); only local notify first-come-first-served → dep_graph.add_match (Sec. 5.4)
   ▼
④ Network task parameter completion: direction inference / eid resolution (Sec. 5.5)
   ▼
⑤ CCU sub-graph expansion (Chapter 6): CCU_GRAPH placeholder → microcode instruction interpretation → internal nodes + Event pairing
   ▼
⑥ Output assembly: (SimConfig, DependencyGraph) → HcclSimulator(config, dep, topo)
```

### 5.1.1 SQE Batch Merge (Pre-Processing Before Graph Building)

On the collection side, one UDMA doorbell (SQE) can carry multiple WQEs; tasks are still stored per WQE and marked with
`sqeGroupId` for batch membership (cyclically incrementing doorbell batch number 1..255, wraps at 255, never takes 0; 0 = construction
default "unmarked", Sec. 4.4.1). Before graph building, batches are split by **(streamId, sqeGroupId)** — the split key explicitly
includes stream; all data tasks within a batch (MEM_CPY/REDUCE/WRITE_WITH_NOTIFY) are merged into a single entry, aligning with
hcomm batch transfer profiling's `ExecProfilingAll` granularity — all three entry points (`pipeline.py`/
`__main__.py`/`dump_graph.py`) call uniformly, ensuring simulation and dump granularity are consistent:

- **Isolation guarantee by stream grouping**: The split key includes streamId; **cross-stream tasks never merge into the same batch** (one doorbell
  belongs to only one stream; a single doorbell batch is not split by the key); group numbers (doorbell batch numbers) are unique within the 255 window
  — doorbells from other operators are not visible in this operator's task list but equally consume batch numbers, so **same stream same number must be
  the same doorbell**, preventing cross-doorbell mis-merging (safe for multi-operator db).
- **Descriptive information** (type/address/endpoint/notifyId/reduction parameters/profInfo) takes the **last** data
  task within the batch; **length takes the full-batch cumulative sum** — consistent field by field with `ExecProfilingAll(loc[insNum-1], rmt[insNum-1],
  totalSize, transferOp[insNum-1])`; trace/dump display names similarly take the last task's
  op name (Sec. 11.4.1). When a batch mixes write/read directions or different dataTypes, it similarly follows the last task
  (a single SQE batch is typically homogeneous; mixing is a rare scenario).
- **Non-participating in merge**: notify/SDMA/sync tasks (notifyId is unique; pairing depends on order; local tasks
  are retained as-is without affecting batch boundaries); unmarked/no-group-number tasks (`sqeGroupId <= 0`: 0 = unmarked,
  -1 = v3 and earlier BLOB without this field) are retained as-is and **break the current batch** (batch membership cannot be determined; conservatively
  end the current batch); UDMA non-data tasks (inline write notify) participate only in batch splitting; those interleaved within a batch are placed after the merged task
  (notify causality).
- The merged task occupies the slot of the first data task in the batch (batch head); the split key containing stream ensures tasks within a batch share the same
  stream and same jetty; the stream serial chain (Sec. 5.3) and notify pairing (Sec. 5.4) relative order are unaffected;
  local (non-UDMA) tasks do not participate in merging and do not reset batch boundaries (retained in place, not affecting batch continuity).

### 5.1.2 DPU Sequence Merge (Pre-Processing Before Graph Building)

DPU (host CPU RoCE) path large message WriteWithNotify sequence merging, executed after 5.1.1
(`pipeline.py` unified invocation). Rules and semantics in Sec. 11.4.6: scan in DPU stream order for
MEM_CPY chains with **consecutive src/dst addresses** (only data tasks with `src_type==1` DPU source participate;
other tasks are retained as-is without breaking the chain), terminated and merged into a single entry by an address-consecutive WRITE_WITH_NOTIFY —
**the terminating task is a natural terminator**; two independent WriteWithNotify calls cannot merge across (each call's
tail block necessarily closes the current chain); the only mis-merge scenario remaining is "an independent pure write that happens to adjoin the WriteWithNotify start address" — one
theoretical scenario (semantically equivalent to a continuous data segment + terminating notification; simulation results are identical). Chains without termination are not merged
(independent pure writes remain per-entry `RDMASend`); the merged task occupies the terminating task slot (notify causality); merged
front blocks are removed from their original positions; unmerged tasks maintain the original stream order.

**Single-rank graph building strategy (design premise, primary difference from checker)**: The db contains only single-rank task data;
the DependencyGraph describes only that rank's tasks and their dependencies. **Cross-rank pairing edges (post-wait relationships)
are entirely not built (task nodes retained, approximated with fixed remote latency)**:

- SQE level: RECORD with waitDev ≠ this device (cross-rank post), WAIT with recordDev ≠ this device (cross-rank wait)
  **retain task nodes** — graph-built normally in stream order (src≠dst expresses remote, uses fixed remote latency), but **no
  pairing edges are established** — only post-wait pairing within the single rank is considered (Sec. 5.4).
- CCU level: instructions posting to remote Events, waits depending on remote Event assertion **similarly generate task nodes**
  (Notify_Record/Notify_Wait, src=local, dst=INVALID → fixed remote latency), but no cross-rank
  pairing edges are established (Sec. 6.4/6.6).
- **Cross-rank data transfers are retained** (MEMCPY with direction=write/read): they are the main source of network
  bandwidth usage from this rank's perspective; only synchronization dependency edges are ignored, not the transfers themselves.
- makespan semantics: single-rank perspective operator end-to-end duration = local computation/copy + locally initiated network transfers
  + local synchronization waits; when the remote side becomes ready is not modeled (equivalent to assuming the remote side is immediately ready).

**Simulation granularity**: Default per-operator (one DependencyGraph per `(commId, opIter)`, one simulation, one record),
consistent with checker's per-operator verification granularity; `--all-in-one` optional full-volume mode (multiple operators on the same stream are naturally serial).

### 5.2 taskType → hccl_sim Task Mapping

Aligned with HCCLSIM new task model (`TaskType`: `NOTIFY_RECORD`/`NOTIFY_WAIT`/`REDUCE_INLINE`/
`MEMCPY`/`UB_INLINE_WRITE`/`UB_WRITE_OR_READ`/`WRITE_WITH_NOTIFY`, see Sec. 7.3):

| db taskType | Condition | hccl_sim TaskType | Task Parameters | Latency Model |
|-------------|-----------|-------------------|-----------------|---------------|
| NOTIFY_RECORD (1) | waitRank == local rank | `NOTIFY_RECORD` | notify_id=int, src=dst=local_rank | Fixed + trigger Event |
| NOTIFY_RECORD (1) | waitRank != local rank | `NOTIFY_RECORD` (cross-rank post: node retained, not paired) | notify_id=int, src=local_rank, dst=peer rank (INVALID if unresolved) | Fixed remote_record_delay, no Event trigger |
| NOTIFY_WAIT (0) | recordRank == local rank | `NOTIFY_WAIT` | Same as above | Wait for Event |
| NOTIFY_WAIT (0) | recordRank != local rank | `NOTIFY_WAIT` (cross-rank wait: node retained, not paired) | notify_id=int, src=peer rank (INVALID if unresolved), dst=local_rank | Fixed remote_wait_delay |
| MEM_CPY (3) | jettyId == INVALID | `MEMCPY` | size=len, src/dst_rank=taskData endpoint rank (fall back to local if unfilled) | On-chip memory dynamic shared bandwidth |
| MEM_CPY (3) | jettyId valid, write direction | `UB_WRITE_OR_READ` | size=len, src_rank=local, dst_rank=peer rank, src/dst_eid, jetty_id, protocol, is_write=True | Port write direction bandwidth sharing |
| MEM_CPY (3) | jettyId valid, read direction | `UB_WRITE_OR_READ` | size=len, src_rank=peer rank, dst_rank=local, src/dst_eid, jetty_id, protocol, is_write=False | Port read direction bandwidth sharing |
| REDUCE (2) | jettyId == INVALID | `REDUCE_INLINE` | size=dataCount, data_type=task-level dataType enum name, src/dst_rank same as MEMCPY local branch | On-chip memory dynamic shared bandwidth |
| REDUCE (2) | jettyId valid | Same as MEM_CPY network branch (Sec. 5.7 approximation) | Same as above (including data_type) | Same as above |
| WRITE_WITH_NOTIFY (7) | Always write direction (local src → remote dst, no direction inference) | `WRITE_WITH_NOTIFY` (**single task, not split**) | size=len, notify_id=notifyId, src_rank=local, dst_rank=peer rank (fallback via eid→rank index if unresolved), src/dst_eid, jetty_id, protocol, is_write=True; when isReduce=1, data_type=dataType enum name | Port write direction bandwidth sharing + fixed record latency for post-write notify (remote) |
| CCU_GRAPH (4) | — | (Expansion, Chapter 6) | — | — |
| AIV_GRAPH (5) | — | Skipped | — | Warning |
| SYNC_STREAM (6) | — | **Skipped** (host-side stream sync point, Sec. 5.6) | — | — |

> waitRank/recordRank are the taskData endpoint rankIds (carried in deviceId fields, src=record side,
> dst=wait side, Sec. 4.4.1/4.6); used only for **determining local/cross-rank**: cross-rank notify tasks
> retain nodes during the mapping phase but do not enter pairing (Sec. 5.1 strategy);
> pairing occurs only among local notifies. Notify local/remote in the new model is expressed by `src_rank == dst_rank`
> (`is_local_notify`).
> **WRITE_WITH_NOTIFY (7) single-task mapping** (corresponding one-to-one with db tasks, not split into write + NOTIFY_RECORD
> two sub-tasks): write direction is always write (sending end convention: src local → dst peer; no Sec. 5.5 direction inference);
> uses port write direction bandwidth sharing; after write data lands, notify is sent to the peer, charged at fixed record latency
> (peer wait is not in this db; single-rank simulation premise; same as cross-rank record semantics; self-write src==dst edge
> case uses local record latency); `notifyId` is retained with the task (for trace/dump display);
> when `isReduce=1`, only `data_type` display is affected (same enum name mapping as the REDUCE branch).
> **Network tasks consume taskData rank directly** (deviceId field carries rankId; sending end convention:
> write=src local/dst peer, read=src peer/dst local): write fills `dst_rank=peer rank`,
> read fills `src_rank=peer rank`; when the peer rank is unresolved (`0xFFFFFFFF` or equals local), fall back to INVALID
> (Sec. 10.1). Direction is explicitly carried as `is_write`; the simulator no longer depends on the `src_rank==0` heuristic
> (Sec. 10.3 #1 fixed; task.txt old input path still uses heuristic).
> Local eid prefers the SQE explicit lcEid (v2 collection new addition); defaults to jettyId → ccuChannels channel
> leid derivation (Sec. 5.5, Sec. 10.1).
> The TaskType in this table is the internal latency model classification; the task display name for trace/dump has already switched to
> the profInfo.profTaskType op name (Sec. 11.4.1 mapping table, Sec. 10.3 #18 implemented);
> the correspondence with this table's TaskType follows the "Simulation TaskType" column in Sec. 11.4.1.

### 5.3 Stream Serial Chain Construction

- Group by `(deviceId, streamId)`; within each group, chain by `global_id` ascending (= SQE dispatch order).
- The simulator launches one simpy process per chain for serial execution (Chapter 7) — **no need for checker's mainStart virtual
  start point and main stream special casing**: simulation focuses on end-to-end makespan; each chain launches from time 0 in parallel;
  the first WAIT in a slave stream naturally forms the actual synchronization with the main stream.

### 5.4 Notify Pairing Algorithm (Local Only)

**Cross-rank post-wait edges are entirely ignored** (Sec. 5.1 strategy): before pairing, classify by Sec. 4.6 peer backfill results;
cross-rank notify tasks (RECORD with waitDev ≠ this device, WAIT with recordDev ≠ this device) retain nodes during the Sec. 5.2
mapping phase (graph-built in stream order) but do not enter pairing. Therefore, the pairing key simplifies from checker's triple to a single notifyId:

```
checker semantics:   pairing key = (recordDeviceId, waitDeviceId, notifyId)   ← multi-rank
this plugin:         filter cross-rank first → pairing key = notifyId                    ← single-rank
```

Consistent in principle with checker's `AddLocalNotifyEdges` (dispatch order simulation + first-come-first-served consumption,
`task_graph_generator_v3.cc:847-946`), light_performance simplifies to
**two-pass scan + first-come-first-served**:

```
Input: All local notify tasks for an operator (cross-rank nodes retained but not participating in pairing)
Pass 1: Scan by global_id ascending
    RECORD → pending_records[notifyId].append(task_id)                        # queue
    WAIT   → if pending_records[notifyId] is non-empty:
                  rec = pop front; dep_graph.add_match(wait_id, rec_id)          # first-come-first-served
              else:
                  unmatched_waits.append(wait_id)                               # wait for "later" records
Pass 2: Process unmatched_waits (cases where WAIT appears before RECORD)
    Replay by global_id order: unmatched WAIT is suspended; RECORD pairs with the earliest suspended WAIT
    (successfully paired RECORD is simultaneously removed from pending_records, ensuring diagnostic statistics are accurate)
Cleanup: remaining WAITs → normal (see below); remaining RECORDs → warning, should not appear in normal data
```

Notes:

- The db id order is the dispatch order, and the stream serial chain already guarantees "successor tasks on the same stream after a WAIT do not execute first,"
  so checker's full queue simulation and deadlock detection are unnecessary (deadlocks manifest in simulation as makespan anomalies/unexecuted tasks).
- Multiple RECORDs/WAITs with the same notifyId (for example, n11/n15 appearing repeatedly on stream 6 in early 169B samples, two main stream iterations)
  are naturally paired by the first-come-first-served rule; **notifyId reuse is allowed**.
- **Unpaired local WAITs are normal semantics**: in the v2 AICPU sample (Sec. 4.7), notifyIds are reused across ranks —
  local `RECORD(nid=30, dst=dev2)` (wait side is on the peer, node retained but not paired) and local `WAIT(nid=30)`
  (record side is on the peer, not in this db) appear in pairs; this WAIT is charged at `remote_wait_delay` fixed latency during simulation
  (Sec. 7.6); the warning is informational only.
- Cross-rank pairs in early 169B samples: `RECORD(n18, s1→d2)` (notifies dev2) and `WAIT(n18)` (backfilled recordDev=2)
  both retain nodes; no cross-rank pairing edges are built: graph-built in stream order; `src≠dst` uses fixed remote latency;
  data transfer tasks on stream 8 (MEMCPY(jet=3) etc.) are unaffected.

### 5.5 Network Task Direction and eid Resolution

**Direction determination** — network data tasks are uniformly mapped to `UB_WRITE_OR_READ`; the direction determines the port charging
direction (is_write) and rank endpoint values. The taskData endpoint rankIds are carried by deviceId fields
(Sec. 4.4.1 rank field rollback); sending end convention (`device_sqe_parse_stub.cc`):
write = src local/dst peer, read = src peer/dst local:

| Priority | Rule | Basis |
|----------|------|-------|
| 1 | srcRank == local and dstRank ≠ local → write; dstRank == local and srcRank ≠ local → read | Collection endpoint rankId (carried in deviceId field, Sec. 4.4.1); all measured samples (Sec. 4.7) are determined by this rule; no inference needed |
| 2 | When rank does not distinguish (equal or both unfilled), use memory ranges: dst falls in local output range → read; src falls in local input range → write | Derived from measured samples (Sec. 4.7): fails when task offsets do not intersect with opMemInfo ranges |
| 3 | Fallback: default write; diagnostic statistic `direction_inferred` count | — |

> Network tasks explicitly carry `is_write` on the Task (Sec. 10.3 #1 fixed; CCU path Sec. 6.7 synchronously fixed);
> the simulator selects write/read charging direction and port query eid (`src_eid`/`dst_eid`) accordingly, no longer depending on
> the `src_rank==0` heuristic; the task.txt old input path lacks this field and falls back to heuristic determination.
> `WRITE_WITH_NOTIFY` (7) **does not participate in direction determination**: semantics are always write (local src → remote dst);
> during graph building, `is_write=True` is set directly.

**eid resolution** — network tasks look up ranktable by eid to get physical ports (`TopoResolver.get_ports_by_eid`):

```
SQE:        Local eid prefers SQE explicit lcEid (v2 new); defaults via jettyId
            ──ccuChannels (this device outbound channels)──> leid (local eid)
            Peer prefers channel reid; defaults back to SQE rmEid; when channel is missing, local eid degrades to any local channel leid
CCU microcode: channelId → channels[(dev,die,channelId)] → leid/reid + jettyId list
            When channel is missing, local eid prefers the mission SQE's lcEid; degrades to any same-die channel leid
write direction: simulator queries task.src_eid → ports; read direction queries task.dst_eid → ports
```

- `protocol`: channel protocol column → HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO (CommProtocol name,
  display only); `link_type`/`transport_type`: task profInfo (DfxLinkType/
  ProfTransportType name, Sec. 11.4.2/11.4.3, `_add_task` unified filling, for reconciliation with msprof L1).
- When the local eid cannot be resolved: warning + `eid_unresolved` count; when network tasks exist but no ranktable is available,
  simulation is skipped entirely (TopoResolver throws directly for missing eid, see Sec. 10.3).

**Peer rank resolution** — peer rank (`_peer_rank`) for SQE network tasks: taskData endpoint rankId
(carried in deviceId field, already resolved by the sending end through `GetRmtRankIdByEid`) is preferred; when unresolved
(UINT32_MAX/local value), fall back through eid→rank index (Sec. 4.8) by peer eid; if still unavailable,
INVALID + `rank_unresolved` count.

### 5.6 SYNC_STREAM Handling

Host-side stream synchronization point (appears once per iteration on the sample main stream), indicating the host machine waits for this stream to complete;
**does not represent a device-side task**: no task node is generated, no time is consumed (makespan is the end time of the last task in the stream chain).

- Diagnostic statistic `skipped_sync` count;
- Previously approximated with "unpaired local `NOTIFY_RECORD` fixed latency placeholder," which was removed because it introduced spurious
  `local_record_delay` at stream end (and appeared in trace as a pseudo notify task with notify_id=invalid).

### 5.7 REDUCE Approximation

Reduction = network transfer + computation. The current model approximates with data transfer:

- Network (jettyId valid): charged at port bandwidth latency as `UB_WRITE_OR_READ`
  (direction determined as in Sec. 5.5).
- Non-network (jettyId INVALID): mapped as `REDUCE_INLINE`, charged at on-chip memory bandwidth with dynamic sharing
  (`size / hbm_bandwidth`); computation latency is not separately listed at this time (can be calibrated by `dataType×dataCount` later,
  see Sec. 10.1 data_type field gap).

### 5.8 Comparison with Checker Graph Building and Simplifications

| Checker Step | light_performance | Rationale |
|--------------|-------------------|-----------|
| mainStart virtual node + main stream incoming edge special casing | Removed | Simulation only needs makespan; the first WAIT in a chain already expresses synchronization |
| AddLocalNotifyEdges queue simulation + deadlock detection (GRAPH_DEADLOCK) | Two-pass scan first-come-first-served; no deadlock detection | No verification requirement; anomalies manifest as simulation results |
| AddInterRankNotifyEdges cross-rank pairing (full-graph BFS simulation) | **Completely removed**: all cross-rank post-wait pairing edges ignored (notify task nodes retained, graph-built in stream order) | Single-rank graph building (core difference) |
| MemSlice address resolution + memory layout table | Range attribution for direction determination only (optional) | Simulation only needs len, not a precise memory model |
| CheckSlaveTaskQueue slave stream structure verification | Removed | Not a verification tool |
| Memory conflict/semantic verification/sync conflict/reachability/single-task checks | All removed | Not a verification tool |
| CompactSyncNodes (SYNC_STREAM merging) | Skipped, no node generated (host-side sync point, not a device-side task) | makespan is the end time of the last task in the stream chain (Sec. 5.6) |
| AIV sub-graph expansion | Not supported (skipped with warning) | Requirement explicitly excludes this |
| CCU sub-graph: queue parallelization transformation, loop merge, resource conflict verification | Queue parallelization simplified implementation (Sec. 7.4.1: assembly-phase skeleton/unit splitting + fork/completion-set merging with extra edge attachment + `#v` virtual streams); loop merge equivalently implemented (Sec. 6.5); resource conflict verification removed | Parallelization eliminates queue serialization distortion (Sec. 7.4.1); no verification needed; node count is manageable |

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

- `SimConfig` source: JSON specified by `--config` / manifest setting (Sec. 7.2 format, units
  us/GBps), overlaid with bandwidth utilization configuration.
- Topology: `TopoResolver` is constructed at the CLI layer (ranktable location in Sec. 3.3) and passed to the simulator.
- The output can immediately be sent to `HcclSimulator(config, dep_graph, topo).run()` (returns dict, see Sec. 7.1).

---

## 6. CCU Mode Task Graph Generation (Sub-Requirement 3, Focus of This Iteration)

### 6.1 CCU Data Organization in the db

```
SQEs with taskType=4 (CCU_GRAPH) in opTask_P_<pid>
    └─ taskData.ccu: {dieId, missionId, timeout, instStartId, instCnt, key, argSize, args[13]}
           │  instStartId/instCnt define instruction execution window [instStartId, instStartId+instCnt)
           │  args[13] consumed one by one by LOAD-class instructions (cursor advances after reading all 13 args)
           ▼
ccuInstrRes (deviceId, dieId) → instrSpace: 32K × 32B microcode instruction space
ccuChannels (deviceId, dieId, channelId) → {dstDeviceId, dstDieId, leid/reid, jettyId[]}
    └─ Channel register values → peer location and jetty
```

- SQEs are grouped by `(dieId, missionId, queue)` (single-device graph building; deviceId is constant; queue is the SQE stream)
  into dispatch contexts (checker's `CcuMissionKey` + latest-node-append semantics,
  `task_meta_translator_v3.h:47-62`): SQEs with the same key merge into the same context only when **contiguous with the previous one** (previous
  `instCnt==13` and `instStartId` connects head-to-tail, `IsContiguousCcuSqe`,
  `task_meta_translator_v3.cc:213-231`); otherwise **a new context is created** —
  multiple dispatches for the same mission (instruction window wraparound re-dispatch; typical pattern: "head SQE (13 Load
  SqeArgs) + body SQE" dispatched twice consecutively) each expand independently: each context fully replays its own instruction window
  (including the Load SqeArgs header, re-initializing loop counters and other registers with **this dispatch's args**),
  preventing a later dispatch from carrying over residual register state from the previous one, which would cause data transfers to be lost (Sec. 10.3 #20).
  - **args consumption** (checker `CcuGraphStateV3::GetSqe` semantics): SQEs within a context are arranged by dispatch order;
    args are consumed sequentially slot by slot — after reading all 13 args (reaching the last slot `sqeArgsId==12`), the cursor advances
    to the next SQE; in the typical layout, the head SQE's 13 Load SqeArgs consume the head SQE's args[0..12],
    and subsequent Load instructions within the body SQE window consume the body SQE's args.
  - **Register/Event bitmap shared globally across contexts** (checker `AllRankParamRecorder::Global()`
    semantics): waits in a later dispatch can be resolved by Event bits/residual post accounting left from a previous dispatch,
    consistent with checker's multi-graph shared recorder behavior.
  - **Dispatch contexts within the same queue advance strictly in dispatch order** (checker queue semantics: a single
    queue strictly follows instruction order; the next TaskCcuGraph on the same stream must wait for the previous one to complete); cross-queue polling
    advances jointly (Sec. 6.6).
- Sub-graph attachment point: The position of the CCU_GRAPH SQE in the stream chain is the sub-graph entry; after expansion, the placeholder is removed
  (no placeholder nodes are introduced in the task graph); the sub-graph attaches to SQE predecessor/successor tasks through extra edges (Sec. 7.4.1).

### 6.2 Microcode Instruction Format (32B Fixed Length)

Same source as `${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode`
(reference copy in this repository: `ai_skills/skills/convert-ccu-microcode/references/ccu_microcode_v1.h`):

```
CcuInstr = CcuInstrHeader (2B) + union { V1(A5) | V2(A6) | V3 } (30B)

CcuInstrHeader bit fields (little-endian u16):
    code     : 11 bit   (header & 0x7FF)      instruction code
    type     :  4 bit   ((header >> 11) & 0xF) instruction major class
    reserved :  1 bit

Instruction major classes:  LOAD_TYPE=0x0 | CTRL_TYPE=0x1 | TRANS_TYPE=0x2 | REDUCE_TYPE=0x3
```

Chip generation → instruction set version mapping (checker `GetCcuInstrVersion`, `ccu_task_transform_v3.cc:175-184`):
`Ascend950 → A5/V1`, `Ascend960 → A6/V2`, other devTypes fall back to A5; derived from Device's soc_version
(hccl_sim.db) or `opDetails.devType` (`expand_ccu_graph` selects `_MissionContext`/`_MissionContextA6` interpreter by `dev_type`). **Both instruction set generations are implemented** (A6 ported from
checker `ccu_task_transform_instruct_a6_v3.cc`, Sec. 11.3.3); A5/A6 instruction set difference overview in Sec. 6.8.

### 6.3 Instruction Type → Task Mapping

#### 6.3.1 Summary Table

| Instruction Major Class | Representative Instructions | light_performance Behavior | Output Task |
|------------------------|------------------------------|---------------------------|-------------|
| LOAD (register write) | LoadSqeArgsToXn/GSA, LoadImd, LoadX, arithmetic-logic (A6 Add/Sub/...) | Update register model; data plane **produces no nodes**; A6 LOAD-class instructions commonly carry `setCKE` tail field (non-zero mask → local `NOTIFY_RECORD`, checker `ProcessSetMask` semantics); A6 Load/Store (on-chip memory↔Variable data plane) does not model on-chip memory content; registers are not backfilled (only setCKE synchronization semantics retained, warning logged) | — (Notify_Record when A6 setCKE non-zero) |
| CTRL: SetCKE/SetCKBit | Set Event bits | Local → `NOTIFY_RECORD` (register Event accounting, with mask); remote target (cross-rank post, SyncCKE etc.) → remote `NOTIFY_RECORD` (not paired, Sec. 6.6) | Notify_Record |
| CTRL: Wait/Jmp/Fence | Wait for Event bits / jump / fence | Event wait (A5 embedded waitCKE field): if bits insufficient, pause this queue pointer; if bits satisfied, generate `NOTIFY_WAIT` (split by contributing bits, with mask); waits depending on remote bits → remote `NOTIFY_WAIT` (not paired, fixed latency, Sec. 6.6). A6 independent Wait instruction (Variable value comparison spin, checker `TransformWaitInstr`): satisfied → pass through with no node; unsatisfied = depends on peer remote register write (SyncWtX/TransMem notify) → top-level stall resolution generates remote `NOTIFY_WAIT` (fixed `remote_wait_delay`); within loop body, directly generate node and continue. Jmp: A5 evaluates by `Xn[conditionXnId] != expectData` (checker `TransformJmpInstrA5` semantics); A6 supports 6 condition codes (==/!=/>/>=/</<= , ≥6 unconditional) + relative/absolute two modes (jumpMode, target `Xn[relTarInstrXnId]`); target limited to current instruction window; out-of-bounds warning pass-through; within loop body, pass-through; with step budget defense against backward jump infinite loops. Fence (A6 only): checker does not support; warning pass-through (queue serial model naturally guarantees order) | Notify_Wait |
| CTRL: ClearCKE | wait + clear bits | wait semantics; A6 follows checker `ClearWaitMask` semantics to clear the wait bits consumed this time (`clearMask != 0` is a checker-unsupported scenario; warning + tolerant clearing of target bits) | Notify_Wait |
| CTRL: Loop/LoopGroup | Loop | loopCnt from register; **merge expansion** (Sec. 6.5; falls back to full expansion on failure); A6 Loop `mode==1` is conditional expansion (`wishCKEBit == Xp.loopCtxId + 1024` must match to execute; always takes full expansion path) | None (body instructions mapped per this table; data task size × total iteration count under merge path) |
| TRANS (transfer) | LocMem↔LocMS/RmtMem/RmtMS (A5), local 4 codes + unified TransMem dmaOpCode (A6) | Mapped by role (Sec. 6.3.2) | `Memcpy`/`Ub_Write_Or_Read` |
| TRANS: Sync (cross-rank sync) | SyncCKE/GSA/Xn (A5), SyncXnWtx/SyncAtx (A6) | Remote assertion/notification → remote `NOTIFY_RECORD` (not paired, Sec. 6.6); value sync is remote register write, no node built | Notify_Record (remote) |
| REDUCE | Add/Max/Min | Same as SQE REDUCE approximation (Sec. 5.7); CcuBuffer is rank-local buffer, always local on-chip memory reduction | `Reduce_Inline` |

#### 6.3.2 TRANS Transfer Role Determination (Consistent with checker `ccu_loop_merge_v3.cc:73-144`)

By src/dst device ownership:

| src | dst | Role | hccl_sim Task |
|-----|-----|------|---------------|
| Local | Local | LOCAL_COPY | `Memcpy` (on-chip memory dynamic sharing) |
| Local | Remote | WRITE | `Ub_Write_Or_Read` (src_rank=local, dst_rank=peer, is_write=True) |
| Remote | Local | READ | `Ub_Write_Or_Read` (src_rank=peer, dst_rank=local, is_write=False) |

> Cross-die/card read/write bidirectional transfers are uniformly mapped to `Ub_Write_Or_Read`; direction is expressed through `is_write`/trace
> `direction` field; ports are charged independently by write/read direction; peer rank is resolved through channel
> reid → eid→rank index (INVALID when unresolved). To preserve microcode-distinguishable buffer
> information, CCU-translated transfer tasks carry `src_buf_type`/`dst_buf_type` (mem/ms, determined by
> instruction source/destination operands); SQE/CLI path tasks with the same name lack this information and output invalid value N/A.

> Cross-rank **data transfers** (WRITE/READ roles) are retained — they are the main source of network bandwidth usage;
> only cross-rank **synchronization edges** (post-wait, see Sec. 6.4/6.6) are ignored.

Key field sources:

- `size`: length register (A5 `lengthXnId` direct read / A6 `xlId` through `GetXnId` — when bit15=1,
  register number is offset by loop `xnIdOffset`) → query register model.
- Peer location: A6 channel through **register value** (`Xn[GetXnId(xcId)]` value is channelId) →
  `channels[(dev,die,channelId)]` → `{dstDeviceId, dstDieId}`; A5 channel is instruction
  immediate field. A5 CcuBuffer-class instructions `msId / 0x8000` = rank, `msId % 0x8000` = CcuBuffer offset (CcuBuffer page 4KB) — **addressing encoding only**: CcuBuffer is a rank-directly-accessible local buffer (checker `GenSliceFromMs` +
  `AddLocalBatchReduce` treats all msIds as always local); msId high bits are **not used as** local/remote task type
  basis; cross-rank data movement is expressed by Rmt* role instructions (through channels). A6 TransMem's
  `msIdMode==1` marks the CcuBuffer end (write source / read destination is local CcuBuffer,
  `src/dst_buf_type=ms`).
- eid: channel leid/reid (hex strings); local eid prefers the mission SQE's `lcEid`
  (v2 new addition); defaults to same-die channel leid approximation.
- A6 `TransMem` `dmaOpCode` (implemented, checker `TransformTransMemInstr`):
  3=write, 5=write with notify (notify value writes to remote Variable/Event; target type unresolvable in single rank →
  approximated as remote record), 6=read, 0x1A=remote atomic count (no data plane, approximated as remote record),
  0x19 (compressed write+atomic) and others unsupported (warning, skip data plane; setCKE local
  post still executes to maintain synchronization skeleton); `udfEnable` (network reduction) charged as ordinary UB transfer
  (Sec. 5.7 approximation; peer reduction not modeled); `udfType != 0` warning, skip data plane.

### 6.4 Execution Model: Register + Event Accounting (register_model.py)

Python simplification of checker's `AllRankParamRecorder` (`ccu_all_rank_param_recorder_v3.h:27-110`):

```python
class CcuRegisterModel:
    xn:    dict[(dev, die, id)] -> int      # General registers (length/loop count etc.)
    cke:   dict[(dev, die, ckeId)] -> int   # CKE bitmap (set/clear)
    gsa:   dict[(dev, die, id)] -> int      # A5 GSA registers (A6 has no GSA, only xn/cke)
    seen_post: dict[key, list[PostRecord]]  # CKE post accounting (for pairing)

    key = (waitDeviceId, dieId, ckeId)
    PostRecord = {record_device_id, cke_mask, remaining_mask, task_id, is_local}
```

- LOAD/arithmetic-logic instructions "execute" on this model (read/write registers); uninitialized reads → warning and treated as 0
  (checker errors and exits; this is more lenient to accommodate incomplete data). A6 arithmetic-logic instructions
  (Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt) execute with 64-bit wraparound semantics (Mul takes
  low 32-bit product, shift amount truncated by 0x3F, arithmetic shift uses signed interpretation); LoadX/StoreX
  indirect addressing target register number out-of-bounds (>16383) warning, skip write (checker errors and aborts).
- **A6 in-loop register number offset** (checker `GetXnId`/`UpdateXnId`): when the register number
  field bit15=1 in the payload, within the expansion region (`curLoopIdx >= expandOffset`), offset the register number by
  `xnIdOffset × curExpandCnt`; bit15=0 uses the original number directly. Event/CcuBuffer id
  loop offset follows the same rule across both generations (`UpdateId`).
- **Event pairing rules** (consistent with checker `GenWaitNode`, `ccu_task_transform_v3.cc:56-121`):
  ```
  post matches wait ⇔ post.waitDeviceId == wait.waitDeviceId
                   && post.dieId == wait.dieId
                   && post.ckeId == wait.ckeId
                   && (post.ckeMask & wait.ckeMask) != 0        # bits have intersection
  After matching: one wait instruction aggregates into at most 1 local wait node (mask = OR of all contributing
          post bits) + 1 remote wait node (bits depending on remote assertion)
          post.remaining_mask &= ~wait.ckeMask                  # one post's bits can be consumed by multiple waits in parts
          Emptied → removed from seen_post
  Node dependency: aggregated wait node builds edges with all contributing posts — first contributing post uses
          notify pairing (add_match, Event semantics); others use extra edges
          (child=wait, parent=post, "wait for all parents to complete" pure timing constraint)
  ```
- **Event accounting registers only local rank posts** (single-rank graph building): cross-rank posts generate remote
  `Notify_Record` nodes (src=local, dst=INVALID → fixed remote latency) but do not register
  `seen_post` (the assertion action occurs on the peer rank's Event); wait bits depending on remote assertion generate
  remote `Notify_Wait` nodes (fixed latency, not paired), see Sec. 6.6.
- When bits are not satisfied, the queue pointer pauses (waiting for posts from other queues of this rank to unlock) — multi-queue joint advancement
  in Sec. 6.6 algorithm; when the same queue has multiple dispatch contexts (same mission dispatched multiple times, Sec. 6.1),
  the later context must wait for the previous context to complete before starting to advance.
- Final validation: `seen_post` residuals → warning and counted in diagnostic statistics (checker reports GRAPH_UNMATCHED; this is downgraded).

### 6.5 Queue Model and Loop Handling

- **queue**: `CcuSqeParam` grouped by SQE stream (checker uses ccuParams[queueId]);
  each queue is one serial chain. The new task model `stream_id` is **int**:
  SQE streams use the original streamId; **CCU queue chains also belong to their SQE original stream** — whichever stream the CCU-type SQE
  is on, the expanded skeleton tasks are on that same stream (`queue` is the original stream number;
  `alloc_stream("ccu_d{die}_q{queue}")` resolves back to the original stream number through `ccu_queue_origin_stream`;
  same-name reuse merges multiple missions of the same queue); only virtual streams newly created by queue parallelization are allocated sequentially from base
  `1<<20`; logical names are retained in `stream_labels` for dump/report display.
  Exception: when the same original stream is occupied by a queue of **another die**, the latter falls back to base-allocated synthetic stream with a warning
  (chaining would serialize cross-die parallel execution); in this case, the stream's kind=ccu in `stream_meta`.
  Instructions within the same queue are strictly ordered (checker semantics: `ccu_task_transform_v3.cc:189`,
  "a single queue strictly advances in instruction order; cross-queue coordination is expressed only through generated wait/post edges").
- **Assembly-phase queue parallelization splitting** (Sec. 7.4.1): after interpretation completes, each CCU queue chain is further split into
  skeleton (wait / free set / remote sync nodes — free set refers to record nodes not immediately following a data task)
  and asynchronous data transfer unit two categories:
  the skeleton stays on the queue serial chain (= SQE original stream chain, reassembled by dispatch order to the placeholder's original position) to express the timing framework;
  asynchronous units are reattached to virtual streams
  `ccu_d{die}_q{queue}#v{n}` (sequentially allocated from base `1<<20`), reused across missions,
  not reused within a mission, connected to the skeleton through fork extra edges — the serial chain constrains only the skeleton and unit internals,
  eliminating the distortion of "entire queue task-by-task serialization."
- **Loop expansion** (merge preferred, aligned with checker `ccu_loop_merge_v3.cc`):
  - `Loop`/`LoopGroup` instruction iteration count comes from registers and is known at interpretation time:
    - A5: LoopGroup parameters in Variable (LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41])
      and Xm (gsaOffset/CcuBufferOffset/ckeOffset); Loop iteration count in Variable[xmId] low 13 bits;
    - A6: LoopGroup parameters changed to be carried by **Variable/Xm/Xp three registers** (checker
      `LoopGroupXnA6/XmA6/XpA6`) — Variable: `loopInsCnt:10 | expandOffset:9 |
      expandCnt:9`, Xm: `ckOffset:10 | msOffset:11 | gsaOffset:32`,
      Xp: `xnIdOffset:32` (GetXnId offset, Sec. 6.4); Loop iteration count in
      `Variable[GetXnId(xmId)]` low 13 bits, Address step in `Variable[xnId]` low 32 bits, context id
      in `Variable[xpId]` low 9 bits; `mode==1` is **conditional Loop** — expand only when instruction
      `wishCKEBit == loopCtxId + 1024` (`A6_LOOP_BOARD_ENTRY`) matches; otherwise
      this expansion is skipped (checker `ProcessLoopIns`); conditional Loop always takes the full expansion path
      (checker `CollectLoopExpandA6` refuses merging).
    Each Loop within a LoopGroup first attempts the **merge path** (checker `TryProcessMergedLoopA5/A6`
    → `LoopIterationMerge` / `LoopExpandMerge` semantics); falls back to full iteration expansion if validation fails.
  - **Merge prerequisite validation** (any failure triggers WARN fallback, same semantics as checker "Skip merged-loop
    optimization"):
    1. Loop body contains only mergeable instructions:
       - A5: simple TRANS / SYNC TRANS (SyncCKE/GSA/Xn) / REDUCE / ClearCKE
         (`CollectLoopCkeOpsA5` rules; SetCKE, LOAD, LOOP, JMP,
         TransMem with `setRmtCKEMask != 0`, ClearCKE with `clearMask != 0` are all rejected);
       - A6 (`CollectLoopCkeOpsA6` + `Collect*LoopInstrA6` rules): local TRANS
         4 codes / TransMem (`udfType==0` and `dmaOpCode ∈ {3,6}` — 5 write-with-notify
         triggers remote post per iteration, rejected; channel must be resolvable, `IsExistRemoteDieInfo`) /
         REDUCE (TransMem with `udfEnable` network reduction similarly folded into trans slot) /
         ClearCKE (`clearMask==0`); SetCKBit, LOAD, LOOP, JMP, Wait are all rejected;
    2. wait/set masks are **balanced** within the loop body (`ConsumeProducedMask`/`ProduceMask`:
       each wait's bits must be produced by an earlier set; all produced bits are exhausted at sequence end);
    3. No Event/CcuBuffer resource conflicts across expansions (`CheckResourceConflict`: Event/CcuBuffer numbers after
       ckOffset/msOffset offset must not be reused across expansions; A6 CcuBuffer conflict checking covers
       local transfer/TransMem `msIdMode` CcuBuffer end and reduction msId lists);
    4. Transfer/reduction length > 0 (`ValidateLoopLen`) and expansion count × iteration count > 0;
    5. **A6 additional**: wait Event IDs within the loop body must not be reused (`CheckA6LoopWaitCkeIdUnique`
       hardware constraint; checker errors and aborts; this downgrades to fallback full expansion); each expansion's iteration count/
       length/channel value must be consistent (checker each expansion `loopCnt` equal + `InstrMerge`/
       `MemMerge` mergeability performance-side approximation — register numbers after xnIdOffset offset may
       point to different registers).
  - **Merge emission** (`EmitMergedLoopInstrsV3` semantics): the loop body is executed **once** by sample expansion
    (curExpandCnt=0), slot by slot: wait → data task → set → clear wait bits;
    data task size = single-iteration length × (expansion count × iteration count) (timing-equivalent folding, consistent with checker's
    AddBatchTransMem/AddBatchReduce batch tasks); other expansions' Event register
    bitmaps are replayed with or/clear idempotency, ensuring post-loop register state matches full expansion (only
    PostRecord task accounting is not duplicated). All tasks within the emission window (data tasks and folded
    notify tasks) carry `loop_cnt` (loop count after expansion = N+1 after N expansions) and
    `loop_iter_cnt` (single loop self-iteration count); total iterations = `loop_cnt` ×
    `loop_iter_cnt`, preserving pre-merge loop details for subsequent simulation; straight-line instructions, full-expansion fallback, and
    SQE/CLI path tasks have both fields as 0 (invalid; trace uniformly outputs N/A).
  - **Quantified effect** (REDUCE_SCATTER measured db, 128 expansions × 50 iterations × 13 instructions):
    task count 192,005 → 66 (10 data tasks, several local post/wait, 35 remote sync nodes — 21 remote posts + 14 remote waits, Sec. 6.6); graph building + simulation end-to-end
    ~55 s → ~6 ms; makespan 70,113.650 us → 2,753.711 us (this section's merge eliminates
    128K task-level notify overhead and 64K small transfers' per-task latency; queue parallelization
    and remote sync node inclusion final value, Sec. 7.4.1).
  - A6 LoopGroup configuration validation (`loopInsCnt≤512`, `expandCnt≤511`,
    `expandOffset≤511` and `expandOffset≤loopInsCnt`, checker
    `TransformLoopGroupInstr`): when unsatisfied, warning and skip the entire LoopGroup (defending against abnormal
    register values causing uncontrolled expansion); "waitCKE within loop must not duplicate Event ID" validation see above
    merge prerequisite validation #5 (`ccu_task_transform_instruct_a6_v3.cc:1126-1146`).

### 6.5.1 Merged Loop Pipeline Concurrency Recovery (Loop Tag + Signature Chain Resolution)

**Problem — folded pseudo-serialization (merge is not timing-equivalent for notify structure)**: Sec. 6.5 merge is
timing-equivalent folding for **data tasks** (size × total iteration count; total duration unchanged in bandwidth-limited scenarios), but not for Event
wait/set handshakes within the loop body. During real microcode execution, the N expansions of a LoopGroup are emitted concurrently; each expansion
internally advances through M inner iterations in a pipelined manner (typically 4KB per iteration); the loop body's
`TransMemToMs → setCKE → waitCKE → TransLocMsToRmtMem / TransLocMsToLocMem`
is **per-iteration fine-grained handshaking**: iteration k+1's on-chip memory→CcuBuffer prefetch (memcpy engine) overlaps with iteration k's
network write + local copy-out (channel / memcpy engine); steady-state throughput ≈ max(phase totals).
After merge folding, per-iteration handshaking degrades to a **single pair** of wait/record nodes; the semantics become "all front-phase data
(× N × M iterations) complete → back-phase can start" **full barrier** — pipeline concurrency is folded into serialization.

Taking the first loop template from `hccl_vm_install/data/hccl_vm_data.db` as an example (microcode 0–10:
`TransMemToLocMs(4KB, set 0:1) → waitCKE(0:1) → 7× TransLocMsToRmtMem(each 4KB,
set 0:1..0x40) → TransLocMsToLocMem(4KB, set 0x80) → waitCKE(0:ff)`; LoopGroup
128 expansions × 256 inner iterations = 32768 times × 4KB = 128MB/task):

```
Current graph building (after Sec. 7.4.1 parallelization; AA = loopgroup predecessor task representative,
                             i.e., the last node before the loop emission window in the queue skeleton):
  AA -> memcpy(mem->ms) -> record(0:1)                     ← prefetch unit (128MB, 95.95us)
  record(0:1) →[paired]→ wait(0:1) → 7× ub_write + memcpy(ms->mem)
                                    ↑ Full serialization: 8 back-phase units fork from wait(0:1),
                                      waiting via pairing edge for prefetch 128MB to fully complete
  Multiple { record(0:xx) / record(0x80) -> wait(0:ff) }       ← loop-end barrier (retained)
```

- Real execution: steady-state ≈ max(prefetch 95.95, 7-port write 2526.48) us;
- Current graph building: serial 95.95 + 2526.48 us.
- **Quantified** (sample db ALLGATHER, measured before and after this section's fix): makespan **2657.52 →
  2561.54 us**. Eliminates 95.95 us serial prefetch phase; after prefetch and local copy-out run concurrently through on-chip memory dynamic sharing,
  each 191.79 us (correct expression of pipeline steady state, Sec. 7.5); 7-port network write 2526.48 us becomes the sole
  critical path. Error upper bound: when two phases belong to different engines (on-chip memory prefetch vs network write) with comparable durations,
  loop body duration is overestimated by nearly **2×** (serial t1+t2 vs steady-state max(t1,t2)); same-engine phases
  (such as pure on-chip memory prefetch+copy-out) serialization error is absorbed by on-chip memory dynamic sharing (total unchanged);
  error accumulates linearly with the number of merged loops.

**Correctness premise — merge validation invariant**: Sec. 6.5 merge prerequisite validation #2 (wait/set masks balanced within
the body) guarantees that within the merged loop body, **all wait bits are produced by earlier sets within the body and exhausted at the end** — i.e.,
all Event dependencies within the merge emission window are intra-loop per-iteration pipeline handshakes; no cross-loop
real data dependencies exist (otherwise validation fails and falls back to full expansion). Therefore, resolving "mid-body waits and their consumed
sets" does not break loop-external synchronization: the **loop-end wait** (the last wait within the emission window, after which
no same-loop data tasks exist) and the **mission completion set** (skeleton tail + all unit tails, Sec. 7.4.1) continue to
carry post-loop instruction / SQE successor / same-queue next mission queue-level serial semantics.

#### Approach (Loop Tag + Signature Chain Identification + Deletion and Reconnection, Iterate to Convergence)

1. **Loop tag**: `_MissionContext` maintains the current emission window's `_loop_tag`, named
   `loopgroup_{i}_loop_{j}` (i = LoopGroup instruction execution sequence number within the dispatch context, starting from 0;
   j = cur_loop_idx within the group, starting from 0; nested LoopGroup within loop body each assigned by emission window,
   not forming hierarchy). `_emit_merged_loop` sets the tag within the window; `add_task` defaults to inheriting the new field
   `Task.loop_tag` (same pattern as `loop_cnt`). **Only merge path is tagged**; full-expansion fallback /
   SQE / CLI / straight-line instructions are empty string (trace outputs N/A). The tag also serves resolution scope limitation
   (preventing cross-loop / cross-mission mis-deletion) and trace/dump traceability.
2. **Signature chain identification** (recorded during emission + determined before assembly, avoiding graph result reverse lookup): when `_gen_waits`
   successfully pairs, record (wait_task_id → all contributing post task_ids) into
   `ctx.wait_posts` (aggregated wait's match + extra all contributors, Sec. 6.4). During the assembly phase
   (`expand_ccu_graph` assembly loop beginning, **before** skeleton/unit splitting), scan the ctx.nodes sequence;
   matching a signature chain requires all of:
   - W is a locally paired `Notify_Wait` with `loop_tag == T` (no remote wait within the merge window — validation #2 guarantees all body wait bits are locally produced);
   - All contributing posts R* of W are `Notify_Record` with `loop_tag == T`, and each
     **immediately follows a same-tag data task M\*** in ctx.nodes (unit-tail record, i.e., the folded producer
     task, such as memcpy folded from TransMemToMs; data tasks ⊆ `_QUEUE_DATA_TYPES`,
     not limited to Ub_Write_Or_Read — Memcpy/Reduce_Inline also exhibit this pseudo-serialization);
   - **After W (in ctx.nodes order), same-tag data tasks exist** (mid-body determination; the loop-end
     wait has no same-loop data tasks after it, naturally exempt; the next loop's data tasks have a different tag,
     also not misidentified);
   - None of R* immediately follow non-data nodes (free set / sync slot local assertion **does not match** — these
     wait for real dependencies such as remote sync; resolution is prohibited).
3. **Deletion and reconnection** (pruning ctx.nodes, not post-assembly graph surgery):
   - **W always deleted**: removed from ctx.nodes and dep task table; its pairing edge (`remove_match`)
     and aggregated extra edges (`extra_edges[W]`) are revoked — this is the essential operation breaking the barrier;
   - **R\* depends on consumers**: R does not appear in any other wait's `wait_posts` values → delete together
     (in the sample db, set24 is consumed only by wt25, directly deleted); otherwise **retain R, only delete
     W** (its retained edges are unaffected; the loop-end wait proceeds as normal — timing-equivalent, semantics-lossless
     conservative branch; taken when post bits are partially consumed by multiple waits);
   - **Successor unit reconnection happens automatically**: after pruning, the assembly logic re-scans ctx.nodes; units that originally forked from W
     automatically fork to W's preceding skeleton node (the loopgroup predecessor task AA);
     when W is the skeleton head (no preceding skeleton), that unit becomes the entry unit; entry dependencies are expressed by SQE attachment edges
     (Sec. 7.4.1). **No edge-reconnection graph surgery needed** — fork parent / skeleton chain /
     completion set are naturally generated by existing assembly rules based on the post-pruning sequence.
   - Register / Event accounting **zero changes**: pairing consumption was completed during emission; pruning is purely a graph structure
     operation; post-loop instruction wait matching continues using the accounting results unaffected (consumed post bits
     were already consumed by W; no residual waiters remain).
4. **Iteration and statistics**: Repeat steps 2–3 until no new matches (handling multi-level chains within the body, such as
   prefetch→wait→mid-section→wait→final-section: after deleting the first layer, the next layer is exposed as a new chain head);
   `CcuBuildStats` adds `pipeline_chains_broken`; summary line presents as `pb=N`.

Fixed sample task graph after the fix (compared to the beginning of this section; deleted nodes are record(0:1)/wait(0:1); the original
"wait(0:1) → ub_write" gate is taken over by AA, timing-equivalent):

```
AA -> wait(0:ff)                                        ← loop-end barrier (skeleton)
AA -> memcpy(mem->ms)                                   ← prefetch unit (record deleted)
AA -> Multiple { ub_write_or_read -> record(0:xx) }      ← 7-way write, concurrent with prefetch
AA -> memcpy(ms->mem) -> record(0x80)                   ← copy-out unit, concurrent with prefetch
Multiple { record(0:xx) / record(0x80) -> wait(0:ff) }  ← loop-end: wait for all data to complete
```

#### Implementation Details (Code Placement)

- `simulator/task_model.py`: `Task` adds `loop_tag` slot (default empty string);
  `to_trace_args` outputs `loop_tag` (empty string → `N/A`, same granularity as `loop_cnt`);
- `ccu_graph_builder.py`:
  - `_MissionContext.__init__`: `_loop_tag = ""`, `_lg_seq = 0`, `wait_posts = {}`;
  - `_exec_loop_group`: on entry, records `lg_tag = f"loopgroup_{self._lg_seq}_loop_{
    cur_loop_idx}"` (forwarded to `_emit_merged_loop`); after LoopGroup processing,
    `self._lg_seq += 1`;
  - `_emit_merged_loop`: sets `self._loop_tag = lg_tag` within the window (`finally` resets,
    same pattern as `_loop_cnt`);
  - `_gen_waits`: after successful pairing, `self.wait_posts[wait_task.task_id] =
    [p.task_id for p in posts]`;
  - `expand_ccu_graph` assembly loop beginning calls `_break_pipeline_chains(ctx, dep)`
    (returns deleted chain count, accumulated into `stats.pipeline_chains_broken`);
- `graph_dumper.py` / `simulator/report.py`: dump `nodes[]` and trace args
  add `loop_tag`; DOT/HTML legend synchronized (optional, presented alongside loop_cnt).

Trace presentation (`export_trace`, display layer only, does not affect simulation): tasks on virtual slave streams carrying
`loop_tag` have their display row re-attached to the loop tag row of the main stream (row name is the tag, such as
`loopgroup_0_loop_0`), indicating they are converted from in-loop instructions; other tasks on the same slave stream
maintain the original display row — cross-mission reused mixed slave streams are therefore split into original row + tag row.

Acyclic guarantee: pruning only deletes nodes and edges, does not add edges; deleted W/R are all emission-window-internal nodes;
fork/merge edges are recalculated by the assembly phase based on the post-pruning sequence (Sec. 7.4.1 acyclic argument continues to hold);
`_drop_cyclic_matches` (Sec. 5.4) and simulator runtime deadlock detection remain as fallback.

#### Approach Evaluation (Pros and Cons) and Alternatives

**Pros**:

1. Does not modify Sec. 6.5 merge semantics and checker alignment logic — post-hoc resolution; risk isolated;
2. Prunes ctx.nodes before assembly (rather than graph surgery on assembly results): fork parent / skeleton chain /
   completion set are naturally generated by existing assembly rules, avoiding the correctness burden of "edge reconnection after node deletion"
   (post-assembly surgery requires manual reconnection of fork edges, skeleton chain, and completion set — error-prone);
3. Records wait→posts pairing during emission; identification does not depend on graph result dependency shape — insensitive to Sec. 7.4.1
   assembly rule evolution (fork parent selection, unit splitting adjustments);
4. Loop tag limits resolution scope (same loop, same mission), preventing cross-loop / cross-mission
   mis-deletion; the tag also feeds back into trace/dump traceability;
5. Deletes nodes rather than only deleting pairing edges: avoids retaining a semantically orphaned unpaired local wait (although its
   measured `local_wait_delay` = 30ns makes no timing difference, the trace statistical granularity is cleaner).

**Cons / Risks**:

1. Signature chain is a structural heuristic; identification conditions are coupled with emission / assembly logic and must be maintained as both evolve
   (the coupling surface has been compressed to the ctx.nodes sequence layer through "emission-time recording + pre-assembly pruning");
2. `Task` model field expansion; trace/dump/JSON presentation must be synchronized (Chapter 11 granularity);
3. "Full concurrency + bandwidth manager arbitration" is a pipeline **steady-state** approximation: ignores pipeline fill/drain
   latency (single iteration 4KB level, sub-microsecond, negligible) and per-iteration handshake **rate** overhead
   (if Event engine throughput is the bottleneck, a dedicated rate model is needed; currently both folding / deletion count as single occurrence,
   same granularity as before the fix; logged as Sec. 10.3 #21);
4. Full-expansion fallback path (`mf > 0`) does not go through this resolution — its per-iteration wait/record is real 4KB
   granularity handshaking, but inter-iteration overlap is similarly serialized by the queue chain; this path has a large node count and is not triggered by current samples;
   treated as a known residual (Sec. 10.3 #21).

**Alternative comparison**:

| Alternative | Approach | Pros | Cons | Conclusion |
|------------|----------|------|------|------------|
| B. Direct emission-time judgment (no tag / no identification) | `_emit_merged_loop` directly determines mid-body wait by slot structure (data slots follow it) and skips emitting its wait/pairing | Simplest implementation; no Task field expansion; judgment information (slot order) is at hand; no heuristic | Loses handshake nodes and loop traceability in trace; skipping emission changes register pairing accounting (loop-end wait absorbs prefetch set; timing harmless but widens checker behavior divergence); free set / sync scenarios need additional exclusion conditions | Viable simplified variant; selectable if trace presentation is not needed |
| C. Segmented pipeline modeling | Data tasks folded into K segments (8~16); inter-segment wait/record retained; explicitly models fill/drain | Includes pipeline fill effect; more precise | Node count × K; K value has no physical basis; under steady state, differs from full concurrency by only one segment at head/tail | Over-engineering; rejected |
| D. Simulator soft dependency edge | New edge type "pipeline edge" (child starts after parent completes X%) | Graph structure unchanged | Simulator semantics complicated; X has no physical basis; and serial chain structure (units fork from wait) still requires graph-layer modification | Rejected |

Conclusion: Adopt the original approach (tag + signature chain + deletion), with implementation placement per steps 2/3 above and safety
condition strengthening (emission-time recording + pre-assembly pruning + mid-body determination + consumer conservative branch).

**Implementation status (implemented)**: `Task.loop_tag` field (task_model / trace args /
dump `nodes[]` / input_builder round-trip) + `DependencyGraph.remove_task()`
(including extra edge and notify pairing cleanup) + `_break_pipeline_chains()` (called at assembly loop
beginning, statistic `pipeline_chains_broken`, summary presents `pb=N`) + trace
loop slave stream renamed presentation (`export_trace`, see above). Validation:
sample ALLGATHER db cross-engine scenario makespan 2657.521 → 2561.541 us (56 → 54
tasks, pb=1, all 9 units fork-reconnected to preceding skeleton node rwt22, matching manual prototype item by item);
archive library regression — same-engine phase scenario (mv→mv all on-chip memory) makespan unchanged (zero-error scenario,
theoretically expected); SQE-only library pb=0 unaffected. `loop_tag` retained as a persistent task attribute
for subsequent performance simulation consumption (such as per-loop bandwidth/rate modeling, Sec. 10.3 #21).

### 6.6 Graph Building Algorithm (Pseudocode)

```
expand_ccu(device_id, die_id, sqe_params, instr_space, dep_graph, reg_model):
    SQEs grouped into dispatch contexts by contiguous segments (Sec. 6.1: non-contiguous = new dispatch, independent window)
    Same-queue contexts in dispatch order serial; each queue takes first context into active set
    while active set is non-empty:
        progress = False
        for q in active set's each queue current context:        # cross-queue polling
            while ptr[q] has not reached window end:
                instr = decode(instr_space[ptr[q]])
                if instr is WAIT and reg_model.cke bits insufficient:
                    break                              # this queue pauses;轮到 other queues
                nodes = transform(instr, reg_model)   # Sec. 6.3 mapping; LOAD only changes registers
                for node in nodes: append to queue q's chain tail
                if instr is SET_CKE: reg_model.seen_post register
                if instr is WAIT:     pair with seen_post → aggregated wait node (Sec. 6.4)
                if instr is LOOP:     merge path (Sec. 6.5 validation) or repeat-interpret loop body by iteration
                ptr[q] += 1; progress = True
            if ptr[q] reached window end: current context complete; same-queue next context enters active set
        if not progress:
            # Stall = wait depending on remote CKE exists (cross-rank, invisible in single-rank graph)
            For stalled queue: generate remote Notify_Wait node (notify_id+mask, fixed latency),
                then replay the rest of this instruction (local set / data transfer), continue advancing
    SQE stream attachment (Sec. 7.4.1): placeholder removed from SQE stream chain (task graph introduces no placeholder nodes);
        entry edges (SQE predecessor → queue skeleton head) / exit edges (mission completion set → SQE successor)
        expressed through extra edges
    Assembly splitting (Sec. 7.4.1): each mission queue sequence → skeleton (stays on queue chain) + async units
        (attached to #v virtual stream, stream_id changed to point to virtual stream), fork (predecessor skeleton→unit head) /
        completion set merge (skeleton tail+all unit tails → SQE successor / same-queue next mission entry)
        extra edge connection
```

- **Cross-rank post-wait edges not built, but synchronization task nodes retained** (single-rank graph building; primary difference from checker): checker's `GenerateRawCcuGraphsV3` (`ccu_task_transform_v3.cc:1275-1406`)
  alternates multi-rank CCU graph advancement, with cross-rank posts mutually unlocking; this plugin has only single-rank instruction data:
   - Instructions posting to remote Events / remote register notifications → generate remote `Notify_Record` nodes
     (src=local, dst=INVALID → fixed `remote_record_delay`; not registered in local Event accounting,
    not participating in pairing):
    - A5: SyncCKE's `rmtCKEId/locCKEMask`, SyncGSA/SyncXn's
      `setRmtCKEId/setRmtCKEMask`, TransLocMSToRmtMS's `setRmtCKE` (consistent with checker
      `AddPost` trigger conditions);
    - A6: TransMem `dmaOpCode=5` post-write notify and `0x1A` remote atomic count,
      SyncXnWtx's `notifyValid=1` notify, SyncAtx's atomic sync — remote target type
      (Variable/Event) resolved by peer address mapping (checker `GetXnAndTypeIdByAddr`); in single rank
      unavailable; all approximated as remote record (mask takes notification value low 16 bits for trace display only).
    SyncGSA/SyncXn (A5) and SyncXnWtx primary write (A6) value sync is only remote register write
    (checker builds nodes/writes state on the peer graph); the local side does not separately generate tasks.
   - Wait bits that cannot be satisfied by local rank posts (stall detection discovers this) → the wait depends on remote → generate
     remote `Notify_Wait` node (fixed `remote_wait_delay`, not paired), and replay the rest of the instruction;
     unsatisfied waits within loop bodies similarly generate remote wait nodes and continue. Partially satisfied: already-satisfied
     bits pair normally with local posts (aggregated into 1 local wait node, Sec. 6.4); unsatisfied bits generate one
     remote wait node. A6 independent Wait instruction (Variable value comparison) when unsatisfied uses the same mechanism: top-level stall →
     stall resolution generates remote `Notify_Wait` (task id tagged `xwt`) then passes through.
   - The task graph does not spontaneously generate any nodes: all nodes come from original db data or microcode instruction translation
     (attachment dependencies expressed through extra edges, see Sec. 7.4.1).
- Local rank post-wait pairing proceeds normally: multiple queues within the same device, multiple dies' Events
  are all within the local graph; edges built per Sec. 6.4 rules.

### 6.7 Comparison with Checker CCU Graph Building Simplifications

**CCU task rank/eid/direction semantics**:

- **Channel-class network tasks** (write/read): local rank takes opDetails.rankId; peer rank resolved through
  channel peer eid (reid) via eid→rank index (Sec. 4.8); filled as INVALID +
  `rank_unresolved` count when not found. Equivalent to checker `GetRemoteRankByChannel`
  (`g_allRankChannelInfo`: channelId → peer), but using RankEidMap's communicator rank
  instead of checker's physical dstDeviceId.
- **Reduction tasks** (REDUCE microcode): CcuBuffer is a rank-local buffer (checker treats all msIds as always
  `GenSliceFromMs` + `AddLocalBatchReduce`); reduction is always a local on-chip memory operation; src/dst both filled as
  local rank; cross-rank data is already charged in preceding Rmt* role transfers; reduction does not repeat network charging.
- **Local tasks** (local transfer Memcpy, local reduction Reduce_Inline): src/dst both filled as local rank.
- **is_write**: CCU-produced UB tasks all explicitly carry (write=True / read=False);
  not dependent on simulator rank heuristic.
- **Local rank self-validation**: if channel leid ↔ index-resolved rank differs from opDetails.rankId,
  opDetails takes precedence with a warning (eid numbering offset defense, such as deviceId and rankId having an offset).

| Checker Step | light_performance | Rationale |
|--------------|-------------------|-----------|
| Two-phase (raw graph building → queue parallelization → reconnection) | Single-phase interpretation + assembly-phase queue parallelization splitting (Sec. 7.4.1: `ApplyCcuQueueParallelizationV3` simplified port; skeleton/unit splitting + fork/merge connection and attachment extra edges + `#v` virtual streams) | No verification needed; no need to preserve modifiable raw intermediate form; parallelization completed once in the assembly phase (reuse granularity differences in Sec. 7.4.1) |
| Loop merge (BatchTransMem/BatchReduce, resource conflict checking, producedMasks balancing) | **Equivalently implemented** (Sec. 6.5: prerequisite validation + folded emission + fallback full expansion; memory slice merging not needed — simulation only cares about data volume and timing) | Aligned with checker semantics to eliminate node inflation |
| SQE grouping into TaskCcuGraph (`IsContiguousCcuSqe` contiguous-only merging + `CcuMissionKey` latest-node-append) | **Equivalently implemented** (Sec. 6.1: contiguous segment splitting into dispatch contexts; each context with independent window replay + sequential args consumption; register/Event shared across contexts; same-queue contexts serial by dispatch order) | Aligned with checker multi-dispatch (window wraparound re-dispatch) semantics; previously merged into a single context, losing the second dispatch's data transfers (Sec. 10.3 #20 fixed) |
| GetRemoteRankByChannel (channelId → dstDeviceId/dstDieId, mapped to rank through Communicator table) | Peer eid (reid) → RankEidMap/ranktable index → communicator rank (Sec. 4.8/6.7) | Estimate mode Communicator table has only one row for the local rank; ccuChannels.rankId field is endpoint deviceId, unusable |
| AllRankParamRecorder complete address space mapping (Variable/Event/PFE/CHANNEL/JETTY/MISSION/CcuBuffer ...) + on-chip memory data content + remote address→register reverse lookup (`GetXnAndTypeIdByAddr`) | Only xn/cke/gsa/channel four categories retained; A6 Load/Store on-chip memory data plane and remote target reverse lookup not modeled (Sec. 6.4, `_MissionContextA6` deviation list) | Simulation only needs length and peer; data content/remote register number does not affect timing |
| A6 checker error abort scenarios (unsupported dmaOpCode/dstType, clearType!=1, clearMask!=0, illegal LoopGroup config, LoadX/StoreX indirect number out-of-bounds etc.) | Downgraded to warning + skip data plane (synchronization semantics still execute) or skip instruction | Performance tool tolerates incomplete/abnormal data; maintains queue skeleton integrity |
| CcuTraceInfo traceability, ccu_convert_config.json generation | Removed | Verification/rewrite tool exclusive |
| CheckAllPostMatch error reporting | Downgraded to warning + diagnostic statistics | Adapts to single-rank data |
| Cross-rank post unlocking others' waits (multi-rank joint advancement) | **No cross-rank edges built**: cross-rank post/wait generate remote sync nodes (fixed latency); when peer becomes ready approximated by fixed latency | Single-rank graph building (core difference) |

### 6.8 A5 and A6 Instruction Set Difference Overview

| Dimension | A5 (V1/Ascend950) | A6 (V2/Ascend960) |
|-----------|-------------------|-------------------|
| LOAD class | 7 instructions: includes Address family | 18 instructions: unified X register + LoadX/StoreX/ClearX/Nop + arithmetic-logic (Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt) |
| CTRL class | 5 instructions: Loop/LoopGroup/SetCKE/ClearCKE/Jmp | 7 instructions: + SetCKBit/Wait/Fence |
| TRANS class | 12 instructions: Mem↔CcuBuffer full combinations + SyncCKE/GSA/Xn | 7 instructions: local 4 types + unified TransMem (dmaOpCode distinguished) + SyncWtX/SyncAtX |
| REDUCE class | Add/Max/Min (msId list addressing, ≤8 CcuBuffer) | ReduceAdd/Max/Min (same structure, no embedded waitCKE) |
| Loop register | v1.loop{xnId, startInstrId, endInstrId}; loopCnt in Variable | v2.loop{xpId(context), xnId, xmId(iteration count), mode}; mode==1 conditional expansion (Sec. 6.5) |
| LoopGroup parameters | Variable: LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41]; Xm: gsaOffset/CcuBufferOffset/ckeOffset | Variable: loopInsCnt:10/expandOffset:9/expandCnt:9; Xm: ckOffset:10/msOffset:11/gsaOffset:32; Xp: xnIdOffset:32 |
| Register number addressing | Payload field direct use | `GetXnId`: when bit15=1, offset by loop xnIdOffset (Sec. 6.4) |
| Embedded sync fields | TRANS/REDUCE commonly carry waitCKE + setCKE + clearType | Local TRANS/REDUCE only setCKE (wait handled by independent SetCKBit/ClearCKE/Wait instructions) |
| Channel/length value source | Channel is instruction immediate field | Channel/length are both Variable register values (`Xn[xcId]`/`Xn[xlId]`) |

Both instruction set generations have been implemented for graph building (A6 interpreter `_MissionContextA6` reuses A5's task emission/
Event pairing/advancement/merge emission infrastructure; only instruction decoding and loop organization are overridden); A6 instruction →
task type full table in Sec. 11.3.3; established deviation list with checker in `_MissionContextA6`
class docstring and Sec. 6.7 comparison table.

---

## 7. Performance Simulation Module (Sub-Requirement 4, Incorporated per HCCLSIM Baseline)

> This chapter is the design description for the `simulator/` sub-package (transferred from `docs/light_performance/HCCLSIM/hccl_sim/`);
> interfaces are based on the HCCLSIM authoritative baseline. Differences from the old version and backlog items are in Chapter 10.

### 7.1 Overview

```python
sim = HcclSimulator(config, dep_graph, topo_resolver)
result = sim.run(trace_filepath=None, max_trace_events=0)   # → dict
```

The simulator accepts three parameters:

| Parameter | Type | Content | Builder |
|-----------|------|---------|---------|
| `config` | `SimConfig` | Latency, bandwidth, utilization configuration | Caller |
| `dep_graph` | `DependencyGraph` | Task nodes + dependencies | `graph_builder` (db path) / `input_builder` (JSON path) |
| `topo_resolver` | `TopoResolver` | eid → physical port resolution (ranktable.json) | Caller |

`run()` returns a result dict (no longer returns raw makespan):

```python
{
    "makespan": float,        # End-to-end duration (seconds)
    "makespan_us": float,     # (microseconds)
    "trace_file": str|None,   # trace output path
    "trace_events": int,      # trace event count
    "sim_time_s": float,      # Software execution duration (seconds)
    "tasks": list[Task],      # All tasks (including start/end_time)
    "tasks_by_stream": dict,  # int stream id → [Task] (chain order)
    "ports": dict,            # tuple(ports) → NetworkPort
}
```

The simulator does not accept task sequence lists and does not infer dependencies — all are obtained from `dep_graph`.

### 7.2 Parameter One: `config: SimConfig`

**Unit convention**: internal latency=seconds, bandwidth=B/s; **JSON configuration units: latency=us, bandwidth=GBps**
(`from_dict`/`to_dict` auto-convert). Port identifiers are `"die_id/port_number"` strings
(such as `"1/0"`), consistent with ranktable.json.

#### Field Definitions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `local_record_delay` | float | `1e-6` | Local record fixed duration; JSON value unit us |
| `remote_record_delay` | float | `1e-6` | Remote record fixed duration; JSON value unit us |
| `remote_wait_delay` | float | `1e-6` | Remote wait / unmatched local wait fixed duration; JSON value unit us |
| `port_bandwidth` | dict | `{}` | `"die/port"` → bandwidth (B/s). One eid can correspond to multiple ports, configured per port, total bandwidth summed |
| `default_port_bandwidth` | float | `53.125e9` | Default bandwidth for unconfigured ports (425 Gbps = 53.125 GBps); JSON value unit GBps |
| `soc_version` | str | `"Ascend950"` | Chip version; authoritative source is db `Device.soc_version` (cluster configuration `soc_version` field ingested through servers_info.json). Prefix-matches `ascend950`/`ascend960` family to select port bandwidth table; errors if no match (see "Port Bandwidth Table Selection" below) |
| `port_type` | str | `"x4"` | Port type (`x2`/`x4`); authoritative source is db `Server.port_type` (server/pod topology configuration `port_type` field ingested through servers_info.json). ascend960 selects port bandwidth table by it (`ascend960_x2`/`ascend960_x4`); ascend950 has only x4 ports; `x2` errors (see "Port Bandwidth Table Selection" below) |
| `hbm_bandwidth` | float | `1.4e12` | On-chip memory bandwidth, used for Reduce_Inline/Memcpy; JSON value unit GBps |
| `bw_profile_set` | BandwidthProfileSet? | `None` | Bandwidth utilization configuration set; when None, utilization is 100% (including multi-jetty scaling step `jetty_utilization_step`, see below) |

#### Port Bandwidth Configuration Format

```json
{"port_bandwidth": {"1/0": 53.125, "0/7": 53.125}}
```

#### Port Bandwidth Table Selection (`soc_version` + `port_type`/`topo_type`)

The port bandwidth tables in `sim_config.json` are divided by **chip generation** (`<table_name>_port_bandwidth`):

| soc_version Family | Selection Dimension | Table Name | Description |
|-------------------|---------------------|------------|-------------|
| `ascend950` | `topo_type` | `ascend950_8p_server` / `ascend950_1d_pod` | ascend950 has only x4 ports (no x2 port bandwidth table); configuring `port_type` as `x2` errors |
| `ascend960` | `port_type` | `ascend960_x2` / `ascend960_x4` | Selects table by server/pod topology configuration `port_type` field; **no bandwidth scaling** |

soc_version family is matched by prefix (case-insensitive; for example, `Ascend950DT_95A1` →
`ascend950`, `Ascend960DT_961BY` → `ascend960`); non-empty but not matching any family, or
selected table not in configuration → **error**.

The server/pod topology configuration file's `port_type` field (`x2`/`x4`, default `x4`, illegal values error;
ascend950 topology configuring `x2` is validated and errors by `generate_server_topo.sh`/`generate_cluster_topo.sh`
and the C++ cluster parser) is written by `generate_cluster_topo.sh` into the cluster-level
`servers_info.json` at each **leaf node** entry (server form: one entry per server; POD form: one entry per
board, with explicit `super_pod_id`/`server_id`/`board_id` triple id for the parser to match by
id rather than by positional order; this file is tool-exclusive, not entering topo.json/rootinfo/ranktable or other
hccl business native format files), then ingested by the C++ cluster parser (`ParseServersInfoJson`) into
`Server.port_type` column (`sim_models.h`/`db_sim_runner_ops.h`, RunnerDB v5; from v5 onward,
`Server` table adds `board_id` column; POD form has one `Server` row per board; server form
`board_id` is always 0).

During simulation, `pipeline`/`__main__`/`dump_graph` entry points call
`db_reader.load_soc_version(db_path, device_id)` to read the chip version from `hccl_sim.db`'s `Device`
table (defaults to `Ascend950` when old databases lack the `soc_version` column),
`db_reader.load_port_type(db_path, device_id)` to read the port type from the `Server` table (locates
the target device's server by `Device.id → server_id → Server.id`; falls back to the unique value in the full table when not located; defaults to `x4` when old databases lack the `port_type` column), then applies
`SimConfig.with_soc_and_port_type()` once to the configuration and re-selects the port bandwidth table (avoiding
intermediate-state false reports when values take effect one by one with old values). `get_port_bandwidth()` directly queries the selected table,
no longer performing x2 scaling; `d2h_bandwidth` and on-chip memory bandwidth are unaffected. Command-line
`--soc-version`/`--port-type` can override db values.

#### Bandwidth Utilization Configuration (`bw_profile_set`)

```
BandwidthProfileSet
├── profile_low:   BandwidthProfile  ← Used when shared-port concurrency count < 7
├── profile_high:  BandwidthProfile  ← Used when shared-port concurrency count ≥ 7
└── THRESHOLD: 7
```

Each `BandwidthProfile` is a data volume (32B~128MB) → utilization (0~1) mapping table; queries round down to the
largest tier not exceeding that data volume (floor matching).

JSON format:
```json
{
  "profile_low":  {"32": 0.30, "64": 0.35, "...": 0.90, "134217728": 0.95},
  "profile_high": {"32": 0.20, "64": 0.25, "...": 0.84, "134217728": 0.85},
  "single_port_utilization": 1.0,
  "jetty_utilization_step": 0.02745
}
```

> The actual key names in the file are `multi_port_utilization_low` / `multi_port_utilization_high`
> (`from_dict` also accepts the old key names `profile_low` / `profile_high`).

#### Multi-Jetty Scaling (`jetty_utilization_step`)

`multi_port_utilization_low` is calibrated based on the standard configuration of **one channel, one jetty**.
When `HCCL_UB_MULTI_CHANNEL_NUM=n`, one channel has n jettys; concurrent transfer capability scales approximately linearly.
Lightweight simulation does not build a jetty-level model (jettys are used only for eid/protocol resolution, see Sec. 5.5); instead, the
jetty count is converted into an overall uplift of the `profile_low` table:

```
util'(size) = min(1.0, util(size) + (n - 1) × jetty_utilization_step)
```

- `n` is parsed per operator with priority **CLI > db actual link value > env stored value**
  (`db_reader.resolve_group_jetty_num`):
  - **db actual value (primary data source)**: env is only business configuration intent; whether hcomm actually establishes multi-jetty
    links depends on the expansion mode/algorithm path — when `HCCL_UB_MULTI_CHANNEL_NUM` is configured but the actual setup is still
    "one channel, one jetty," scaling by env would be overly optimistic. Actual value has two paths (both are proxy
    actual stored results; takes the **minimum** across the operator's involved eid pairs/channels, conservative to avoid overestimation):
    - AICPU mode: runnerdb (`hccl_sim.db`) `RaJetty` table, aggregated by "local eid (ctx →
      `RaContext.endpoint_id` → `EndPoint.eid`) → peer eid (`peer_endpoint_id`)"
      for `mode=3` (USER_CTL_NORMAL) jetty row count (`load_pair_jetty_counts`,
      same granularity as dump-side `CreateJettyInfo`); excludes unattributed (`peer_endpoint_id=0`)
      and unconnected/destroyed (`peer_jetty_handle=0`) rows. The operator group's eid pairs are collected from SQE task
      meta's `lcEid`/`rmEid` and ccuChannels' leid/reid;
    - CCU mode: `ccuChannels.jettyNum` (the per-channel jetty list actually dispatched by hcomm AddChannel command);
      after the task `jettyId` matches an active channel, takes `len(jettyId[])`;
  - **env fallback**: when actual value is unavailable (old databases without `RaJetty` table / operator has no network tasks and no channel
    data), falls back to `opDetails.ubMultiChannelNum` (proxy level0 stub reads environment variable and stores,
    Sec. 4.2); when both are available and inconsistent, the actual value takes precedence and `[JETTY][WARN]` is logged;
  - When `n ≤ 1` (unconfigured/invalid falls back to 1, and actual is single jetty), the original table is used directly; no uplift;
- `jetty_utilization_step` value range `[0.0109, 0.044]`, default midpoint `0.02745`,
  configured only by `bw_profile.json` (clamped to range boundaries with WARNING on out-of-range);
- Only applies to `profile_low`: `profile_high` is the high-concurrency tier (ports already saturated by multiple transfers; multi-jetty
  provides no additional gain); single-port/`d2h` use constant utilization (`single_port_utilization` /
  `d2h_utilization`); neither is scaled;
- Entry point: `SimConfig.for_jetty_num(n)` → `BandwidthProfileSet.for_jetty_num(n)`
  (`bandwidth_profile.py`); pipeline generates a scaled config copy per operator;
  debugging can use `--jetty-num` to override the entire parsing result above (`run_perf_sim.py` /
  `python -m light_performance`).

### 7.3 Parameter Two: `dep_graph: DependencyGraph`

#### Four Layers of Content

```
DependencyGraph
├── tasks: dict[str, Task]                       ← Task nodes (with all parameters)
├── stream_chains: dict[int, list[str]]          ← Serial dependencies (int stream id)
├── notify_matches: dict[str, str]               ← Notify dependencies (wait_task_id → record_task_id)
└── extra_edges: dict[str, list[str]]            ← Additional dependency edges (child_task_id → [parent_task_id, ...])
                                                  ← Sources: CCU queue parallelization fork/merge edges and SQE attachment edges (Sec. 7.4.1)
```

#### Layer One: Task Nodes (`Task`)

Each Task carries all parameters needed for execution (invalid values uniformly use `INVALID = 0xFFFFFFFF`,
eid is empty string, string attributes are `""`):

| Field | Type | Applicable Task Types | Simulator Purpose |
|-------|------|----------------------|-------------------|
| `task_id` | str | All | Dependency graph index |
| `task_type` | TaskType | All | Determines latency model (internal classification, Sec. 7.3) |
| `task_name` | str | Display | trace/dump display name: profInfo.profTaskType op name (`_add_task` unified filling, Sec. 11.4.1); when empty, `display_name` falls back to `task_type.value` (CCU sub-tasks/CLI input) |
| `stream_id` | **int** | All | Serial chain membership |
| `notify_id` | int? | Notify_Record/Wait/Write_With_Notify | Dependency pairing identifier (CCU path = cke id); Write_With_Notify display only (its notify targets the peer; does not participate in pairing) |
| `mask` | int | Notify_Record/Wait | Notify bitmask (CCU path): notify_id + mask uniquely identify a notify action; wait node carries aggregated mask (one wait instruction's all contributing posts' consumed bits ORed, Sec. 6.4); SQE path and 0=unfilled |
| `src_rank` / `dst_rank` | int? | notify/transfer | Local determination (src==dst); read direction local determination |
| `size` | int | Memcpy/Reduce_Inline/Ub_*/Write_With_Notify | Compute latency + query utilization |
| `port` | str/list? | (Reserved field) | Ports resolved in real-time by eid; not filled in db path |
| `src_eid` / `dst_eid` | str | Ub_*/Write_With_Notify | eid → ranktable → physical ports |
| `transport_type` / `link_type` | str | Display | msprof reconciliation dimensions (profInfo transportType/linkType names, Sec. 11.4.2/11.4.3, `_add_task` unified filling; CCU sub-tasks not filled) |
| `data_type` | str | Display | trace/report annotation |
| `protocol` | str | Display | Channel CommProtocol name (Sec. 11.4.2 note; empty when no channel info) |
| `jetty_id` | int? | Ub_*/Write_With_Notify | trace annotation |
| `is_write` | bool? | Network tasks | Charging direction and port query eid selection (Sec. 10.3 #1) |
| `start_time` / `end_time` | float | All | Filled by simulator (-1=not executed) |

7 TaskTypes:

| TaskType | Latency Model | Description |
|----------|--------------|-------------|
| `NOTIFY_RECORD` | Fixed latency (local additionally triggers Event) | Local = `is_local_notify`(src==dst); unpaired local record only fixed latency |
| `NOTIFY_WAIT` | Wait for Event / fixed latency | Matched→wait for record's Event; unmatched/remote→fixed `remote_wait_delay` |
| `REDUCE_INLINE` | On-chip memory dynamic sharing | `size / hbm_bandwidth`, sharing bandwidth with other on-chip memory tasks |
| `MEMCPY` | On-chip memory dynamic sharing | Same as above (on-chip memory copy, not network transfer) |
| `UB_INLINE_WRITE` | Fixed latency (same as Notify_Record: local/remote record_delay) | Notify inline write; not produced by SQE graph building path; only used by old input path (task.txt) |
| `UB_WRITE_OR_READ` | Port bandwidth sharing (write/read directions charged independently) | Network data send/receive; `is_write` explicitly carried by graph building (SQE network tasks in both directions uniformly mapped to this type) |
| `WRITE_WITH_NOTIFY` | Port write direction sharing + post-write notify fixed latency | Network write + attached notify single task, corresponding one-to-one with db `WRITE_WITH_NOTIFY` (7); notify uses fixed remote record latency (peer wait not in this db) |

#### Layer Two: Serial Dependencies (`stream_chains`)

```
int stream id → [task_id_0, task_id_1, ..., task_id_n]
```

- List order = execution order: task[i] must complete before task[i+1] can execute
- Each stream_id launches one simpy process, executing sequentially through the chain via `yield`
- Different streams are naturally parallel
- db path stream id addressing: SQE streams use the original streamId; **CCU queue skeleton chains belong to their SQE original stream**
  (Sec. 6.5; whichever stream the CCU-type SQE is on, the expanded tasks are on that stream; only when the original stream is occupied by another die's
  queue is a synthetic stream allocated); virtual streams allocated from base `1<<20`
  (Sec. 6.5; virtual stream logical names carry `#v{n}` suffix); logical names retained through `stream_labels`;
  `GraphBuilder.stream_meta` provides real/ccu/virtual three-category classification and virtual stream
  parent_sid (Sec. 7.4.1; points to its queue's stream id, normally the SQE original stream) on top of this, for report/trace/dump annotation

#### Layer Three: Notify Dependencies (`notify_matches`)

```
wait_task_id → record_task_id
```

Explicit 1:1 pairing: which wait waits for which record. When multiple records share the same notify_id, only one
pairs with the wait; unpaired records use fixed latency.

#### Layer Four: Additional Dependency Edges (`extra_edges`)

```
child_task_id → [parent_task_id, ...]
```

- Semantics: child must wait for **all** parents to complete before starting (pure timing constraint; does not carry
  record/wait event semantics — event pairing is still expressed by `notify_matches`).
- Current sources: CCU queue parallelization fork edges (predecessor skeleton → unit head), mission completion set
  merge edges (skeleton tail + unit tails → SQE successor / same-queue next mission entry), CCU sub-graph
  attachment edges on the SQE stream (entry/exit, see Sec. 7.4.1), and aggregated wait node multi-parent edges
  (wait → all contributing posts, Sec. 6.4 — first contributing post uses `notify_matches`, others use this layer).
- Writer side `add_edge` auto-deduplicates (same child-parent pair recorded only once); simulator queries through
  `get_extra_parents` / `extra_parent_ids`.
- Acyclic: the above edges always connect "forward" along the original instruction order (skeleton before unit, unit before merge point,
  attachment at placeholder position before/after, post registered before the wait instruction that consumes it); graph building direction guarantees acyclicity,
  also counted in `_drop_cyclic_matches` cycle detection fallback (Sec. 5.4).

### 7.4 Simulator Consumption Flow

```python
def run(self, trace_filepath=None, max_trace_events=0):
    # 1. Retrieve all task nodes
    tasks = list(dep_graph.tasks.values())

    # 2. Create synchronization primitives: only matched waits create Events
    for wait_task_id in dep_graph.matched_wait_ids:
        notify_events[wait_task_id] = env.event()
    for parent_id in dep_graph.extra_parent_ids:  # all parent tasks from extra edges
        task_done_events[parent_id] = env.event() # Must be pre-created before stream processes start:
                                                  # if parent completes first and child creates Event later, it waits forever (Sec. 7.4.1)

    # 3. HBM bandwidth manager (Reduce_Inline + Memcpy shared, utilization always 1.0)
    hbm_mgr = _BandwidthManager(env, config.hbm_bandwidth, empty utilization table)

    # 4. Launch one simpy process per stream, executing serially in chain order
    for task in chain:
        for pid in dep_graph.get_extra_parents(task.task_id):
            yield task_done_events[pid]           # extra edges (Sec. 7.4.1), start_time does not include waiting
        if NOTIFY_RECORD:
            yield timeout(calc_notify_record_delay(task, config))
            if local and matched: trigger matched wait's Event
        elif NOTIFY_WAIT:
            if local and matched: yield notify_events[task_id]   # wait for Event
            else:              yield timeout(remote_wait_delay)
        elif REDUCE_INLINE / MEMCPY:
            yield hbm_mgr.send(task_id, size)                 # HBM dynamic sharing
        elif UB_INLINE_WRITE / UB_WRITE_OR_READ / WRITE_WITH_NOTIFY:
            is_write = task.is_write  # Explicit direction; old input path None falls back to heuristic
            eid = src_eid if is_write else dst_eid
            ports = topo_resolver.get_ports_by_eid(eid)       # Throws if missing
            port = lookup/create NetworkPort(ports, total bandwidth)
            yield port.send(task_id, size, is_write)          # Port bandwidth sharing
            if WRITE_WITH_NOTIFY:
                yield timeout(calc_write_with_notify_delay(task, config))
                                                           # Post-write notify:
                                                           # Fixed record latency
                                                           # (peer wait not in this db)
        task.end_time = env.now
        if task_id in task_done_events: task_done_events[task_id].succeed(env.now)
    # 5. makespan = max(end_time); assemble result dict (including trace export)
```

Latency formulas are unified and maintained at a single point in `task_timing.py`.

#### 7.4.1 CCU Queue Parallelization and Virtual Streams (Extra Edges)

**Problem** — The CCU queue chains produced by Sec. 6.5/6.6 are task-by-task serial flows: all
data transfers, waits, and sets within the same queue execute in order. On real hardware, a queue's asynchronous transfer units execute concurrently;
serialization causes three distortions:

| Distortion | Mechanism |
|------------|-----------|
| makespan overestimation | Asynchronous transfers are separated by skeleton (wait/set) one by one, preventing overlap |
| Utilization tier distortion | Concurrent transfers within the same queue degrade to sequential execution; `_BandwidthManager` observes port concurrency always 1; always queries profile_low tier (Sec. 7.5) |
| Port peak concurrency statistic distortion | `peak_concurrency` always 1; cannot reflect concurrent bandwidth sharing behavior (Sec. 8.2 port-level metrics) |

**Checker reference** — `ApplyCcuQueueParallelizationV3` (`ccu_task_transform_v3.cc:1052`):
secondary traversal transformation after raw graph building; `CollectAsyncBoundRecordNodesV3` collects asynchronous transfer nodes;
`QueueIdReuseAllocatorV3` reuses virtual queue ids by `[startNodeId, afterNode]` intervals;
fork/join edges reconnect to skeleton.

**Simplified port** (`ccu_graph_builder.py` assembly loop, completed once during assembly phase, no raw
intermediate form) — splits each mission's queue node sequence:

1. **Unit identification**: `_QUEUE_DATA_TYPES` (`Memcpy`/`Ub_Inline_Write`/
   `Ub_Write_Or_Read`/`Reduce_Inline`) data tasks + immediately following consecutive
   `NOTIFY_RECORD` chain = one **asynchronous unit**; other nodes (wait / free set /
   remote sync nodes) are **skeleton**, staying on the queue serial chain — the queue chain is the skeleton itself;
   no head/tail placeholder nodes are introduced.
2. **Fork edges**: `add_edge(unit head, preceding skeleton node)` — the unit must wait for the queue to advance
   to its predecessor skeleton before starting (adjacent units share the same fork parent node); units appearing before the skeleton
   (mission first node is a data task) have no skeleton predecessor; their entry dependencies are expressed by attachment edges instead.
3. **Completion set merge**: mission completion set = skeleton tail + all unit tails, merged through extra edges to
   — the same-queue next mission's entry (skeleton head + pre-skeleton unit head; queue serial semantics;
   skeleton tail ordering is already covered by the queue chain serial) and SQE stream successor tasks (exit edges,
   added by graph_builder during the attachment phase).
4. **Virtual stream reattachment**: per-queue counter/reuse pool (`vstream_seq`/
   `vstream_free`/`vstream_busy`); unit tasks' `stream_id` changed to point to
   `ccu_d{die}_q{queue}#v{n}` (same `1<<20` base address space); unit chains registered in
   `result.virtual_chains` and `add_stream_chain` — serial chains constrain only skeleton and
   unit-internal task order.
5. **SQE stream attachment** (`graph_builder._expand_ccu` consumes `result.attach`):
   CCU placeholder removed from SQE stream chain (task graph produces no placeholder/spontaneous nodes);
   entry edges = SQE predecessor tasks (including immediately preceding CCU mission completion set) → skeleton head and
   pre-skeleton unit head; exit edges = mission completion set → SQE successor tasks. All expressed through extra edges;
   timing semantics: SQE predecessor (and preceding mission completion set) must complete before the sub-graph can start;
   sub-graph mission completion set must all finish before SQE successor tasks can start.
   Empty chains are removed for SQE streams where all tasks are placeholders.

**Event timing preservation**: Splitting does not modify `add_match` pairing — which unit a record falls into, or a wait
staying on the skeleton, does not affect event pairing (wait crosses streams via notify_events). In the measured sample,
the trailing memcpy unit's fork parent wt28 happens to be on the preceding read unit's tail set accounting wait;
cross-stream event + fork edge together guarantee the original "read complete → trailing copy" timing. **Exception**: merged loop
emission window mid-body wait/record pairing is a folded pseudo-serialization barrier (per-iteration pipeline handshaking
folded into a full barrier); resolved by Sec. 6.5.1 signature chain resolution, pruned before assembly — this section's splitting is based on the post-pruning
sequence; skeleton/unit/fork structure is unaffected.

**Reuse rule differences from checker**:

| Dimension | Checker (V3) | light_performance |
|-----------|--------------|-------------------|
| Reuse granularity | afterNode interval (join successor node) level | Mission level (all preceding released when entering a new mission) |
| Intra-mission reuse | Allowed (DAG explicit edges; no pseudo-dependencies) | Not reused (units within the same mission are concurrent; chain model would introduce pseudo-serialization with same-stream reuse) |
| Residual strictness | GRAPH_UNMATCHED error exit | Tolerated (performance tool; Sec. 6.4 degraded warning) |

Cross-mission reuse safety: units reusing the same virtual stream are serial within that stream chain (the later mission's
unit chain follows the earlier mission's unit chain); no time-axis overlap.

**Acyclic guarantee**: fork (skeleton → unit head) and completion set merge edges (unit tail → SQE successor /
next mission entry) always connect "forward" along the original instruction order; splitting does not change the original order;
`_drop_cyclic_matches` cycle detection also counts extra edges (Sec. 5.4) as fallback; the simulator
additionally has runtime deadlock detection (errors and exits when task `end_time<0`).

**Statistics and labeling**: `CcuBuildStats` carries parallelization (`parallel_units`/`virtual_streams`/
`fork_edges`/`join_edges` — join_edges now counts completion set merge and SQE attachment edges), remote sync
(`local_posts`/`remote_posts`/`wait_matched`/`remote_waits`) counts;
summary line presents as `pu/vs/lp/rp/wm/rw`;
`GraphBuilder.stream_meta` classifies by logical stream name — `s{n}` → real, containing `#` →
virtual (with parent_sid), others → ccu; consumed at three locations: report (`Stream count:
N (including virtual M)`, virtual stream annotated as `vir_stream {n} (label)`, n numbered from 0
by allocation order; detail folded when exceeding 64 streams), trace (virtual stream thread_name `vir_stream
{n} ({label})` + task args `virtual_stream: true`), dump (streams[] with
kind/virtual/parent_sid,
extra_edges list; DOT purple dashed cluster + blue dotted extra edges; HTML badge
+ dotted arcs). stream_meta default (simulator standalone CLI old input) produces no annotation; backward compatible.

**Quantified effect** (sample db REDUCE_SCATTER iter0, 66 tasks): queue parallelization makes makespan
20,404.20 → 5,149.36 us (≈3.96x); subsequent two corrections further reduce to **2,753.71 us**:
① remote CcuBuffer reduction correction (Sec. 6.3.2; reduction always local on-chip memory) — original second phase 128MB reduction charged as
network read ~2526 us, corrected to on-chip memory ~96 us; ② cross-rank sync node completion (Sec. 6.6) —
14 waits depending on remote assertion were previously skipped (read units launched early at t≈0), corrected to simulate at fixed
remote latency. 10 asynchronous units reattached to 10 virtual streams (fork edges 10; attachment/merge edges generated by
SQE neighbors and queue serial conditions) — first phase 8 units concurrent (7 network read units each
~2526 us, belonging to 7 independent ports `1/0`..`1/6` + 1 local memcpy ~96 us), followed by
reduction ~96 us and trailing memcpy ~96 us in serial connection. Note that single-port read peak concurrency is still
1 — the benefit comes from **cross-port overlap**, not single-port concurrency improvement.

### 7.5 Bandwidth Sharing Mechanism (`_BandwidthManager`)

#### Generic Manager (Port and On-Chip Memory Share the Same Model)

- **Single management process model**: when a new transfer joins, `interrupt` the management process; the management process equally divides bandwidth by current concurrency N:
  `bw_per = total_bandwidth / N`.
- **Utilization query basis is `min_remaining`**: concurrent tasks within the same group use "minimum remaining data volume" to query
  the utilization table (concurrency <7 uses profile_low; ≥7 uses profile_high); same basis within the group.
- **profile_low table already scaled by jetty count**: before querying, the pipeline has already used
  `SimConfig.for_jetty_num(n)` to generate a config copy; `n` parsed per operator as "db actual link value
  (`RaJetty`/`ccuChannels`) preferred, `opDetails.ubMultiChannelNum` (env
  `HCCL_UB_MULTI_CHANNEL_NUM`) as fallback"; when multi-jetty (`n>1`), table values are overall
  uplifted; the manager itself is unaware of jettys (Sec. 7.2 "Multi-Jetty Scaling").
- Effective bandwidth = `(total_bandwidth / N) × utilization(min_remaining, N)`; computes earliest completion time,
  timeout settles progress; recalculates on join/completion.
- **On-chip memory dynamic sharing**: `Reduce_Inline` and `Memcpy` share one on-chip memory manager
  (utilization always 1.0) — when multiple on-chip memory tasks are concurrent, they equally share on-chip memory bandwidth (different from the old per-task independent
  `size / hbm_bandwidth`; see Sec. 10.3).

#### Bidirectional Independence

Each `NetworkPort` contains write and read two independent managers with independent bandwidth that does not interfere:
when port bandwidth B is configured, the write direction can use B while the read direction can simultaneously use B.

#### Multi-Port Aggregation

When one eid corresponds to multiple physical ports, `port_key = tuple(port_ids)` serves as the logical port;
total bandwidth = sum of each port's bandwidth (`get_ports_total_bandwidth`).

### 7.6 Parameter Three: `topo_resolver: TopoResolver`

- Input `ranktable.json` (`TopoResolver(ranktable_path=...)`); provides
  **eid (16B hex string) → physical port list (`["die/port", ...]`)** resolution.
- Transfer task routing: write direction queries `task.src_eid`; read direction queries `task.dst_eid`.
- **Throws directly when eid cannot resolve to a port** (no jettyId fallback, no default port) — therefore the db path
  skips simulation entirely when "network tasks exist but no ranktable" (Sec. 3.3); db graph building counts and warns for tasks with unresolvable
  eid (Sec. 5.5).
- Port bandwidth queried and summed for each resolved port individually from `SimConfig.port_bandwidth`.

### 7.7 Construction Methods and CLI

All input methods uniformly construct `(config, dep_graph)`:

| Input Source | Construction Function | Returns | Status |
|-------------|----------------------|---------|--------|
| **hccl-vm db (new in this iteration)** | `graph_builder.build(group)` | `(config, dep_graph)` | Chapters 5, 6; primary input |
| JSON task file | `load_dep_graph(filepath)` + `load_config(...)` | `(config, dep_graph)` | Retained (hccl_sim compatible) |
| task.txt / algorithm-generated old input sources | — | — | **Removed with HCCLSIM version** (see Sec. 10.3) |

Construction flow unified as:
1. Create `Task` node, call `dep_graph.add_task(task)`
2. Call `dep_graph.add_stream_chain(stream_id, task_ids)` by task order within stream
3. Call `dep_graph.add_match(wait_task_id, record_task_id)` by notify pairing result

CLI entry points:

```bash
# db input (light_performance primary entry, see Sec. 3.3)
python3 -m light_performance --db data/hccl_vm_data.db \
    --config sim_config.json --bw-profile bw_profile.json --ranktable ranktable.json -v

# One-click task graph dump (repository root)
python3 dump_graph.py data/hccl_vm_data.db --html

# hccl_sim compatible entry (JSON task file)
python -m simulator --tasks tasks.json --config sim_config.json \
    --bw-profile bw_profile.json --ranktable ranktable.json -v --trace trace.json
```

`--config` overrides default latency/bandwidth for each mode (units us/GBps); `--bw-profile` independently loads
utilization configuration; report and trace are generated by `report.py`'s `generate_report` / `export_trace`
from the `run()` result dict.

Trace files are in Chrome Trace Event format: `ts`/`dur` values are in **us** (format specification convention);
the file top-level carries `"displayTimeUnit": "ns"` specifying the viewer's (chrome://tracing etc.)
Start/Wall Duration **display** unit as ns (default ms).

`export_trace` display adaptation conventions:

- **tid sequential numbering**: thread rows mapped to tid `0,1,2,...` in stream id ascending order (when original
  streamIds 2/6..13 are used directly as tid, viewers that sort by string would place 10..13 before 2);
  the real stream id is retained in thread_name label and `args.stream_id`.
- **Endpoint rounding**: `ts`/`end` rounded independently before computing `dur` — `ts` and `dur` rounded independently
  would produce 0.001us pseudo-overlap; same-tid overlapping slices would be nested into sub-rows by the viewer (displayed on new lines).
- **Virtual stream annotation** (Sec. 7.4.1): db path passes stream metadata through `export_trace(..., stream_meta=...)`;
  virtual stream thread_name changed to `vir_stream {n} ({label})` (n numbered from 0 by
  sid ascending order, i.e., allocation order, avoiding the large stream numbers starting from 2^20 base; real stream number in
  in-row task `args.stream_id`); its task args append `virtual_stream: true`;
  real/ccu stream names unchanged (CCU queue chain logical names appear only in text report rows). stream_meta
  default (simulator standalone CLI old input) produces no annotation; backward compatible.
- **Notify task args carry mask**: CCU path `Notify_Record`/`Notify_Wait`
  task args append `mask: "0x…"` (notify_id + mask uniquely identify a notify action,
  Sec. 7.3); SQE path and mask=0 do not output this field.
- **Operator summary first row (op_summary)**: when pipeline calls
  `export_trace(..., op_summary=...)` per operator, each operator (or each pid process group when multiple operators are merged for export)
  reserves an `op_summary` first row with tid 1 + `thread_sort_index` 0; task row
  tid and sort_index follow sequentially (display layer only; tid avoids 0 — some viewers such as Perfetto treat
  tid 0 as idle thread; `thread_sort_index` is the Chrome Trace standard row order control);
  summary events placed at the front of the event array; triple guarantee that op_summary is always the first row. The first row contains a
  summary `X` slice spanning the operator's start to end (start time and duration are the slice's own ts/dur, taking
  earliest task start to latest task end); row label `"<op_name> | <algorithm_name>"`;
  args carry: `algorithm_name` (algorithm selection result, opDetails.algorithmName log
  capture backfill (DFX primary path, Sec. 4.2 note)), `data_type`/`data_count` (element count; for V2 layout operators,
  send/receive element counts and types for both sides; others use count and single-side type), `task_count` (total task count after graph building,
  i.e., the count after CCU mode microcode instruction translation into task graph), and op_type/
  comm_id/op_iter/rank/rank_size/input_bytes/output_bytes (opMemInfo's
  inputSize/outputSize, computed from business API parameters at collection time) context; summary events are not counted toward
  `max_events`.

### 7.8 Alignment Adaptation Completion Status

Differences between the old version (before this repository's changes) and the HCCLSIM baseline have been resolved in this alignment:

| # | Item | Handling |
|---|------|----------|
| 1 | Task model 6 types → new 6 types (local/remote record/wait merged into Notify_*; LOCAL_COPY split into Memcpy/Reduce_Inline; network transfer changed to Ub_*; later expanded to 7 types with db addition of WRITE_WITH_NOTIFY, Sec. 10.3 #19) | `task_model.py` replaced per HCCLSIM |
| 2 | stream_id str → **int**; notify_id str → **int** (INVALID=0xFFFFFFFF) | `dependency_graph.py`/graph building side synchronized |
| 3 | Port specified by `port` int field → **eid real-time resolution** (write queries src_eid / read queries dst_eid) | `topo_resolver.py` + graph building side fills eid |
| 4 | Unit system: JSON latency s → **us**, bandwidth B/s → **GBps**, port key int → **"die/port"** | `config.py` from_dict conversion |
| 5 | On-chip memory latency `size/bw` independent → **dynamic sharing** (Reduce_Inline+Memcpy share manager) | `_BandwidthManager` reuse |
| 6 | `run()` returns makespan → **result dict**; simulator two-param → **three-param** (+topo_resolver) | `simulator.py` |
| 7 | Latency formulas scattered → `task_timing.py` single point | Incorporated |

Unmatched legacy items (including `src_rank==0` hardcoding, no eid fallback, broadcast pairing triggering only first wait etc.)
are uniformly moved to Chapter 10 for tracking.

---

## 8. Performance Profiling Output (Sub-Requirement 5, Overview)

### 8.1 Output Location and Files

```
<installRoot>/plugin/light_performance/data/<timestamp>/
├── perf_profiling.json      # Primary artifact: performance metrics + task graph summary
├── trace_<commId>_<opIter>.json   # (Optional) Chrome tracing, chrome://tracing visualization
└── light_performance.log    # Runtime log
```

- Directory naming: `hccl-vm plugin run @estimate` one-shot execution channel does not pass data_id (Sec. 3.2);
  uses run timestamp as substitute; independent debugging mode (run_perf_sim.py) specifies output path via `-o`.
- Output is fixed in the plugin data directory rather than `data/`: the host machine clears `data/` at each start (`src/main.cc:32`);
  artifacts under the plugin directory are preserved across sessions; `archive.sh` archiving covers only logs/ and data/; plugin artifacts
  do not participate in automatic archiving (Sec. 10.3 #15).

### 8.2 Metric Definitions (Draft)

| Level | Metric | Source |
|-------|--------|--------|
| Operator-level | opType/opIter/rankSize/dataSize, makespan, software simulation duration | opDetails + `sim.run()` result dict |
| Operator-level | Operator summary first row (algorithm_name/data_type/data_count; start/end and duration carried by slice ts/dur, Sec. 7.5 op_summary) | opDetails.algorithmName (DFX/logging dual-path capture) + simulation task timeline |
| Operator-level | Total transfer volume, effective bandwidth utilization (total transfer / (makespan×port bandwidth)) | Summarize result["tasks"] |
| Stream-level | Per-stream start/end time, duration, task count + virtual stream annotation (Sec. 7.4.1) | result["tasks_by_stream"] + stream_meta |
| Graph-level | Async unit count, virtual stream count, fork edge count, completion set/attachment edge count (Sec. 7.4.1); local/remote post count (lp/rp), local wait pair count (wm), remote wait count (rw) | CcuBuildStats (parallel_units / virtual_streams / fork_edges / join_edges / local_posts / remote_posts / wait_matched / remote_waits, summary line `lp/rp/wm/rw`) |
| Port-level | Per-port peak concurrency, write/read transfer volume and average bandwidth | result["ports"] (NetworkPort) |
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

> `cross_rank_notify` is the retained cross-rank notify node count (diagnostic information): these nodes are graph-built normally in stream order,
> use fixed remote latency, but do not participate in local pairing (no cross-rank record-wait edges); this count is used to identify
> data/classification anomalies (such as local notifies misidentified as cross-rank). `kind`/`virtual_stream_count`/
> `extra_edge_count` are new fields from CCU queue parallelization (Sec. 7.4.1): stream classification real/ccu/virtual,
> virtual stream count, and extra edge (fork/merge/attachment) count.

---

## 9. Appendix

### 9.1 Checker Graph Building Module → light_performance Graph Building Module Full Comparison

| Aspect | Checker (C++) | light_performance (Python) |
|--------|---------------|---------------------------|
| Objective | Correctness verification DAG (mainStart reachable, supports various checks) | Simulation DependencyGraph (three layers: tasks / stream_chains / notify_matches) |
| Input | db (through StorageManager/sim::Query) | Same db (sqlite3 direct read + struct decoding) |
| Graph building scope | **Multi-rank joint graph building** (all rank tasks in one DAG) | **Single-rank graph building** (no cross-rank pairing edges; remote sync tasks retained) |
| SQE translation | TaskMetaTranslatorV3 (including MemSlice address resolution) | Sec. 5.2 mapping table (only takes len/jettyId/notify fields) |
| WRITE_WITH_NOTIFY consumption | Splits at decoding boundary: `DecodeAndExpandTaskMeta` expands to [write, NOTIFY_RECORD] two sub-tasks | **Not split**: single-task mapping `Write_With_Notify` (Sec. 5.2); one-to-one correspondence with db tasks; trace/dump/statistics granularity consistent |
| Dependency edges | Stream chain + local/cross-rank notify edges (queue simulation) | Stream chain + local-only notify first-come-first-served (two-pass scan) + extra edges (CCU fork/merge/attachment, Sec. 7.4.1) |
| CCU | Two-phase expansion + queue parallelization + loop merge + Event accounting | Single-phase interpretation + assembly-phase queue parallelization (Sec. 7.4.1) + loop merge (Sec. 6.5) + Event accounting (Chapter 6) |
| Post-verification processing | Memory conflict/semantic/sync conflict/reachability/single-task checks | None |
| Output | Verification conclusion + dump | (SimConfig, DependencyGraph) → makespan → profiling json |

### 9.2 Glossary

| Term | Description |
|------|-------------|
| SQE | Submit Queue Element; task unit submitted from AICPU/SDMA side; one row in the opTask table |
| WQE | Work Queue Element (UB/DB mode); not involved in this iteration |
| notify | AICPU-side event synchronization primitive; record asserts / wait waits; paired by (recordDev, waitDev, notifyId) |
| Event | CCU-side event synchronization register (bitmap); post asserts / wait waits; paired by (waitDev, dieId, ckeId, mask) |
| jetty | Network card (communication engine) port number; determines memcpy bandwidth attribution |
| die | Chip die; CCU microcode space and Events are both organized by (device, die) |
| mission | One CCU task dispatch (instStartId/instCnt define instruction window; args[13] pass parameters) |
| makespan | Task graph end-to-end duration (latest task end time) |
| Main stream / Slave stream | Operator main stream marked by opDetails.streamId; others are slave streams (first WAIT, last RECORD structure) |

---

## 10. HCCLSIM Alignment Backlog and Matching Checklist

> This chapter records the **field gaps, task type mismatches, and other conflict points** confirmed during the alignment of the "db graph building path + design document" with the HCCLSIM baseline.
> Current implementations have all been landed per "default fill invalid values / approximate mapping."
> This is documented here for future joint resolution with the collection side and simulation module.

### 10.1 Task Field Availability Table (db Graph Building Path)

New model Task field sources and defaults in the db graph building path (invalid value convention: int fields `INVALID=0xFFFFFFFF`,
eid empty string, other strings `""`):

| Task Field | db Source | Availability | Default/Approximation | Future Matching Path |
|------------|----------|-------------|----------------------|---------------------|
| `task_id` / `stream_id` | Global sequence / streamId+synthetic base | Available | — | — |
| `notify_id` | opTaskMeta `NotifyTask.notifyId` | Available | — | — |
| `size` | MEM_CPY `len` / REDUCE `dataCount` / CCU length register | Available | — | — |
| `jetty_id` | SQE table column / ccuChannels.jettyId array | Available | INVALID (non-network tasks) | — |
| `link_type` | opTaskMeta `HcclTaskProfInfo.linkType` (v3, enum name mapping Sec. 11.4.2) | Available (v3) | "" (v2 old BLOB without profInfo displays `Unknown`) | — (Sec. 11.4 consumed) |
| `protocol` | ccuChannels.protocol (CommProtocol enum name mapping) | Available | "" (when no channel info; trace displays `N/A`) | — (migrated from `link_type`, Sec. 11.1 approach a) |
| `src_eid` (local, write) | SQE `lcEid` (v2 collection new, Sec. 4.4.1) preferred; defaults to jettyId → this device outbound channel `leid` derivation | Available (v2) / approximate (fallback) | Empty→any local channel leid fallback | — (v2 already supplemented explicit field) |
| `dst_eid` (peer) | Channel `reid` preferred; falls back to SQE `rmEid` | Approximate | "" (tasks without channel info) | Collection side to supplement explicit peer eid |
| `src_rank` / `dst_rank` | taskData endpoint rankId (carried in deviceId field, Sec. 4.4.1 rank field rollback) consumed directly: write=src local/dst peer, read=src peer/dst local; CCU/unresolved falls back through eid→rank index (Sec. 4.8) | Available/index fallback | INVALID + `rank_unresolved` count when both explicit and index are unresolved | — (Sec. 5.2/5.5/6.7) |
| `transport_type` | opTaskMeta `HcclTaskProfInfo.transportType` (v3, enum name mapping Sec. 11.4.3) | Available (v3) | "" (v2 old BLOB without profInfo displays `Unknown`) | — (Sec. 11.4 consumed) |
| `data_type` | REDUCE task-level `dataType` (v2 new, enum name mapping int8..bfp16); MEM_CPY task-level has no source | Partially available (v2) | REDUCE fills enum name; MEM_CPY "" | MEM_CPY collection side to supplement |
| `port` | (Reserved field) | Not filled | None | Simulator resolves ports in real-time through eid (Sec. 7.6) |

**Instruction-level gaps (CCU microcode expansion)**: peer rank for channel-class network tasks is already resolved through eid→rank index
(Sec. 4.8/6.7); reduction tasks are always local (CcuBuffer is a rank-local buffer, Sec. 6.3.2); no peer eid
needed. CCU-produced UB tasks all explicitly carry `is_write` (Sec. 6.7); direction determination does not depend on rank
values.

### 10.2 Task Type Mismatch Table

| db taskType / Old Type | New Model TaskType | Mismatch Handling |
|------------------------|-------------------|-------------------|
| SYNC_STREAM (taskType=6) | **No corresponding type** | Skipped, no node generated (host-side stream sync point, not a device-side task, Sec. 5.6); statistic `skipped_sync` |
| AIV_GRAPH (taskType=5) | None | Skipped + warning (old and new models agree; both unsupported) |
| WRITE_WITH_NOTIFY (taskType=7, db new addition) | `WRITE_WITH_NOTIFY` (new, 7th type) | Single-task mapping (**not split** into write + NOTIFY_RECORD): write always write direction uses port bandwidth sharing; attached notify uses fixed record latency (Sec. 5.2; Sec. 10.3 #19 supported) |
| Old `REMOTE_RECORD` / `REMOTE_WAIT` | Merged into `NOTIFY_RECORD` / `NOTIFY_WAIT` (src≠dst expresses remote) | db single-rank path cross-rank notify generates remote nodes with `src≠dst` (Sec. 5.2; nodes retained, not paired) |
| Old `MEMCPY` (network transfer) | Semantics changed: new `Memcpy` = on-chip memory copy | Network transfers now expressed by `Ub_Write_Or_Read` (write/read direction distinguished); `Ub_Inline_Write` retains only notify inline write semantics (fixed latency) |
| Old `LOCAL_COPY` | Split into `Memcpy` (pure copy) / `Reduce_Inline` (reduction) | REDUCE non-network → Reduce_Inline |
| Old `MemcpyDirection` enum (direction field) | Deleted | Direction expressed by task type + rank/eid values (Sec. 5.5) |
| Old `sync_stream_delay` configuration item | Cancelled along with type gap | Reused as local_record_delay |

### 10.3 Other Conflict Points (Future Matching)

1. ~~**`src_rank == 0` local hardcoding** (`simulator.py` TRANSFER branch): read-direction tasks where
   the peer happens to be rank 0 with explicit value would be misidentified as write.~~ **Fixed**: Task adds explicit
   `is_write` field (filled by db graph building path; task.txt old path None falls back to heuristic); simulator
   consumes it with priority; taskData peer rankId (carried in deviceId field) is simultaneously filled directly into src/dst_rank (Sec. 5.2/5.5).
2. **TopoResolver has no fallback**: eid not found in ranktable throws directly (no jettyId→port mapping,
   no default port). db path avoids this with "network tasks exist but no ranktable → skip simulation, only dump"
   (Sec. 3.3); future consideration to integrate ccuChannels/JettyMaps to build a fallback mapping.
3. **Broadcast pairing triggers only first wait** (matched record's `break`): when multiple waits pair with the same
   record, subsequent waits' Events are not triggered; end_time remains -1 (tasks silently lost).
4. **`is_matched_record` linear scan**: `dependency_graph.py` performs O(R×M)
   traversal over matches; large-scale db (tens of thousands of tasks) has performance risk; recommend maintaining a record→wait reverse index (also
   resolving #3).
5. **SimConfig unit system and port key change**: JSON latency s→**us**, bandwidth B/s→**GBps**,
   port key int→**"die/port"** string; old configuration files are incompatible (sample
   `src/plugin/light_performance/sim_config.json` synchronized in this iteration).
6. **On-chip memory dynamic sharing semantic change**: old version per-task independent `size / hbm_bandwidth`; new version
   Reduce_Inline+Memcpy concurrent equally shares on-chip memory bandwidth (manager with utilization always 1.0); result granularity
   differs.
7. **Utilization query basis `min_remaining`**: concurrent tasks in the same group use the minimum remaining data volume to query utilization tier;
   different from the old version's "each task queries by its own data_size" — small tasks in the group drag down large tasks' utilization tier.
8. **Old input sources removed**: task.txt parsing, AllToAll generator, built-in demo removed from input_builder with the HCCLSIM version; log replay debugging requires separate restoration.
9. **TopoResolver has no net_type filtering**: no priority rule when the same eid has multiple network type entries.
10. **Profiling json not implemented** (Sub-Requirement 5): current output is task graph dump (DOT/JSON/HTML)
    + text report + Chrome trace; perf_profiling.json format is a draft in Chapter 8.
11. ~~**local_rank fallback convention**: when opDetails.rankId is missing, falls back by `deviceId−1`
    (following task.txt path convention); deviceId and rank numbering have no inherent relationship; ranktable can
    be more precise.~~ Peer rank is already resolved by eid→rank index (RankEidMap primary path + ranktable
    fallback, Sec. 4.8); no longer depends on numbering convention; local side still uses opDetails.rankId as primary
    (channel leid cross-validation, Sec. 6.7).
12. ~~**CCU queue parallelism not modeled**: CCU queue chains execute as independent synthetic streams in parallel; real hardware
    queue concurrent resource constraints not modeled (checker side has queue parallelization transformation; this plugin Sec. 6.7 simplified).~~
    **Queue parallelization implemented** (Sec. 7.4.1: assembly-phase skeleton/unit splitting + fork/merge/attachment extra edges +
    `#v` virtual streams reused across missions). Still not modeled: real hardware queue concurrent resource upper limit —
    virtual stream count is unlimited (checker V3 is also unlimited); future per-chip-generation capping possible.
13. **@estimate one-shot channel does not pass data_id**: `plugin run @estimate` special-case branch directly executes
    entry (`cmd_base_utils.cc:626-663`); does not go through the general message path's `DumpData` (whose generated
    `data_id` is currently the fixed value "1234"); therefore the output directory is named by run timestamp (Sec. 8.1); future
    switch to daemon model can align data_id semantics.
14. **`plugin install @estimate` semantic degradation**: install uses `RegisterPlugin` daemon startup
    (fork+pipe); main.py completes one-shot execution and exits after stdin EOF; MonitorThread reclaims —
    behavior is harmless but inconsistent with install's daemon semantics; true daemon requires implementing a stdin command loop
    (Sec. 3.2 evolution direction).
15. **Plugin artifacts do not participate in automatic archiving**: `archive.sh` archives only `logs/` and `data/`;
    `plugin/light_performance/data/` perf_profiling.json artifacts do not enter the archive directory
    (benefit: unaffected by `remove_all("data")` at start); archiving requires extending archive.sh.
16. **simpy deployment dependency**: target environment must have simpy pre-installed (not distributed with hccl_vm_install as offline wheel);
    main.py should provide a clear error and installation prompt when import fails at startup (Sec. 3.4).
17. ~~**db_reader.py not synchronized with v3 collection layout**: C-side `HcclTaskMetaData` expanded to 188B (taskType
    followed by `HcclTaskProfInfo` 3 new bytes, Sec. 4.4.1); `db_reader.py` still decodes at 185B
    (`TASK_META_LEN=185`, `<BQIQQI16s16s`) — the length gate `len(blob) >= 185` cannot block
    188B BLOBs; reading v3-collected db causes **fields to be entirely misaligned** (from commId onward). Needs synchronization to
    188B/`<BBBBQIQQI16s16s` and decoding of profInfo three fields (value ranges/names in Sec. 11.4).~~
    **Synchronized**: `db_reader.py` auto-identifies v3 (188B/`<BBBBQIQQI16s16s`,
    profInfo three fields into `HcclTaskMetaData`) and old v2 (185B) layouts by BLOB length, see Sec. 4.4.1 implementation status.
18. ~~**JSON task types do not consume profInfo** (remaining items narrowed): trace args
    `transport_type`/`link_type` already switched to profInfo-driven; CommProtocol migrated to new field
    `protocol` (Sec. 11.4.2/11.4.3, Sec. 11.1 approach a landed); **remaining item** is `task_type`
    field — currently still derived from TaskType enum + jettyId/direction conditions (Sec. 11.2); has **semantic misalignment** with collection-side
    profInfo hcomm op names (such as `Ub_Inline_Write` bidirectionally referring to different
    tasks, Sec. 11.4.4); switching to `profInfo.profTaskType`-driven would enable one-to-one correspondence with msprof timeline
    op names; target state mapping (including each value's correspondence with simulation TaskType) in Sec. 11.4.1.~~
    **Landed (all remaining items completed)**: `Task` adds `task_name` display name (`task_model.py`,
    `display_name` = `task_name or task_type.value`); `graph_builder._add_task`
    uniformly injects `prof_task_type_name(profInfo.profTaskType)`; trace event name/`cat`
    (`report.py export_trace`), `args.task_type` (`to_trace_args`), verbose
    detail and dump (DOT/JSON/HTML legend) all switched to op name. Internal `TaskType` retired to
    latency model classification (Sec. 11.4.5 #3); latency/pairing/charging zero changes. v2 old BLOB
    (profInfo=0xFF) displays `Unknown` (same granularity as link/transport); CCU microcode sub-tasks
    have no profInfo; fall back to internal name (Sec. 11.3). Misalignment elimination comparison in Sec. 11.4.4 implementation record.
19. ~~**WRITE_WITH_NOTIFY (taskType=7) decoding missing**: `db_reader.py`'s
    `HccLTaskMetaType` IntEnum only covers 0~6; `decode_op_task_meta` encountering 7 directly raises and discards
    (counted as warning).~~ **Supported (single-task mapping)**: `db_reader.py` decodes
    `WriteWithNotifyTask` (Sec. 4.4.1) and passes through as single task; `graph_builder` maps to
    `WRITE_WITH_NOTIFY` TaskType (trace display name `Write_With_Notify`, Sec. 5.2);
    **one-to-one correspondence** with db tasks; latency = port write direction bandwidth sharing + post-write notify
    fixed record latency (Sec. 7.3/7.4). **Split strategy not adopted** (checker/runner side's
    "split into [write, NOTIFY_RECORD] two sub-tasks" and this plugin's early decoding-boundary split both
    deprecated): splitting would fragment task granularity, making trace/dump/statistics unable to correspond one-to-one with db tasks.
    Collection-side profInfo=TASK_WRITE_(REDUCE_)WITH_NOTIFY (op names `Write_With_Notify` /
    `Reduce_With_Notify`; hcomm source has the latter misspelled as `Reuce_With_Notify`, see Sec. 11.4.5);
    both profInfo values map to the same `Write_With_Notify` task (isReduce only affects
    data_type display).
20. ~~**Same mission multiple dispatch SQEs merged into single context**: CCU expansion merges all SQEs of the same
    `(dieId, missionId, queue)` into one dispatch context (single instruction window
    [first instStartId, last), single SQE args consumption) — does not satisfy checker
    `IsContiguousCcuSqe`'s "contiguous-only merging" constraint (`task_meta_translator_v3.cc:213-231`).
    Typical victim scenario: instruction window wraparound re-dispatch (head SQE 13 Load SqeArgs + body SQE dispatched twice consecutively;
    second dispatch uses new args); under old implementation, the second dispatch: ① window start counted by head SQE; end JMP jumps over the entire data transfer instructions;
    ② SQE args already consumed by the first dispatch; subsequent Loads get 0 values
    (Address addresses all 0); ③ loop count/Event retains first-dispatch state — second dispatch's
    Ub/Memcpy/local post entirely lost; network data volume only 3/10 of checker.~~
    **Fixed (contiguous segment splitting)**: split into multiple dispatch contexts by `instStartId`/`instCnt` continuity
    (Sec. 6.1); each context with independent window replay (including Load SqeArgs header) + independent sequential
    args consumption (`GetSqe` semantics); register/Event bitmap shared across contexts (`AllRankParamRecorder`
    semantics); same-queue contexts serial by dispatch order; multi-segment task ids disambiguated with `_s{sequence}` suffix
    (`ccu_graph_builder.expand_ccu_graph`). After fix, aligned with checker single-pass graph stream by stream:
    record local/remote 54/54, wait local/remote 27/36, Ub 39/39, Memcpy 6/6,
    network+local data volume 1088MB (2333 validation environment ALLGATHER dev1).
    Same-root-cause **wait node emission deviation** also fixed: old implementation split by "one wait node per contributing post"
    (one wait instruction could produce N nodes; inconsistent with checker `GenWaitNode`'s "at most
    1 local + 1 remote node per instruction"; this environment showed local wait 48 vs 27); now aggregated into single node
    (mask = OR of all contributing post bits; first contributing post uses notify pairing; others through extra edges,
    Sec. 6.4); after fix, local wait 27/27 aligned.
21. **Merged loop folded pseudo-serialization residual gaps**: Sec. 6.5.1 signature chain resolution covers the merge path's
    mid-body wait/record (folded pseudo-serialization barriers), but two residuals remain: ① full-expansion fallback path
    (`mf > 0`) per-iteration wait/record is real 4KB granularity handshaking; its inter-iteration overlap is similarly
    serialized by the queue chain (large node count; not triggered by current samples; not addressed); ② per-iteration handshaking
    **rate** overhead not modeled — folding and resolution both count as single occurrence (if Event engine throughput is the bottleneck, a dedicated
    rate model is needed, such as ns/occurrence × 32768 occurrences level evaluation).

---

## 11. db Task Type and JSON Artifact Task Type Mapping Table

> This chapter is a type mapping quick reference, answering three questions:
> ① Which task types in the JSON artifacts do db database tasks (`opTask_P_<pid>` table taskType column) ultimately convert to (Sec. 11.1, 11.2);
> ② The correspondence between CCU mode microcode instructions and JSON task types (Sec. 11.3);
> ③ The correspondence between collection-side profiling reconciliation information (`HcclTaskProfInfo`) and JSON
> display types and the target state (Sec. 11.4). Implementation basis: `src/plugin/light_performance/
> graph_builder.py` (SQE mapping, `_map_tasks`/`_map_transfer`),
> `src/plugin/light_performance/ccu_graph_builder.py` (CCU instruction mapping, `transform`
> series), `src/plugin/light_performance/simulator/task_model.py` (TaskType enum).
> Same source as Sec. 5.2 (SQE mapping + latency model), Sec. 6.3 (instruction major class mapping); reorganized here by
> "db → JSON" correspondence with reverse index supplemented.
>
> **profInfo background**: from v3 collection onward, db task opTaskMeta BLOBs carry `HcclTaskProfInfo`
> sub-structure (Sec. 4.4.1; single-source defined in `include/sim_profiling_defs.h`; checker/runner pass through as
> opaque); the three fields `profTaskType`/`linkType`/`transportType` value ranges and
> names are aligned one-to-one with hcomm `TaskParamType`/`DfxLinkType`/`DfxTransportType` — i.e.,
> msprof timeline op names and L1 detail linkType/transportType fields. This chapter Sec. 11.1~11.3 describes
> **current TaskType implementation** (profInfo not yet consumed); Sec. 11.4 provides the complete three-field mapping and switchover
> target state (backlog in Sec. 10.3 #17~#18).

### 11.1 Task Type Carriers in JSON Artifacts

The current implementation has three types of JSON artifacts; task types in all are taken from `TaskType` enum values
(string form, `task_model.py:20-26`):

| JSON Artifact | Format | Task Type Field Location | Generation Module |
|--------------|--------|-------------------------|-------------------|
| `perf_profiling.json` (primary artifact, Sec. 8.1) | Chrome Trace Event | `traceEvents[].name` / `.cat` / `.args.task_type` | `simulator/report.py` (`export_trace`; event name is task display name: profInfo op name, Sec. 11.4) |
| `trace_comm<commId>_iter<opIter>.json` (optional per-operator trace) | Chrome Trace Event | Same as above | Same as above |
| Task graph dump `graph_comm<commId>_iter<opIter>_dev<dev>.json` (optional, `--dump-dir`) | `light_perf.graph_dump/2` | `nodes[].type` | `graph_dumper.py` |

> Dump JSON notify nodes carry `nodes[].mask` (Sec. 7.3 Task.mask) when mask is non-zero;
> HTML view tooltip synchronously displays; `perf_profiling.json`/trace task args also contain
> `mask` (Sec. 7.7).

Full set of task types appearing in JSON (7 types):

| TaskType Value (Presented in JSON) | Latency Model | Semantics |
|-----------------------------------|--------------|-----------|
| `Notify_Record` | Fixed latency (local additionally triggers Event) | Notify assertion (record / Event post) |
| `Notify_Wait` | Wait for Event / fixed latency | Notify wait (wait / Event wait) |
| `Reduce_Inline` | On-chip memory dynamic sharing | Local reduction (computation approximated as transfer, Sec. 5.7) |
| `Memcpy` | On-chip memory dynamic sharing | Local copy (on-chip memory, not network) |
| `Ub_Inline_Write` | Fixed latency (same as Notify_Record) | Notify inline write (old input path only) |
| `Ub_Write_Or_Read` | Port bandwidth sharing (write/read directions charged independently) | Network data send/receive (SQE network tasks in both directions uniformly mapped) |
| `Write_With_Notify` | Port write direction bandwidth sharing + post-write notify fixed latency | Network write + attached notify (one-to-one with db `WRITE_WITH_NOTIFY`, Sec. 5.2) |

> The `perf_profiling.json` structured format (meta/ops array) in Sec. 8.3 is a draft; the current primary artifact
> is actually in Chrome tracing format (Sec. 10.3 #10); regardless of format, task types are the above
> 7 TaskType values; this chapter's mapping tables apply uniformly.

**Type-related fields in trace args (current state and profInfo consumption)**: Task's `to_trace_args`
(`task_model.py`) outputs four type-related fields; all four fields have been switched to profInfo-driven
(Sec. 10.3 #18 all landed, Sec. 11.4):

| trace args Field | Current Source | Status |
|-----------------|---------------|--------|
| `task_type` | `profInfo.profTaskType` op name (`prof_task_type_name`, `graph_builder._add_task` uniformly filled as `Task.task_name`; `display_name` falls back to internal TaskType name when empty) | **Landed** (one-to-one correspondence with msprof timeline op name, Sec. 11.4.1; v2 old BLOB is `Unknown`) |
| `transport_type` | `profInfo.transportType` name (`ProfTransportTypeName`, Sec. 11.4.3, `graph_builder._add_task` uniformly filled) | **Landed** |
| `link_type` | `profInfo.linkType` name (`ProfLinkTypeName`, Sec. 11.4.2, `graph_builder._add_task` uniformly filled) — **note: this and `protocol` (CommProtocol) are two different concepts** | **Landed** (approach a: overwritten as DfxLinkType name) |
| `protocol` (new field) | Channel `CommProtocol` enum name (HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO, filled by `graph_builder._resolve_eids` / CCU `_channel_eids`) | **Landed** (CommProtocol migrated from `link_type`; `N/A` when no channel info) |

> `link_type` overwrite (approach a) has been implemented: semantics aligned with msprof L1 detail field names, avoiding same-name-different-meaning;
> existing CommProtocol information is not lost; moved to `protocol` field. CCU microcode sub-tasks are simulator
> semantic products with no profInfo (Sec. 11.3); their `link_type`/`transport_type` are `N/A`;
> `protocol` retains the channel CommProtocol name.

### 11.2 SQE Path: db taskType → JSON Task Type Mapping Table

Forward mapping (db task → JSON task display name). The "JSON Task Type" column is the trace/dump display
op name (already switched to `profInfo.profTaskType`-driven, Sec. 10.3 #18 landed); "Internal TaskType"
(latency model classification, determines charging/pairing) maintains Sec. 11.2 original conditions (jettyId/direction/recordDev);
correspondence with op names in Sec. 11.4.1 "Simulation TaskType" column:

| db taskType (opTask.taskType column) | Condition | JSON Task Type (op name) | Collection-side profInfo.profTaskType (op name) | Description |
|-------------------------------------|-----------|-------------------------|------------------------------------------------|-------------|
| `NOTIFY_WAIT` (0) | recordDev == this device | `Notify_Wait` | `TASK_NOTIFY_WAIT` → `Notify_Wait` ✓ | Local wait: participates in first-come-first-served pairing (Sec. 5.4); waits for Event |
| `NOTIFY_WAIT` (0) | recordDev != this device | `Notify_Wait` | `TASK_NOTIFY_WAIT` → `Notify_Wait` ✓ | Cross-rank wait: node retained, not paired; src≠dst → fixed `remote_wait_delay` |
| `NOTIFY_RECORD` (1) | waitDev == this device (SDMA/local notify) | `Notify_Record` | `TASK_NOTIFY_RECORD` → `Notify_Record` ✓ | Local record: participates in pairing; triggers Event |
| `NOTIFY_RECORD` (1) | waitDev != this device (SDMA/local notify) | `Notify_Record` | `TASK_NOTIFY_RECORD` → `Notify_Record` ✓ | Cross-rank post: node retained, not paired; src≠dst → fixed `remote_record_delay` |
| `NOTIFY_RECORD` (1) | UBDMA WQE `inlineEn==1` (notify inline write) | `Ub_Inline_Write` | `TASK_UB_INLINE_WRITE` → `Ub_Inline_Write` ✓ | Notify inline write (name returned, Sec. 11.4.4); internal TaskType still Notify_Record (fixed record latency) |
| `REDUCE` (2) | jettyId == INVALID | `Reduce_Inline` | `TASK_REDUCE_INLINE` → `Reduce_Inline` ✓ | Local reduction: on-chip memory dynamic sharing charging |
| `REDUCE` (2) | jettyId valid + write direction | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` → `Ub_Reduce_Inline` ✓ | Network reduction approximation (Sec. 5.7); internal TaskType=Ub_Inline_Write; port write direction charging |
| `REDUCE` (2) | jettyId valid + read direction | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` → `Ub_Reduce_Inline` ✓ | Same as above; internal TaskType=Ub_Write_Or_Read; read direction charging |
| `MEM_CPY` (3) | jettyId == INVALID (same device) | `Memcpy` | `TASK_SDMA` → `Memcpy` ✓ | Local copy: on-chip memory dynamic sharing charging; profInfo.linkType=ONCHIP |
| `MEM_CPY` (3) | jettyId == INVALID (cross-device SDMA SQE) | `Memcpy` | `TASK_SDMA` → `Memcpy` ✓ | Cross-chip SDMA transfer; also Memcpy; difference expressed by profInfo.linkType=PCIE / transportType=SDMA (Sec. 11.4) |
| `MEM_CPY` (3) | jettyId valid + write direction | `Ub_Write_Or_Read` | `TASK_UB` → `Ub_Write_Or_Read` ✓ | Network send (direction not encoded in type name; expressed by direction/is_write); internal TaskType=Ub_Inline_Write |
| `MEM_CPY` (3) | jettyId valid + read direction | `Ub_Write_Or_Read` | `TASK_UB` → `Ub_Write_Or_Read` ✓ | Network receive; internal TaskType=Ub_Write_Or_Read |
| `WRITE_WITH_NOTIFY` (7) | Always write direction (no direction inference, Sec. 5.5) | `Write_With_Notify` (single task, not split); `udfFlag==1` → `Reduce_With_Notify` | `TASK_WRITE_WITH_NOTIFY` → `Write_With_Notify`; `udfFlag==1` → `TASK_WRITE_REDUCE_WITH_NOTIFY` → `Reduce_With_Notify` (hcomm source has misspelling `Reuce_With_Notify`, Sec. 11.4.5) | Write+notify combined task (`WriteWithNotifyTask`, Sec. 4.4.1): port write direction bandwidth sharing + post-write notify fixed record latency; both profInfo values map to the same internal WRITE_WITH_NOTIFY (latency model identical; display name and isReduce distinguish) |
| `CCU_GRAPH` (4) | — | (Placeholder, not directly produced) | `TASK_CCU` → `Ccu` | Expands into CCU sub-graph; final artifacts in Sec. 11.3 |
| `AIV_GRAPH` (5) | — | None (skipped) | `TASK_AIV` → `AivKernel` | Warning + `skipped_aiv` count (not supported) |
| `SYNC_STREAM` (6) | — | None (skipped) | `UNKNOWN` → `Unknown` | Host-side stream sync point, not a device-side task (Sec. 5.6); hcomm profiling has no corresponding type |

> Cross-rank determination (waitDev/recordDev) in Sec. 4.6; network direction determination (write/read) in Sec. 5.5.

Reverse index (JSON task display name ← all sources, including CCU path and profInfo dimension; SQE path
display name = profInfo op name; CCU path display name = internal TaskType name, Sec. 11.3):

| JSON Task Display Name | SQE Source (db taskType) | Collection-side profInfo Source (op name) | CCU Source (microcode instructions, Sec. 11.3) |
|----------------------|-------------------------|------------------------------------------|------------------------------------------------|
| `Notify_Record` | `NOTIFY_RECORD` (1), local/cross-rank (SDMA/local notify) | `Notify_Record` (TASK_NOTIFY_RECORD ✓) | SetCKE/SetCKBit local post; TRANS/REDUCE/LOAD-class instruction embedded `setCKE` field (A6 LOAD-class also carries); cross-rank assertion/notification (A5 Sync family setRmtCKE; A6 TransMem notify/0x1A atomic, SyncWtX notify, SyncAtx — remote approximation); sub-graph attachment launch-record / completion-record |
| `Notify_Wait` | `NOTIFY_WAIT` (0), local/cross-rank | `Notify_Wait` ✓ | Various instruction embedded `waitCKE` field; ClearCKE; A6 independent Wait instruction (Variable value comparison; unsatisfied → remote wait); sub-graph attachment completion-wait / launch-wait |
| `Reduce_Inline` | `REDUCE` (2), jettyId invalid | `Reduce_Inline` ✓ | REDUCE-class instructions (msId list all local) |
| `Memcpy` | `MEM_CPY` (3), jettyId invalid (including cross-chip SDMA) | `Memcpy` ✓ | TRANS local transfer instructions (A5 4 codes: 0x0/0x2/0x5/0xA; A6 4 codes: 0x0/0x2/0x5/0x6) |
| `Ub_Inline_Write` | `NOTIFY_RECORD` (1) + `inlineEn==1` (notify inline write) | `Ub_Inline_Write` (TASK_UB_INLINE_WRITE ✓, name returned) | — (CCU network transfers uniformly mapped as `Ub_Write_Or_Read`, Sec. 11.3.1) |
| `Ub_Write_Or_Read` | `MEM_CPY` (3) / `REDUCE` (2), jettyId valid (write/read direction) | `Ub_Write_Or_Read` (TASK_UB ✓, direction expressed by direction/is_write) | TRANS network transfer instructions (A5 write role 3 codes: 0x3/0x7/0x9, read role 4 codes: 0x1/0x4/0x6/0x8; A6 unified TransMem 0x10: dmaOpCode 3/5=write, 6=read, including udfEnable network reduction approximation); REDUCE class with remote CcuBuffer |
| `Ub_Reduce_Inline` | `REDUCE` (2), jettyId valid (network reduction) | `Ub_Reduce_Inline` (TASK_UB_REDUCE_INLINE ✓) | — |
| `Write_With_Notify` | `WRITE_WITH_NOTIFY` (7), isReduce=0 | `Write_With_Notify` (TASK_WRITE_WITH_NOTIFY ✓) | — |
| `Reduce_With_Notify` | `WRITE_WITH_NOTIFY` (7), udfFlag==1 | `Reduce_With_Notify` (TASK_WRITE_REDUCE_WITH_NOTIFY ✓) | — |

### 11.3 CCU Path: Microcode Instructions → JSON Task Type Mapping

The `CCU_GRAPH` (4) SQE in the db is only a placeholder (`CcuPlaceholder`, Sec. 6.1); the final
JSON tasks are produced by microcode interpretation from the `ccuInstrRes` instruction space (Chapter 6). **Both A5 (V1,
Ascend950, devType=6) and A6 (V2, Ascend960, devType=8) instruction sets are implemented**
(selected by `opDetails.devType`; other devTypes fall back to A5): A5 full table in Sec. 11.3.1;
A6 full table in Sec. 11.3.3.

> profInfo correspondence: CCU placeholder SQE's `HcclTaskProfInfo` is entirely
> `TASK_CCU`→`Ccu` (linkType/transportType are UNKNOWN; microcode has no link dimension). Expanded
> microcode sub-tasks (`Memcpy`/`Ub_Write_Or_Read`/`Notify_*` etc.) are
> simulator semantic products with **no corresponding ProfTaskParamType**; display name falls back to internal TaskType name
> (established design choice during Sec. 10.3 #18 implementation, Sec. 11.4.4); msprof side CCU tasks are expressed through
> microcode-level detail (`CcuProfilingInfo`: CCU_TASK/WAITCKE/LOOPGROUP three types, reported with TASK_CCU)
> for equivalent granularity information; reconciliation uses the entire SQE's `Ccu` as the alignment anchor.

#### 11.3.1 A5 Instruction Full Table (Instruction → JSON Task Type)

Microcode instruction = 2B header (type:4bit major class + code:11bit instruction code, Sec. 6.2) + 30B payload:

| Instruction Major Class (type) | Instruction (code) | JSON Task Type | Description |
|-------------------------------|-------------------|---------------|-------------|
| LOAD (0x0) | LoadSqeArgsToGSA (0x0), LoadSqeArgsToXn (0x1), LoadImdToGSA (0x2), LoadImdToXn (0x3), LoadGsaXn (0x4), LoadGsaGsa (0x5), LoadXx (0x6) | **None** (no nodes generated) | Only reads/writes register model (xn/gsa) for subsequent instructions to retrieve length, loop count etc. |
| CTRL (0x1) | SetCKE (0x2) | `Notify_Record` | Local Event post (ckeId as notify_id, ckeMask as mask, src=dst=local); embedded `waitCKE` field can additionally produce `Notify_Wait` |
| CTRL (0x1) | ClearCKE (0x4) | `Notify_Wait` | wait semantics (embedded `waitCKE` field); clear action itself produces no node |
| CTRL (0x1) | Loop (0x0) | **None** | Must be triggered by LoopGroup; standalone occurrence is anomalous; warning and skip |
| CTRL (0x1) | LoopGroup (0x1) | (No direct nodes) | Loop control: merge/expand loop body (Sec. 6.5); body instructions mapped per this table; data task size × total iteration count under merge path; emitted tasks carry loop_cnt (loop count after expansion) / loop_iter_cnt (single loop self-iteration count); total iterations = product of the two fields |
| CTRL (0x1) | Jmp (0x5) | **None** | Conditional jump: when `Xn[conditionXnId] != expectData`, pc jumps to instruction pointed by `Xn[dstInstrXnId]` (checker `TransformJmpInstrA5` semantics; target limited to current instruction window; out-of-bounds warning pass-through); within loop body, pass-through (merge prerequisite validation already rejected loop bodies containing JMP); step budget defends against backward jump infinite loops |
| TRANS (0x2) | LocMemToLocMs (0x0), LocMsToLocMem (0x2), LocMsToLocMs (0x5), LocMemToLocMem (0xA) | `Memcpy` (src/dst_buf_type annotated mem/ms) | Local transfer (LOCAL_COPY role, Sec. 6.3.2); on-chip memory dynamic sharing |
| TRANS (0x2) | LocMsToRmtMem (0x3), LocMemToRmtMem (0x9), LocMsToRmtMs (0x7), RmtMemToLocMs (0x1), RmtMsToLocMem (0x4), RmtMsToLocMs (0x6), RmtMemToLocMem (0x8) | `Ub_Write_Or_Read` (is_write distinguishes direction; src/dst_buf_type annotated mem/ms) | Cross-die/card transfer (write/read roles unified type, Sec. 6.3.2); peer rank resolved through channel reid → eid→rank index (Sec. 4.8/6.7) |
| TRANS (0x2) | SyncCKE (0xB), SyncGSA (0xC), SyncXN (0xD) | Local post portion → `Notify_Record`; remote post portion → `Notify_Record` (remote, not paired) | Remote assertion fields (SyncCKE's `rmtCKEId/locCKEMask`, SyncGSA/SyncXN's `setRmtCKEId/setRmtCKEMask`) generate remote record nodes (src=local, dst=INVALID → fixed `remote_record_delay`; not registered in local accounting); counted in `remote_posts` (Sec. 6.6); SyncGSA/SyncXN carrying a second remote post similarly generate; embedded `waitCKE` can produce `Notify_Wait` |
| REDUCE (0x3) | Add (0x0), Max (0x1), Min (0x2) | `Reduce_Inline` | Reduction computation approximated as transfer (Sec. 5.7); CcuBuffer always rank-local buffer (msId high bits are addressing encoding only, Sec. 6.3.2); local on-chip memory charging |
| (Sub-graph attachment) | — (synthesized during graph building phase, not microcode) | None (no task nodes produced) | Attachment dependencies all expressed through extra edges (Sec. 7.4.1): entry edges SQE predecessor→skeleton head/pre-skeleton unit head; exit edges mission completion set→SQE successor; task graph does not spontaneously generate any nodes |

#### 11.3.2 Embedded Synchronization Fields (Cross-Instruction Universal)

TRANS / REDUCE / SetCKE / Sync-class instruction payloads commonly carry synchronization fields that **additionally produce**
JSON tasks alongside the main semantics:

| Payload Field | JSON Task Type | Behavior |
|--------------|---------------|----------|
| `waitCKEId` / `waitCKEMask` | `Notify_Wait` | When bits are satisfied, pairs with local post accounting (aggregated into single node: mask=OR of all contributing post consumed bits; first post add_match, others extra edges, Sec. 10.3 #20); when bits insufficient: top-level instruction blocks this queue (stall detection discovers it, generates remote wait node and replays rest of instruction, `remote_waits`); within loop body, cannot block; directly generates remote wait node and continues |
| `setCKEId` / `setCKEMask` | `Notify_Record` | Local Event post; registers in seen_post accounting (Sec. 6.4) |
| `clearType` bits / SetCKE clear flag | None (register clear only) | Clears bits after wait; produces no node |

> Therefore, one TRANS instruction can produce 1 data transfer task (`Memcpy`/`Ub_Write_Or_Read`),
> and additionally produce several `Notify_Wait` (by bit pairing,
> 0..n) and 1 `Notify_Record` based on embedded fields. Under merged loops (Sec. 6.5), each slot similarly retains
> wait/set tasks; only data tasks are folded to size × total iteration count.
>
> **A6 differences**: embedded `waitCKE` exists only in SetCKBit/ClearCKE (CTRL class); local
> TRANS/REDUCE and TransMem/Sync classes carry only tail `setCKEId/setCKEMask` (@26/28,
> additionally producing `Notify_Record`); wait semantics are handled by independent Wait instructions (Variable value comparison).

#### 11.3.3 A6 (Ascend960) Instruction Full Table (Instruction → JSON Task Type)

devType=960 (`DEV_TYPE_960`) corresponds to V2/A6 instruction set, **implemented** (ported from checker
`ccu_task_transform_instruct_a6_v3.cc`, interpreter `_MissionContextA6`). Microcode instruction
= 2B header (isomorphic with A5) + 30B V2 payload (`ccu_microcode_v1.h` CcuV2 struct,
pack(1)); register numbers generally through `GetXnId` (bit15 addressing mode, Sec. 6.4).

| Instruction Major Class (type) | Instruction (code) | JSON Task Type | Description |
|-------------------------------|-------------------|---------------|-------------|
| LOAD (0x0) | LoadSqeArgsToX (0x1), LoadImdToX (0x2), LoadX (0x6), StoreX (0x7), ClearX (0x8), Nop (0x9), arithmetic-logic Add (0xD)/Sub (0xE)/Mul (0xF)/And (0x10)/Or (0x11)/Not (0x12)/Xor (0x13)/Shl (0x14)/Shr (0x15)/Popcnt (0x16) | Data plane **none**; `setCKEMask != 0` additionally produces `Notify_Record` | Unified Variable register read/write (no Address): SQE args sequential consumption (Sec. 6.1), immediate loading, indirect read/write (LoadX/StoreX, oMode 0=immediate offset/1=register offset), range clear (ClearX), 64-bit wraparound arithmetic-logic (Mul takes low 32-bit product, shift amount 0x3F truncated, Shl/Shr support logical/arithmetic/circular three types); all except Nop carry tail setCKE field (checker `ProcessSetMask`, non-zero mask → local post node) |
| LOAD (0x0) | Load (0xA), Store (0xB) | Data plane **none**; setCKE additionally produces `Notify_Record` | On-chip memory ↔ Variable data plane: on-chip memory content not modeled (checker reads/writes data through AllRankParamRecorder); registers not backfilled; warning logged; only synchronization semantics retained |
| CTRL (0x1) | SetCKBit (0x2) | `Notify_Record` (+ embedded `waitCKE` additionally produces `Notify_Wait`) | wait → local post → clearType==1 clears wait bits (checker only supports clearType=1; others warning skip clear) |
| CTRL (0x1) | ClearCKE (0x4) | `Notify_Wait` (embedded `waitCKE`) | wait then follows checker `ClearWaitMask` semantics to clear consumed wait bits; `clearMask != 0` is checker-unsupported scenario (warning + tolerant target bit clear) |
| CTRL (0x1) | Loop (0x0) | **None** | Must be triggered by LoopGroup; standalone occurrence warning skip. `mode==1` conditional Loop: expands only when `wishCKEBit == Xp.loopCtxId + 1024` matches (Sec. 6.5) |
| CTRL (0x1) | LoopGroup (0x1) | (No direct nodes) | Loop control: parameters from Variable/Xm/Xp three registers (loopInsCnt:10/expandOffset:9/expandCnt:9 etc., Sec. 6.5); merge/expand same as A5 (merge validation includes A6 additional rules: wait Event ID non-reuse, cross-expansion iteration count/length/channel consistency); illegal config (loopInsCnt>512 etc.) warning skip |
| CTRL (0x1) | Jmp (0x5) | **None** | Conditional jump: conditionType 0~5 (==/!=/>/>=/</<=), ≥6 unconditional; jumpMode 0=relative pc (mod 0x10000)/1=absolute instruction number; target takes `Xn[relTarInstrXnId]`; limited to current instruction window; out-of-bounds warning pass-through; within loop body, pass-through; step budget defends against infinite loops |
| CTRL (0x1) | Wait (0x7) | Remote `Notify_Wait` when unsatisfied | Variable value comparison spin (conditionType same as Jmp; default is equal): satisfied → pass through with no node; unsatisfied = depends on peer remote register write → top-level stall resolution / within loop body directly generates remote wait node (fixed `remote_wait_delay`, task id tagged `xwt`, counted in `remote_waits`) |
| CTRL (0x1) | Fence (0x8) | **None** | Checker unsupported (error abort); here warning pass-through (queue serial model naturally guarantees order) |
| TRANS (0x2) | TransLocMemToLocMS (0x0), TransLocMSToLocMem (0x2), TransLocMSToLocMS (0x5), TransLocMemToLocMem (0x6) | `Memcpy` (src/dst_buf_type annotated mem/ms) | Local transfer (length `Variable[GetXnId(xlId)]`); no embedded waitCKE; tail setCKE additionally produces `Notify_Record` |
| TRANS (0x2) | TransMem (0x10) | `Ub_Write_Or_Read` (dmaOpCode 3/5=write, 6=read); notify/atomic → remote `Notify_Record` | Unified network transfer: channel = `Variable[GetXnId(xcId)]` register value → peer rank/eid resolution same as A5 (Sec. 4.8/6.7); when `msIdMode==1`, write source / read destination is local CcuBuffer (buf_type=ms); dmaOpCode 5 post-write notify and 0x1A remote atomic count (no data plane) approximated as remote record (mask=notification value low 16 bits); `udfEnable` network reduction charged as ordinary UB transfer (Sec. 5.7 approximation); 0x19 and other dmaOpCode / `udfType != 0` warning skip data plane (setCKE post still executes) |
| TRANS (0x2) | SyncXnWtx (0xD), SyncAtx (0xE) | Remote `Notify_Record` (approximation); SyncAtx additionally has setCKE → local `Notify_Record` | Cross-rank register sync: SyncWtX's `notifyValid==1` notify and SyncAtx's atomic write approximated as remote record (remote target type unresolvable in single rank, Sec. 6.6); primary value sync is remote register write, no node built (same granularity as A5 SyncXn); SyncWtX tail setCKE field not consumed by checker (local does not post); unresolvable channel warning skip (`eid_unresolved`) |
| REDUCE (0x3) | ReduceAdd (0x0), ReduceMax (0x1), ReduceMin (0x2) | `Reduce_Inline` | Local CcuBuffer batch reduction (msId[8] list + `Variable[XnIdLength]` length); no embedded waitCKE; tail setCKE additionally produces `Notify_Record`; CcuBuffer always local on-chip memory charging (Sec. 6.3.2) |
| (Sub-graph attachment) | — (synthesized during graph building phase, not microcode) | None (no task nodes produced) | Same as A5 (Sec. 7.4.1 extra edge attachment) |

> A6 established deviations from checker (on-chip memory data plane, remote target type resolution, checker error scenario
> tolerant downgrading etc.) summarized in `_MissionContextA6` class docstring and Sec. 6.7 comparison table;
> loop merge/pipeline concurrency recovery (Sec. 6.5/6.5.1), queue parallelization (Sec. 7.4.1), SQE merging
> and stall resolution (Sec. 6.1/6.6) and other framework mechanisms are shared across both generations.

### 11.4 HcclTaskProfInfo Three-Field Mapping (profInfo → JSON Display)

From v3 collection onward, db tasks carry `HcclTaskProfInfo` (Sec. 4.4.1; single-source defined in
`include/sim_profiling_defs.h`). This section provides the complete "value → enum → JSON display string"
mapping for all three fields and each value's correspondence with simulation TaskType (latency model classification, Sec. 7.3);
C++ side naming functions (`ProfTaskParamTypeName`/`ProfLinkTypeName`/
`ProfTransportTypeName`) are the sole authoritative source for display strings.

> **Implementation status**: `task_type`/`transport_type`/`link_type`/`protocol` four fields all landed per
> this section's mapping (Python-side display name mappings in `db_reader.py`'s `PROF_*_NAMES`, character-by-character
> consistent with C++ naming functions, Sec. 11.4.5; `graph_builder._add_task` uniformly filled — `task_type`
> through `Task.task_name`/`display_name`, Sec. 10.3 #18 completed).

#### 11.4.1 profTaskType (ProfTaskParamType, 0~22 → msprof timeline op name)

This table supplements each value's correspondence with simulation TaskType (`task_model.py`, latency model classification) — previously
db tasks and simulation task types were matched through db taskType column (Sec. 11.2); after profInfo consumption, this table's
`profTaskType → simulation TaskType` serves as the second matching dimension (determination conditions unchanged; same db taskType
routed to different TaskTypes by jettyId/direction noted within the row):

| Value | Enum | JSON Display String (op name) | Simulation TaskType (Latency Model Classification) | light_perf Occurrence Scenario |
|-------|------|------------------------------|---------------------------------------------------|-------------------------------|
| 0 | TASK_SDMA | `Memcpy` | `Memcpy` (jettyId invalid, on-chip memory charging) / `Ub_Inline_Write`·`Ub_Write_Or_Read` (jettyId valid, by direction, port charging) | Local copy, cross-chip SDMA SQE (linkType distinguishes, Sec. 11.4.2) ✓ |
| 1 | TASK_RDMA | `RDMASend` | `Memcpy` (jettyId invalid, on-chip memory charging approximation) | DPU host ibv ordinary read/write (`RDMASend`, same granularity as hcomm UbMemTransport::Write/Read) ✓ |
| 2 | TASK_REDUCE_INLINE | `Reduce_Inline` | `Reduce_Inline` (jettyId invalid, on-chip memory charging) / `Ub_Inline_Write`·`Ub_Write_Or_Read` (network approximation, Sec. 5.7, port charging) | Local reduction ✓ |
| 3 | TASK_REDUCE_TBE | `Reduce_TBE` | — (does not appear) | 910 generation reserved; does not appear |
| 4 | TASK_NOTIFY_RECORD | `Notify_Record` | `Notify_Record` | SDMA/local notify record ✓ |
| 5 | TASK_NOTIFY_WAIT | `Notify_Wait` | `Notify_Wait` | Notify wait ✓ |
| 6/7 | TASK_SEND_NOTIFY / TASK_SEND_PAYLOAD | `Send_Notify` / `Send_Payload` | — (does not appear) | hcomm reserved; does not appear |
| 8 | TASK_WRITE_WITH_NOTIFY | `Write_With_Notify` | `Write_With_Notify` | WRITE_WITH_NOTIFY (7) whole task (single-task mapping, Sec. 10.3 #19 supported) ✓ |
| 9 | TASK_WRITE_REDUCE_WITH_NOTIFY | `Reduce_With_Notify` (tool-side corrected spelling; hcomm source/msprof has misspelling `Reuce_With_Notify`, Sec. 11.4.5) | `Write_With_Notify` (isReduce=1 reduction write payload; data_type distinguishes display) | WRITE_WITH_NOTIFY (7) isReduce=1 |
| 10 | TASK_CCU | `Ccu` | (CCU placeholder whole SQE; expands to Chapter 6 sub-graph, Sec. 11.3) | CCU placeholder whole SQE ✓ |
| 11/12 | TASK_AICPU_KERNEL / TASK_AICPU_REDUCE | `AicpuKernel` / `Aicpu_Reduce` | — (not consumed) | AICPU kernel dispatch (through device process; currently no collection point/not consumed) |
| 13 | TASK_AIV | `AivKernel` | — (skipped) | AIV kernel task (skipped) |
| 14 | TASK_UB_INLINE_WRITE | `Ub_Inline_Write` | `Notify_Record` (notify fixed record latency; display name returned to op name ✓) | UBDMA `inlineEn==1` notify inline write |
| 15 | TASK_UB_REDUCE_INLINE | `Ub_Reduce_Inline` | `Ub_Inline_Write` (write) / `Ub_Write_Or_Read` (read) (network reduction approximation, port charging) | UBDMA `udfFlag==1` network reduction |
| 16 | TASK_UB | `Ub_Write_Or_Read` | `Ub_Inline_Write` (write direction) / `Ub_Write_Or_Read` (read direction) (direction expressed by direction/is_write) | UBDMA ordinary read/write WQE (both directions same value) |
| 17~22 | TASK_DPU_KERNEL / THREAD_FENCE / CHANNEL_FENCE / INLINE_WRITE / NOTIFY_WAIT / WRITE_WITH_NOTIFY | `DpuKernel` / `Dpu_ThreadFence` / `Dpu_ChannelFence` / `Dpu_Notify_Record` / `Dpu_Notify_Wait` / `Dpu_Write_With_Notify` | `Dpu_Notify_Record`→`Notify_Record`, `Dpu_Notify_Wait`→`Notify_Wait` (cross-rank fixed latency), `Dpu_Write_With_Notify`→`Write_With_Notify` (single-task port charging, Sec. 11.4.6); `DpuKernel`/Fence class current collection path does not produce | Host CPU RoCE (ibv_dpu_stub) tasks ✓ (Sec. 11.4.6) |
| 0xFF | UNKNOWN | `Unknown` | — (SYNC_STREAM etc. already skipped; no node produced) | SYNC_STREAM, level1 channel API tasks, v2 old BLOB and other untagged sources |

> `task_type` has been switched to op name (Sec. 10.3 #18 completed); this table's "Simulation TaskType" column is now
> an internal latency model classification index (Sec. 11.4.5 #3); ✓ rows' op names match TaskType values.
> After SQE batch merge (Sec. 5.1.1), merged task op name and link/transport take the **last** data task's profInfo within the batch
> (description follows last task; length takes full-batch cumulative sum); v3 and earlier BLOBs without
> sqeGroupId are not merged; displayed per task.

#### 11.4.2 linkType (ProfLinkType, 0~9 → hcomm DfxLinkType Enum Name)

| Value | Enum | JSON Display String | light_perf Occurrence Scenario |
|-------|------|-------------------|-------------------------------|
| 0 | LINK_ONCHIP | `ONCHIP` | Local Memcpy/Reduce_Inline/Notify_* |
| 1 | LINK_HCCS | `HCCS` | Reserved (current collection path does not produce) |
| 2 | LINK_PCIE | `PCIE` | Cross-chip SDMA SQE (same-device determined as local ONCHIP, Sec. 4.4.1 collection rules) |
| 3 | LINK_ROCE | `ROCE` | Host CPU RoCE (DPU, ibv_dpu_stub) tasks ✓ (Sec. 11.4.6) |
| 4/5/6 | LINK_SIO / HCCS_SW / STANDARD_ROCE | `SIO` / `HCCS_SW` / `STANDARD_ROCE` | Reserved |
| 7 | LINK_UB | `UB` | Network tasks (UBDMA WQE all classes) |
| 8 | LINK_UBoE | `UBoE` | UBoE channel (collection side does not yet distinguish UB/UBoE; does not yet appear) |
| 9 | LINK_RESERVED | `RESERVED` | Reserved |
| 0xFF | UNKNOWN | `Unknown` | CCU/AIV/SYNC_STREAM, level1 channel API tasks |

> **Implementation status**: trace args `link_type` field filled per this table (`prof_link_type_name`,
> `graph_builder._add_task` unified injection, Sec. 11.1 approach a); this table and the original `link_type`'s
> `CommProtocol` channel protocol (HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO) are **two different
> concepts**: CommProtocol describes the channel establishment protocol; DfxLinkType describes the task execution link (profiling
> dimension). CommProtocol name has been migrated to the `protocol` field (`N/A` when no channel info).

#### 11.4.3 transportType (ProfTransportType, 0~4 → msprof L1 Detail transportType Value Range)

| Value | Enum | JSON Display String | light_perf Occurrence Scenario |
|-------|------|-------------------|-------------------------------|
| 0 | TRANSPORT_SDMA | `SDMA` | Cross-chip SDMA SQE |
| 1 | TRANSPORT_RDMA | `RDMA` | Reserved (910 generation RDMA semantics) |
| 2 | TRANSPORT_LOCAL | `LOCAL` | Local tasks (Memcpy/Reduce_Inline/Notify_*) |
| 3 | TRANSPORT_UB | `UB` | Network tasks (**latency model anchor**: port bandwidth sharing charging, Sec. 5.5) |
| 4 | TRANSPORT_ROCE | `ROCE` | Host CPU RoCE (DPU, ibv_dpu_stub) tasks ✓ (Sec. 11.4.6) |
| 0xFF | UNKNOWN | `Unknown` | Untagged tasks |

> **Implementation status**: trace args `transport_type` field filled per this table
> (`prof_transport_type_name`, `graph_builder._add_task` unified injection).
> Same-named `Memcpy` tasks' local/cross-chip difference is expressed by linkType+transportType:
> ONCHIP+LOCAL (local) vs PCIE+SDMA (cross-chip) — trace args can now losslessly distinguish these two
> previously indistinguishable task types in JSON.

#### 11.4.4 Ub_Inline_Write Bidirectional Semantic Misalignment and Switch Implementation Record

**Historical problem**: before the switch, JSON's `Ub_Inline_Write` (network write direction send) and hcomm
profiling's `Ub_Inline_Write` (TASK_UB_INLINE_WRITE, notify inline write) **referred to different
tasks**, and the misalignment was bidirectional:

| Task (Collection Semantics) | db taskType | ProfTaskParamType (op name) | Pre-Switch JSON Type |
|----------------------------|-------------|---------------------------|---------------------|
| UBDMA ordinary write WQE (network send, write direction) | MEM_CPY (3) | `TASK_UB` → `Ub_Write_Or_Read` | `Ub_Inline_Write` ← name misalignment |
| UBDMA network reduction (write/read direction) | REDUCE (2) | `TASK_UB_REDUCE_INLINE` → `Ub_Reduce_Inline` | `Ub_Inline_Write` / `Ub_Write_Or_Read` ← name misalignment |
| UBDMA inline write notify (`inlineEn==1`) | NOTIFY_RECORD (1) | `TASK_UB_INLINE_WRITE` → `Ub_Inline_Write` | `Notify_Record` ← name misalignment |

**Implemented (Sec. 10.3 #18 completed)** — trace/dump display names switched to `profInfo.profTaskType`
op names (`Task.task_name`, `graph_builder._add_task` unified injection); all three misalignments eliminated:
1. Network data tasks (TASK_UB) write/read uniformly display `Ub_Write_Or_Read`; **direction no longer encoded
   in the type name**; expressed by trace args existing `direction` (write/read) and `is_write` fields —
   latency model unchanged; still charged by port write/read direction bandwidth sharing separately (Sec. 5.5).
2. `Ub_Inline_Write` name **returned** to notify inline write tasks (db taskType=
   NOTIFY_RECORD + inlineEn==1), one-to-one correspondence with msprof timeline.
3. Network reduction displays `Ub_Reduce_Inline` (TASK_UB_REDUCE_INLINE); reduction semantics preserved as distinguishable.
4. `Notify_Record`/`Notify_Wait`/`Reduce_Inline`/`Memcpy` four types were already consistent with op names;
   switch produces no change (`Memcpy` corresponds to TASK_SDMA).

**Implementation trade-offs (established decisions)**:
- v2 old BLOBs (no profInfo, profTaskType=0xFF) display `Unknown` — same granularity as
  `link_type`/`transport_type` v2 behavior (strict semantics; no internal name fallback).
- CCU microcode sub-tasks (Sec. 11.3) have no profInfo; display name falls back to internal TaskType name — simulator
  semantic products have no reconciliation requirement; status quo maintained (network transfers unified as `Ub_Write_Or_Read`; direction expressed by
  `direction`/`is_write`, Sec. 11.3.1).
- Internal `TaskType` enum retired to latency model classification (Sec. 11.4.5 #3); simulation/pairing/charging zero changes.

#### 11.4.5 String Consistency Convention (Reconciliation Basis)

1. **Python-side display strings must be character-by-character consistent with C++ side three naming function outputs**
   (`sim_profiling_defs.h` single source), including hcomm source's existing spellings
   (`Ub_Write_Or_Read`, `Dpu_Notify_Record` etc.) — this is the basis for field-by-field reconciliation with msprof timeline /
   L1 details.
   **Sole exception**: `TASK_WRITE_REDUCE_WITH_NOTIFY` op name unified on the tool side as the corrected spelling
   `Reduce_With_Notify` (hcomm source has misspelling `Reuce_With_Notify` in both locations:
   `profiling_common.h:42` and `dfx_profiling_handler_lite.cc:67`; msprof raw
   data displays the misspelled version); cross-system reconciliation encountering this string requires
   `Reuce_With_Notify ↔ Reduce_With_Notify` difference mapping.
2. trace args field-level switch (**all landed**, Sec. 10.3 #18 completed):
   `transport_type` ← transportType name, `link_type` ← linkType name
   (`graph_builder._add_task` unified injection; value range tables in Sec. 11.4.2/11.4.3); CommProtocol
   migrated to new field `protocol` (approach a, Sec. 11.1); `task_type` ← profTaskType op name
   (`Task.task_name`/`display_name`, Sec. 11.4.4 implementation record).
3. `TaskType` enum retired to internal latency model classification (`task_name` takes over display duty);
   trace event name (`traceEvents[].name`) and `args.task_type` both switched to op name;
   CCU microcode sub-tasks/CLI input have no profInfo; `display_name` falls back to internal name.

#### 11.4.6 DPU (Host CPU RoCE) Task Collection and Merging

Host DPU transfers (hcomm `HostCpuRoceChannel`, pod networking cross-node path) ibv hijack tasks
(`ibv_dpu_stub.cc`) previously lacked profInfo tagging (displayed `Unknown`), WriteWithNotify split into two
tasks, and peer resolution looped back to local. This section is the alignment implementation record for this path.

**profInfo tagging** (aligned with hcomm `host_cpu_roce_channel.cc` DFX granularity; link/transport
always ROCE/ROCE):

| ibv Operation | db taskType | profTaskType (op name) |
|--------------|-------------|----------------------|
| Pure imm notify (len=0 WRITE_WITH_IMM / hybrid) | NOTIFY_RECORD | `TASK_DPU_INLINE_WRITE` → `Dpu_Notify_Record` |
| Write+imm (len>0 WRITE_WITH_IMM) | **WRITE_WITH_NOTIFY (7) single task** | `TASK_DPU_WRITE_WITH_NOTIFY` → `Dpu_Write_With_Notify` |
| Pure RDMA write/read (no imm) | MEM_CPY | `TASK_RDMA` → `RDMASend` (aligned with hcomm `UbMemTransport::Write/Read` precedent) |
| poll_cq pairing | NOTIFY_WAIT | `TASK_DPU_NOTIFY_WAIT` → `Dpu_Notify_Wait` |

**WriteWithNotify merged single task**: the entire operation produces only one taskType=7 task
(`WriteWithNotifyTaskData`: src/dst rank, offsets, len, notifyId, protocol),
aligned with hcomm reporting only one `Dpu_Write_With_Notify` per operation; `DpuPendingNotify`
pairing mechanism unchanged. Simulation side `_map_write_with_notify` uses port write direction bandwidth sharing
(local RoCE eid → ranktable `d2h` port; when `d2h` port bandwidth is not configured, falls back to
`default_port_bandwidth`).

**Real peer resolution (phantom echo problem, socket timing binding)**: estimate mode socket layer constructs
phantom peers (local Send mirrored back as Recv, `hccp_ra_socket_stub.cc`); QP handshake exchanges back
"remote gid" that is actually the local self; `RaTypicalQpModify`'s gid→IP resolution returns to local. Fix
(zero-byte read, relying only on API call timing):
- `RaGetSockets` resolves real peer EndPoint from `SocketInfoT.remoteIp` (hcomm link configuration; both client/server roles
  carry it) and stores in FdInfo (peerDeviceId/peerEid/localEid);
- `RaGetQpAttr` anchor (hccp_stub, executed only within estimate branch) records QP pending binding
  (`DpuRecordPendingBindQp`, `bindAttempted` prevents Modify repeated attr lookup from re-binding);
- Next `RaSocketSend` in the same thread (i.e., hcomm ExchangeData's attr pack→Send timing) through
  `BindDpuPendingQps` binds the QP to that socket's real peer: `DpuQpInfo` adds
  `peerRankId`/`lcEid` (peer rank resolved from FdInfo's peerEid through `RankEidMap`; estimate
  communicator table has only the local rank; lcEid for WRITE_WITH_NOTIFY port charging).
`RaTypicalQpModify` maintains `remoteDeviceId` loopback plane semantics
(`DpuPendingNotify` pairing depends on this; under estimate = local); task metadata uniformly takes binding fields.

**NotifyWait precise synthesis (hcomm primitive interception)**: correct semantics are "only pending notifications inserted by the peer post
generate NotifyWait for me"; under phantom, the peer does not run and this notification has no real source; furthermore, hcomm
`NotifyWait` performs imm hard verification on poll results (`wc.imm_data == dpuNotifyId`; mismatch immediately
errors). Implementation (`hcomm_api_stub.cc` intercepts 5 hcomm northbound primitives; real function pointer static
one-time dlsym resolution; **normal mode first line short-circuits as pure forwarding; zero overhead**):
- Post-class primitives (`HcommChannelNotifyRecordOnThread` / `HcommWriteWithNotify(±Nbi)
  OnThread`; input carries `remoteNotifyIdx`) → `InsertDpuPendingNotifyEstimate`
  records `channelSlots[channel][idx] = (imm, notifyId)` slot mapping — under echo,
  `remoteDpuNotifyIds_ == localDpuNotifyIds_`; post-side idx→imm mapping is the wait-side
  expected mapping (same idx, same slot value);
- Wait-class primitives (`HcommChannelNotifyWaitOnThread(±WithDefaultTimeout)`; input carries
  `localNotifyIdx`) → `ConsumeDpuPendingNotifyAndEmit` precise path: expected
  imm = `channelSlots[channel][idx]` **precisely known**; directly synthesizes "virtual peer post" to construct
  wc return; imm necessarily passes hcomm hard verification; NotifyWait node records the local real wait event;
  only the "notification arrival" external fact is virtualized (isomorphic to socket echo virtualizing "peer response");
- Design constraint: estimate mode wait consumption **only takes the precise path** — all wait entry points must be intercepted
  (single-card phantom FIFO loopback consumption is unreliable; asymmetric send/receive causes imm mismatch/timeout; old mechanism removed);
  unintercepted entry → `pollCq::miss` **ERROR** (deduplicated per qpn+idx combination once to prevent busy-loop
  log flooding) → hcomm timeout exposes the issue; accordingly add interception in `hcomm_api_stub.cc`.

**taskSrc source marker (BLOB v5)**: `HcclTaskMetaData` adds `taskSrc`
(0=default, 1=DPU); tasks emitted by DPU stub set to 1; enables Python-side precise splitting by source
(not dependent on transport==ROCE or other link semantic criteria; future sources such as AICPU RoCE are unaffected).

**DPU sequence merge pass (Sec. 5.1.2, `sqe_group_merge.py::merge_dpu_write_sequences`)**:
hcomm `WriteWithNotify` splits messages larger than maxMsgSize into N-1 ordinary RDMA_WRITE +
1 terminating WRITE_WITH_IMM (the entire operation still reports only one); collection side stores per WR as N-1 MEM_CPY +
1 WRITE_WITH_NOTIFY. The pass scans MEM_CPY chains with consecutive src/dst addresses in DPU stream order;
terminated and merged into a single entry by address-consecutive WRITE_WITH_NOTIFY (len accumulated across the full chain; description follows the terminating task containing
notifyId; occupies the terminating slot); chains without termination are not merged (independent pure writes remain per-entry `RDMASend`);
only `src_type==1` data tasks participate; other tasks are retained as-is without breaking the chain.

> **Validation** (222 networking 8MB all_reduce): first round (20260921) 15 `Unknown` → 0
> (`Dpu_Notify_Record` ×5 / `Dpu_Notify_Wait` ×9 / `Dpu_Write_With_Notify` ×4,
> link=ROCE); peer rank=2/4/6 (real cross-pod peers); `eid_unresolved=0`;
> WRITE_WITH_NOTIFY enters `d2h` port charging. Final version (20260922, after primitive interception):
> estimate (-np 1) makespan 166.596us, primitive interception 16 times, precise synthesis 11 times, miss=0;
> normal (-np 8) checker All Success (primitives purely forwarded; interception log 0 entries); single-machine 8p
> (118) SQE path regression unchanged.

# light_performance Plugin Design Document (Including Simulator Design)

> Document location: `docs/light_performance/simulator_design.md`
> Requirement source: `docs/light_performance/HCCLSIM/需求描述`
> Reference implementation: `src/plugin/checker/src/framework/task_graph_generator_v3/` (graph construction principles),
> `docs/light_performance/HCCLSIM/hccl_sim/` (simulator prototype, **authoritative baseline for the simulation module**)
> Alignment leftovers and pending match items: Chapter 10
> db task / CCU instruction → JSON task type mapping table: Chapter 11

---

## 1. Requirement Background and Document Description

### 1.1 End-to-End Requirement

Integrate light_performance (a lightweight performance simulation tool) as an independent plugin into the current project, and complete the following end-to-end workflow:

```
hccl-vm workflow execution ──generate──> task sequences / instruction db files
        │
        ▼
light_performance plugin reads db files ──generate──> task graph for performance simulation
        │
        ▼
Performance simulation module performs simulation based on the task graph
        │
        ▼
Output performance profiling JSON file
```

### 1.2 Sub-Requirement Breakdown and Chapter Mapping

| Sub-Requirement | Content | Status | Chapter |
|-----------------|---------|--------|---------|
| Sub-Requirement 1 | Plugin engineering: CLI, compilation and installation, complete plugin architecture | CLI and installation design refined (@estimate one-shot execution channel) | Chapter 3 |
| Sub-Requirement 2 | Integration with hccl-vm: db file reading, task graph generation | **Detailed design in this iteration** | Chapters 4, 5 |
| Sub-Requirement 3 | CCU mode task graph generation (SQE queue + microcode instruction sequence → task graph) | **Detailed design in this iteration** | Chapter 6 |
| Sub-Requirement 4 | Performance simulation module (task graph parsing, performance simulation) | **Aligned and consolidated with HCCLSIM baseline in this iteration** | Chapter 7 |
| Sub-Requirement 5 | Performance profiling JSON output (performance metrics, task graph) | High-level design | Chapter 8 |

> Chapter 10 records **field gaps, type mismatches, and other conflict points** identified during this alignment (differences between the db graph construction path and the HCCLSIM simulation model), for future reference. Chapter 11, Section 11.4 provides the mapping and switchover target for the three fields of the v3-collected profInfo (`HcclTaskProfInfo`) versus JSON display types.

### 1.3 Delivery Scope for This Iteration

This iteration focuses solely on **db file reading, parsing, and the graph construction module**:

- **Input**: db files under the hccl_vm_install/data directory (for format and data, refer to `hccl_vm_install/data/hccl_vm_data.db`).
- **Output**: The task graph `DependencyGraph` and `SimConfig` required by the performance module (hccl_sim).
- **Reference code**:
  - Basic graph construction logic: `src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_generator_v3.cc/h`;
  - CCU microcode instruction to task graph conversion: `src/plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/`;
  - AIV mode graph construction: not considered at this time.
- **Graph construction scope (core design premise)**: The db contains **single-rank task data**, and this plugin performs **single-rank graph construction** —
  this is one of the primary differences from the checker (multi-rank joint graph construction). All **cross-rank task edges
  (post-wait relationships) in the original graph construction process are ignored**: only post-wait edges within a single rank are considered; cross-rank edges are not considered
  (for details, refer to Sections 5.1, 5.4, and 6.6).
- **Relationship with the checker**: The output structure of the light_performance graph construction module is not identical to that of the checker, but the basic graph construction and
  instruction conversion principles are the same; the former does not require the checker's subsequent memory conflict verification / semantic verification and other operations, so appropriate simplification is possible
  (for difference comparison, refer to Section 5.8 and Section 6.7).

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
│   sqlite reading    SQE→Task mapping   simpy discrete    json + trace    │
│   BLOB decoding     notify pairing      event simulation                  │
│   operator grouping CCU microcode       port bandwidth sharing            │
│                     expansion           HBM dynamic sharing               │
│                                         eid→port resolution               │
│  Input: data/hccl_vm_data.db (+ data/hccl_sim.db, ranktable.json)          │
│  Output: (SimConfig, DependencyGraph, TopoResolver) → makespan →            │
│        perf_profiling.json (+ graph dump: DOT/JSON/HTML)                  │
└───────────────────────────────────────────────────────────────────────────┘
```

Data contracts between modules:

```
db_reader     ──OpExecutionGroup(intermediate data model, Section 4.5)──────────>  graph_builder
graph_builder ──(SimConfig, DependencyGraph)─────────────────>  simulator   [output for this requirement]
simulator     ──result dict(tasks with start/end + makespan + ports)──>  profiling/report
```

### 2.2 Plugin Directory Structure (Source location: `src/plugin/light_performance`)

```
src/plugin/light_performance/
├── __init__.py
├── __main__.py             # Standalone CLI entry point (Section 3.3)
├── main.py                 # @estimate plugin one-shot execution entry (Sections 3.1/3.2)
├── manifest.json           # Plugin manifest (name=estimate; installed with package)
├── pipeline.py             # Shared pipeline: db → graph → simulation → perf_profiling.json
│                           #   (shared by run_perf_sim.py and main.py, Section 3.1)
├── db_reader.py            # Sub-Requirement 2: db reading and BLOB decoding (Chapter 4)
├── sqe_group_merge.py      # SQE batch merge preprocessing: merge UDMA data tasks within same (stream, sqeGroupId) batch into 1 (Section 5.1.1)
├── graph_builder.py        # Sub-Requirement 2: SQE → DependencyGraph (Chapter 5)
├── graph_dumper.py         # Task graph dump: DOT + JSON + HTML
├── ccu_graph_builder.py    # Sub-Requirement 3: CCU microcode → task graph (Chapter 6)
├── register_model.py       # CCU register/Event execution model (Section 6.4)
├── sim_config.json         # Default latency/bandwidth configuration (Section 7.2, units us/GBps; installed with package)
├── bw_profile.json         # Default bandwidth utilization configuration (installed with package)
└── simulator/              # Sub-Requirement 4: Simulation module (ported from HCCLSIM, Chapter 7)
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

> The `simulator/` subpackage is ported from `docs/light_performance/HCCLSIM/hccl_sim/`, and the interfaces use
> HCCLSIM as the baseline (Chapter 7). The `dump_graph.py` at the repository root is the one-click dump entry point
> (db → DOT/JSON/HTML, with optional simulation timestamps).
> `manifest.json` and `main.py` (one-shot execution entry, Chapter 3) are placed directly in this package directory.
> During installation, the entire directory becomes the `plugin/light_performance` plugin directory
> (for installation layout, refer to Section 3.1; for CMake rules, refer to Section 3.4).

### 2.3 Relationship with Existing Plugins

| Plugin | Responsibility | Relationship with light_performance |
|--------|---------------|-------------------------------------|
| checker | Communication correctness verification (deadlock/memory conflict/semantics) | Reuses its graph construction principles; primary difference: the checker performs **multi-rank joint graph construction**, while light_performance performs **single-rank graph construction** (cross-rank pairing edges are not built; remote synchronization tasks are preserved and approximated with fixed remote latency), and no verification post-processing is performed |
| runner | Virtual runtime (task-by-task simulated execution of the data plane) | Also consumes opTask/ccuInstr; light_performance is read-only and does not advance isDone |
| insight | Visualization | The profiling JSON / Chrome trace from light_performance can be consumed by it (future work) |

> **Plugin tag versus directory/module name**: The installation tag for this plugin is `estimate` (the host `@estimate` command channel
> and mode gate are hardcoded by tag, as described in Section 3.1), while the plugin directory and Python package name are both
> `light_performance` (`plugin/light_performance`) — the tag corresponds to the "host routing identity",
> and the directory/package name corresponds to the "code organization identity".

---

## 3. Plugin Engineering Design (Sub-Requirement 1)

### 3.1 Plugin Identity and manifest.json

**The plugin tag is fixed as `estimate`** (not the module name light_performance): the host CLI has
hardcoded gating and path conventions for this plugin —

- The execution channel for `hccl-vm plugin run @estimate` is implemented in `src/cmd/cmd_base_utils.cc:599-680`
  (`RunUserPlugin`) with special handling for `estimate`: only estimate mode (`sim::IsEstimateMode()`) is available.
  The plugin directory is fixed as `<installRoot>/plugin/light_performance` (tag and directory name are decoupled).
  The host reads its `manifest.json` and synchronously executes the `entry` command (for details, refer to Section 3.2);
- Mode mutual exclusion: `@checker` is unavailable in estimate mode (`cmd_base_utils.cc:618-624`),
  and `@estimate` is unavailable in non-estimate mode — correctness verification and performance simulation belong to two different runtime modes as final actions;
- The plugin scan/resident model (`src/plugin/hccl_plugin_manager.cc:94-169`, `src/plugin/hccl_plugin.cc`)
  matches the tag by the manifest `name` field (scanning `<installRoot>/plugin`, depth ≤ 2).

manifest.json (source and installation are both within the package directory: `src/plugin/light_performance/manifest.json`
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

- `name`: The plugin tag, used by the host for routing (`hccl-vm plugin run @estimate`).
- `entry`: The startup command within the plugin directory. The @estimate command uses the one-shot synchronous execution channel (Section 3.2).
  During execution, the cwd is the plugin directory, and relative paths are resolved relative to the plugin directory.
- `setting`: Plugin-private configuration key-value pairs, read by main.py itself (following the checker's `SettingManager`
  self-managed configuration approach, with direct JSON parsing on the Python side). Path-type configurations are resolved relative to the plugin directory (that is, the package directory):
  `db` defaults to `<installRoot>/data/hccl_vm_data.db` (same source as the checker's
  `Loader::LoadOpTaskFile`); `sim_config`/`bw_profile` are latency/bandwidth and utilization configurations in the format described in Section 7.2,
  defaulting to `sim_config.json`/`bw_profile.json` in the same directory as the manifest, installed with the package;
  `ranktable` defaults to `ranktable.json` in the same directory as the db;
  `device`/`iter` correspond to the CLI `--device`/`--iter`; `out_dir` is the output root directory
  (subdirectories are created under it based on the runtime timestamp); `per_op_mode` currently supports only per-operator mode (when false, a warning is issued
  and per-operator execution is used); `enable_trace`/`dump_graph`/`max_trace_events` control additional artifacts.

Source and installation layout (the plugin directory is the package directory, with no nesting):

```
Source = Installation product structure (src/plugin/light_performance → hccl_vm_install/plugin/light_performance):
plugin(light_performance)/            # Plugin directory = light_performance package directory
├── manifest.json                     # Plugin manifest (name=estimate)
├── main.py                           # @estimate one-shot execution entry (reuses pipeline.py)
├── __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
├── sim_config.json                   # Default latency/bandwidth (Section 7.2)
├── bw_profile.json                   # Bandwidth utilization
├── simulator/                        # Simulation subpackage (Chapter 7)
└── data/                             # Runtime output (perf_profiling.json and so on, Chapter 8;
                                      #   produced only in the installation tree; this directory does not exist in the source tree)
```

> main.py is located within the package directory but is executed as a script (`python3 main.py`): it adds the **parent directory** of the package
> (`plugin/`, or `src/plugin/` in the source tree) to `sys.path` before performing
> `from light_performance.pipeline import ...` — intra-package relative imports are not affected,
> and the source tree and installation tree layouts are consistent.
> The `run_perf_sim.py` at the repository root (manual entry point) shares the same pipeline implementation with `main.py`:
> main.py adds "installation tree path conventions + manifest setting reading" on top of it.
> Source tree debugging still uses run_perf_sim.py (Section 3.3); within the installation tree, `hccl-vm plugin run @estimate`
> triggers main.py.

### 3.2 Call Chain and Process Model (@estimate One-Shot Execution)

The host side requires only one path adaptation — the one-shot execution channel for `@estimate` is already implemented
(`cmd_base_utils.cc:599-680`). This iteration changes the plugin directory in the special-case branch from `plugin/estimate`
to `plugin/light_performance` (the tag remains `estimate`, and the directory is the package directory):

```
hccl-vm plugin run @estimate
    │
    ▼
RunUserPlugin("@estimate")                                  # subcmd_plugin.cc:55-57 routing
    ├─ Gate: must be in estimate mode, otherwise report error and exit  # cmd_base_utils.cc:611-617
    ├─ Read <installRoot>/plugin/light_performance/manifest.json  # Missing file or missing entry field → error
    └─ system("cd <installRoot>/plugin/light_performance && python3 main.py")   # Synchronous blocking execution
       └─ Non-zero exit code → HCCL_VM_ERROR, command returns failure overall
```

**Client wait strategy for long-duration commands** (`StartHostClient`/`ServerListen` in `cmd_base_utils.cc`):
Within a session, `hccl-vm` sends messages to the host through a POSIX message queue, and **the host responds only after `ParseCommand` returns (that is, after the command finishes executing)**;
after CCU CcuBuffer mode microcode expansion, the task count can reach 10^5 (measured at 192,005 tasks, software simulation approximately 54s),
far exceeding the 5s response limit for ordinary commands. To address this:

- After the client identifies `plugin run @estimate`, it uses a long-wait upper limit (default 30 minutes, overridable through the environment variable
  `HVM_CLIENT_RESP_TIMEOUT_MS`). During the fragmented wait period, it prints
  "command is still executing on the host" to the terminal every 30s (first at 5s), and prints a clear warning upon timeout; other commands maintain the 5s limit.
- The response carries the original command string (`ACK:<cmd>`). Before sending a command, the client drains the response queue first and accepts only responses that strictly match the current command
  — this avoids mistakenly reading a late response from a previously timed-out command and returning prematurely (especially important when the same command is retried after interruption).
- The host prints `[HVM] @estimate performance simulation started/finished` markers to the terminal at the start and end of simulation (stdout, included with the report
  output); this mechanism ensures that **the prompt returns only after the simulation truly finishes**, eliminating any misconception that "the simulation has already finished".

main.py (plugin side) execution flow:

```
RunEstimate():
    setting   ← Parse the setting section of manifest.json (paths resolved relative to plugin directory)
    db        ← setting.db (default <installRoot>/data/hccl_vm_data.db, derived through __file__ as ../../data)
    config    ← setting.sim_config / bw_profile (use built-in defaults when missing, Section 7.2)
    ranktable ← ranktable.json in the same directory as db (_find_ranktable default rule; consistent with the host InitHvmCommEnv
                writing to data/ranktable.json, cmd_base_utils.cc:437-450, no configuration needed)
    groups    ← db_reader.load_op_executions(db)            # Chapter 4
    topo      ← TopoResolver(ranktable)                     # Section 7.6
    for group in groups:                                    # Per operator (Section 5.1)
        config, dep_graph = graph_builder.build(group)      # Chapters 5, 6
        result = HcclSimulator(config, dep_graph, topo).run()   # Chapter 7
    profiling.dump(<plugin>/light_performance/data/<timestamp>/perf_profiling.json)   # Chapter 8
```

Comparison with the checker resident process model (stdin single-line JSON command loop) and selection rationale:

| Dimension | checker (resident model) | estimate (one-shot model) |
|-----------|-------------------------|--------------------------|
| Startup method | Fork+pipe resident during `plugin install` (`hccl_plugin.cc:74-156`) | `cd && entry` synchronous execution during `plugin run` (`cmd_base_utils.cc:655-660`) |
| Trigger protocol | stdin JSON: `{"action":"status","payload":{"status":"finish","data_id":...}}` | No protocol: entry starts immediately; all input paths use conventions + manifest setting |
| data_id | Carried in message (generated by host `DumpData`) | Not passed; output directory uses runtime timestamp as substitute (Section 10.3 #13) |
| Lifecycle | Reclaimed upon stop command/pipe EOF/host exit | Process exits naturally; exit code is the execution result |
| Adaptation reason | Verification must be triggered by the host at the end of the workflow, and may run multiple times | Simulation is offline batch processing: consumes the already-persisted db, runs once and exits |

> **Evolution direction** (not implemented in this iteration): If "automatic simulation trigger by the host at workflow end" (resident model) is needed in the future,
> main.py can add a stdin command loop to reuse the same pipeline — the host's message channel
> (`SendMessageToPlugin`) and resident startup (`RegisterPlugin`) are already in place. At that point, only
> the @estimate special-case branch in `RunUserPlugin` needs to be switched back to the general message path.

Data contracts (fixed conventions within the installation tree, all aligned with existing mechanisms):

| Path | Source | Description |
|------|--------|-------------|
| `data/hccl_vm_data.db` | Written during host workflow execution | Main simulation input; estimate mode produces **single-rank task data** (phantom peer), naturally aligned with the single-rank graph construction premise in Sections 4.6/5.1 |
| `data/ranktable.json` | Written by `InitHvmCommEnv` | eid→port resolution (Section 7.6); `_find_ranktable` defaults to the same directory as the db, no explicit configuration needed |
| `plugin/light_performance/sim_config.json`, `bw_profile.json` | Installed with package directory through CMake (Section 3.4) | Section 7.2 format (units us/GBps) |
| `plugin/light_performance/data/<timestamp>/` | Created at runtime by main.py | Output directory. **Must not be placed under `data/`**: the host executes `remove_all("data")` at each start (`src/main.cc:32`), and the contents of the data directory are not preserved across sessions |

### 3.3 Command-Line Design

Host integration mode (`plugin run` is the simulation entry point, replacing manual execution of run_perf_sim.py):

```bash
# ① Start the environment in estimation mode (choose one)
hccl-vm start <topo> --estimate             # Interactive mode, PS1 displays (estimate)$>
hccl-vm run <topo> --estimate <sample command>     # One-shot mode
# (Within an already started session, you can also switch with hccl-vm mode estimate, subcmd_mode.cc)

# ② Run workload → produce data/hccl_vm_data.db (single-rank task list)

# ③ One-click performance simulation (available only in estimate mode, Section 3.2 gating)
hccl-vm plugin run @estimate
# Artifacts: plugin/light_performance/data/<timestamp>/perf_profiling.json (+ trace / graph dump)
```

- The one-shot execution channel does not pass through command-line arguments; simulation parameters (target device, opIter filter, trace/dump switches, and so on)
  are configured through the manifest `setting` section (Section 3.1). After modifying the configuration, rerun `plugin run @estimate` for changes to take effect.
- `plugin install @estimate` is also executable (through the `RegisterPlugin` resident startup channel). main.py finishes in one shot and exits, which is acceptable degraded behavior (Section 10.3 #14); regular use does not require install.

Standalone debugging mode (does not depend on the host, convenient for offline db analysis; already implemented):

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
| `--db` | Path to hccl_vm_data.db (required) |
| `--config` | Latency/bandwidth configuration (Section 7.2 format, units us/GBps; uses built-in defaults by default) |
| `--bw-profile` | Bandwidth utilization configuration (default utilization is 100%) |
| `--jetty-num` | Override the UB multi-channel (jetty) count stored in the db, for multi-jetty utilization conversion (defaults to each operator's `opDetails.ubMultiChannelNum`, as described in Section 7.2) |
| `--ranktable` | Topology file for eid → physical port resolution (defaults to `ranktable.json` in the same directory as the db; when network tasks exist and no topology is available, **simulation is skipped and only dump is performed**) |
| `--device` | Target device (deviceId) for single-rank graph construction; defaults to the first device in opDetails, needed only when the db contains multi-device data |
| `--iter` | Simulate only the operator for the specified opIter (default: all) |
| `--trace` | Export Chrome tracing JSON (for multiple operators, files are split by `_<commId>_<opIter>`) |
| `--dump-dir` | Task graph dump directory (each operator outputs .dot/.json/.html, including simulation timestamps) |
| `-v` / `-s` | Task-level detail / summary only |

### 3.4 Compilation and Installation Design (hccl_vm_install/plugin/light_performance)

Add an install rule in `src/plugin/CMakeLists.txt` (following the precedent of insight's directory installation;
Python plugins do not require compilation targets. The plugin directory is the package directory, and manifest.json/main.py/default configurations are all
within the package directory, so a single DIRECTORY rule can carry all contents):

```cmake
# estimate performance simulation plugin: plugin directory is the light_performance package directory, install entire directory
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/light_performance"
    DESTINATION "plugin"
    FILE_PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
    PATTERN "__pycache__" EXCLUDE
    PATTERN "data" EXCLUDE)      # Runtime output directory is not included in the installation package
```

Comparison with existing plugin installation methods:

| Plugin | Installation Method | Product |
|--------|-------------------|---------|
| checker | `install(TARGETS checker DESTINATION plugin/checker)` + `install(FILES manifest.json)` (`src/plugin/checker/CMakeLists.txt:67-82`) | Compilation product + manifest |
| insight | `install(DIRECTORY insight DESTINATION plugin/visualization PATTERN frontend EXCLUDE)` (`src/plugin/CMakeLists.txt:26-39`) | Python directory + manifest |
| **estimate (this iteration)** | `install(DIRECTORY light_performance DESTINATION plugin)` (single rule; manifest/main.py/default configurations are all within the package directory and installed with the package) | `plugin/light_performance/`: Python package + manifest (plugin directory and package directory are unified, no nesting) |

Build and verify: After `bash build.sh` (internally `cmake && make && make install`, with the installation prefix defaulting to
`hccl_vm_install/` within the repository, `CMakeLists.txt:89-92`), check the installation product:

```
hccl_vm_install/plugin/light_performance/
├── manifest.json
├── main.py
├── __init__.py / __main__.py / db_reader.py / graph_builder.py / pipeline.py / ...
├── sim_config.json / bw_profile.json
└── simulator/ ...
```

- Packaging path: `build.sh --pkg` packages the entire `hccl_vm_install` into a tar archive, and the estimate plugin is distributed with it.
- Dependencies: python3 (≥3.8, including the sqlite3 standard library) and **simpy** (discrete event simulation engine, current environment
  4.1.2, imported in `simulator/simulator.py`); the deployment environment must have these pre-installed (the insight plugin also depends on python3,
  consistent convention), as described in Section 10.3 #16.

---

## 4. db File Reading and Parsing (Sub-Requirement 2 - First Half, Focus of This Iteration)

### 4.1 db Location and Opening

- The path is fixed as `<installRoot>/data/hccl_vm_data.db` (consistent with the checker's `Loader::LoadOpTaskFile`).
- Open in read-only mode: Python `sqlite3` standard library, `mode=ro` URI, to avoid write conflicts with the running hccl-vm:
  ```python
  conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
  ```
- Auxiliary database `hccl_sim.db` (same directory): used only to resolve `opDetails.commId` to communicator identity
  (`Communicator` table, uniquely identified by `comm_id[128] + comm_hash`), optional.

### 4.2 Table List and Association Model

hccl_vm_data.db is created by `OpDbOps::InitSchema()` (`src/runnerdb/modeldb/db_hccl_op_db_ops.cc:76-184`).
The tables consumed by light_performance are as follows:

| Table | Primary Key/Association | Content | Purpose |
|-------|------------------------|---------|---------|
| `opDetails` | `id`; `pid`→opTask table name | One row per operator per rank: pid/deviceId/rankId/commId/opIter/syncIter/streamId(main stream)/root/rankSize/srcRank/dstRank + `opDetail` BLOB (operator parameters) + `algorithmName` (algorithm selection result, captured and backfilled through DFX hijack/log dual paths, see note below) + `ubMultiChannelNum` (UB multi-channel/jetty count, proxy level0 stub reads environment variable `HCCL_UB_MULTI_CHANNEL_NUM` and stores it; defaults to 1 when not configured or invalid, for multi-jetty conversion, refer to Section 7.2) | Operator grouping, main stream identification, operator parameters, algorithm selection results, bandwidth utilization conversion |
| `opMemInfo` | `opDetailId` → opDetails.id | Addresses and sizes for input/output/ccl memory segments | Direction determination (Section 5.5), data volume verification |
| `opTask_P_<pid>` | `opDetailId` → opDetails.id | SQE task queue: deviceId/streamId/taskType/isDone + `opTaskMeta` BLOB | **Primary graph construction input** |
| `ccuChannels` | `(deviceId, dieId, channelId)` | CCU channels: src/dst die/device/rank + leid/reid(16B EID) + protocol + jettyNum + jettyId array | CCU channel → peer/jetty mapping (Chapter 6) |
| `ccuInstrRes` | `(deviceId, dieId)` | Microcode instruction space per die: instrCount + instrSpace BLOB (32K×32B) | CCU microcode source (Chapter 6) |
| `ccuInstr` | `ccuInstrResId` → ccuInstrRes.id | One row per microcode delivery (startId/instrInfoSize) | Used only to detect "new microcode delivery"; decoding does not depend on it |
| `syncRecords` | — | Host-side stream synchronization records | Not used in this iteration |
| `JettyMaps` | `opDetailId` | eid mappings recorded per operator | Backup (not used when ccuChannels provides coverage) |

> For table structure definitions, refer to `include/runnerdb/sim_op_db_types.h`; for writer-side semantics, refer to `src/runnerdb/tracedb/db_sim_op_db_ops.cc`.

> **Source of the algorithmName column (algorithm selection result capture)**: Dual-path capture, both using last-wins backfill to
> `sim::UpdateOpDetailAlgName`, serving as mutual backups and writing the same value.
>
> 1. **DFX hijack stub (primary path)**: After algorithm selection completes, the workload calls
>    `HcclDfxRegOpInfoByCommId(commName, &HcclDfxOpInfoCompat)` per operator (hccl repository
>    op_common.cc), where `algTag` carries the selected algorithm name (format
>    `{opType_commName}_{algName}_{host|device}[_{dataType}...]`, graph mode
>    `Graph_{algName}_{...}`). Level=2 mode (default) hooks only level0+level2;
>    the level1 dfx_stub is not in the process. `src/proxy/level2/hcomm_api_stub.cc` uses
>    level2 strong symbols to preempt resolution (preload order precedes the real
>    libhcomm.so loaded by the workload through the Python plugin chain), parses the algorithm name from algTag, backfills it, and then
>    **forwards to the real implementation through `DlsymRealWithFallback` (RTLD_NEXT + libhcomm.so SONAME fallback + dladdr
>    self-call prevention), passing through the return value without altering call semantics**. The struct layout is replicated from
>    hccl_res_dl.h (`static_assert` locks algTag offset at 68), and content verification
>    (commId prefix + host/device suffix) filters out op-begin and other registrations without algorithm names,
>    providing a fallback for workload header drift. In Level=1 mode, this stub is shadowed by the level1 dfx_stub (behavior
>    same as the old version), and capture falls back to the log path.
> 2. **Log capture (backup path)**: The tool is the implementer of the workload log underlying interface (the workload
>    `DlogRecord` weak symbol resolves to `src/proxy/level2/log_stub.cc`). The level0 stub
>    `RecordOpDbInfo` first inserts the opDetails row (setting the runnerdb global `g_currOpDetailId`) and then calls
>    the real workload. The INFO-level algorithm selection log output by the internal selector (`[HcclGetAlgRes] algName=` /
>    `SelectMinCost: selected algName=` / old selector `the selected algo type is ` — three types of markers,
>    last-write-wins for multiple matches within the same operator) is extracted through the `DlogPrintStub` entry point match. Capture occurs before the message enters the tool log output
>    (immune to `log_config.yaml` level). The proxy layer also provides `acllogCheckDebugLevel`
>    constant-pass stub (immune to workload-side log level thresholds).
>
> Old databases are automatically updated with `ALTER TABLE` to add columns through `OpDbOps::InitSchema()`; the plugin reader falls back to empty values for old databases without the column.

**Operator grouping model**: One operator instance = rankSize opDetails rows under the same `(commId, opIter)` (one per rank),
each with an associated opMemInfo and a set of tasks (`WHERE opDetailId=?`, with the table name derived from the row's `pid` column as `opTask_P_<pid>`,
following the checker's `QueryCompositeOpDetailByOpIter`).

### 4.3 opTask Dynamic Tables and Multi-pid Handling

- Each task-writing process creates its own table `"opTask_P_" + getpid()`; AICPU device-side processes use `getppid()` to share the same table as the host.
- **Discover all task tables** (enumeration is mandatory; multi-pid is the norm; early sample dbs already had `opTask_P_477776`/`opTask_P_477780` — two tables; the current sample has a single table):
  ```sql
  SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'opTask_P_%'
  ```
- Consumption order: within the same operator, tasks are ordered by ascending `id` (= SQE submission order, which is the basis for notify first-come-first-served pairing).

### 4.4 BLOB Decoding Specification (pack(1), Little-Endian)

All BLOBs are raw C struct memcpy (no serialization framework). The Python side uses the `struct` module for decoding.
All layouts below have been verified with actual data from `hccl_vm_install/data/hccl_vm_data.db` (from v5 collection onward, sample
opTaskMeta length is 190B; historical samples are 189B/188B).

#### 4.4.1 opTaskMeta (190 bytes)

The struct definition is shared with the writer side: `include/sim_profiling_defs.h` (single-source definition for profiling reconciliation information)
and `src/plugin/checker/header/internal/task_meta_defs.h` (HcclTaskMetaData).

v5 collection structure (compared to the v4 189B version): `taskSrc` (1 byte) is added after `sqeGroupId` —
task source marker (`HcclTaskSrc`: 0=default SQE/aclrt native path, 1=DPU that is host CPU RoCE
`ibv_dpu_stub` collection path), used by the performance simulation DPU sequence merge pass to precisely split by source
(Section 5.1.2). The taskData starting offset changes from 69 to 70.

v4 collection structure (compared to the v3 188B version): `sqeGroupId` (1 byte) is added after `jettyId` — UDMA
SQE batch grouping marker (all WQEs within one doorbell share the same group), used by the performance simulation entry point for batch merging
(Section 5.1.1). The taskData starting offset changes from 68 to 69. Group number semantics: **circularly incrementing doorbell batch number 1..255**
(wraps at 255; the collection side `device_sqe_parse_stub.cc` maintains the global `g_currSqeGroupId`,
`% 255 + 1`) — batch numbers are unique within a 255-entry window. Doorbells from other operators are not visible in the current operator's task list
but also consume batch numbers, so the same number within the same stream must belong to the same doorbell; **0 is reserved for "unmarked"**
(HcclTaskMetaData construction default), the counter never takes 0, and the merge entry point treats `sqeGroupId <= 0`
as non-participating (Section 5.1.1).

v3 collection structure (compared to the v2 185B version): `HcclTaskProfInfo` sub-structure (3 consecutive bytes,
treated as an opaque whole by checker/runner for pass-through) is added after `taskType`: `profTaskType` (aligned with hcomm TaskParamType value range 0~22) and
`linkType`/`transportType` (aligned with hcomm DfxLinkType value range 0~9 / DfxTransportType value range 0~4,
used for reconciliation with profiling data). The taskData starting offset changes from 65 to 68.

> **Implementation status**: `db_reader.py` automatically identifies the layout by BLOB length — `>= 190B` follows v5
> (`<BBBBQIQQIBB16s16s`, taskSrc decoded into `HcclTaskMetaData.src_type`; this branch
> **must be evaluated before the v4 check**, otherwise 190B would first match the `>= 189` v4 branch causing eid/taskData
> offset misalignment), `>= 189B` follows v4
> (`<BBBBQIQQIB16s16s`, sqeGroupId decoded into `RawTask.sqe_group_id`, src_type=0),
> `188B` follows v3 (`<BBBBQIQQI16s16s`, sqe_group_id filled with -1, does not participate in merging),
> `185~187B` follows old v2 (`<BQIQQI16s16s`, profInfo filled with 0xFF=UNKNOWN) for backward compatibility,
> `< 185B` is discarded with a warning. The consumption of profInfo (`task_type` display name + link/transport)
> **is implemented** (#18, Section 11.4).

v2 collection structure (compared to the earlier 169B version): `lcEid[16]` (explicit local EID) is added to the header,
and taskData members previously had `srcRankId/dstRankId` fields added.

> **rank field rollback: srcRankId/dstRankId are deprecated (current convention)**: The
> `srcRankId/dstRankId` fields that were previously added to taskData members have been rolled back and deleted, uniformly replaced by `srcDeviceId/dstDeviceId` — the collection side
> (estimate mode, `device_sqe_parse_stub.cc` and so on) writes the communicator rankId directly into
> the `srcDeviceId/dstDeviceId` fields, that is, **the deviceId field carries the rankId** (when the peer rank is unresolved,
> the value is `0xFFFFFFFF`). After `db_reader.py` unpacks, it directly consumes the values as `src_rank_id/dst_rank_id`
> (reads `srcDeviceId` as `srcRankId`). This is a hard switch: old BLOBs collected before the field rollback
> (dual-field layout with rank/device) are no longer supported.

```
Header 70B:  struct.unpack_from("<BBBBQIQQIBB16s16s", blob, 0)
┌────────┬──────┬──────────────┬────────┬───────────┬──────────────────────┐
│ Offset │ Size │ Field        │ Type   │ Sentinel  │ Description          │
├────────┼──────┼──────────────┼────────┼───────────┼──────────────────────┤
│ 0      │ 1    │ taskType     │ u8 enum│ —         │ Redundant with table column taskType │
│ 1~3    │ 3    │ profInfo     │ sub-str│ 0xFF      │ HcclTaskProfInfo:    │
│        │  1   │ .profTaskType│ u8 enum│ 0xFF      │ Aligned with hcomm TaskParamType │
│        │  1   │ .linkType    │ u8 enum│ 0xFF      │ Aligned with hcomm DfxLinkType │
│        │  1   │ .transportType│u8 enum│ 0xFF      │ Aligned with hcomm DfxTransportType │
│ 4      │ 8    │ commId       │ u64    │ —         │ Communicator ID      │
│ 12     │ 4    │ rankId       │ u32    │ 0xFFFFFFFF│ Backfilled through device→rank when default │
│ 16     │ 8    │ deviceId     │ u64    │ —         │ Physical device ID   │
│ 24     │ 8    │ streamId     │ u64    │ —         │ Stream ID            │
│ 32     │ 4    │ jettyId      │ u32    │ 0xFFFFFFFF│ Invalid = no network │
│ 36     │ 1    │ sqeGroupId   │ u8     │ 0         │ Doorbell batch number 1..255 circular increment (added in v4, Section 5.1.1); 0 = unmarked │
│ 37     │ 1    │ taskSrc      │ u8 enum│ 0         │ Task source (added in v5, Section 11.4.6): 0=default, 1=DPU(host ibv) │
│ 38     │ 16   │ rmEid[16]    │ bytes  │ All 0     │ Remote EID           │
│ 54     │ 16   │ lcEid[16]    │ bytes  │ All 0     │ Local EID (added in v2) │
│ 70     │ 120  │ taskData     │ union  │ Interpreted by taskType, see below │
└────────┴──────┴──────────────┴────────┴───────────┴──────────────────────┘
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

> Both `len` and `dataCount` are **byte sizes** (the reduce field name has historical reasons; it is actually used as size).
> `WriteWithNotifyTask.len` is also byte size; `isReduce=1` indicates that the write payload is reduction data
> (`dataType`/`reduceOp` are valid in this case), mapped to a single `Write_With_Notify` task (Section 5.2).

#### 4.4.2 taskType Enumeration Full Table

`HccLTaskMetaType` (`include/sim_common_defs.h:184-193`):

| Value | Enumeration | light_performance Handling |
|-------|-------------|---------------------------|
| 0 | `NOTIFY_WAIT` | → local_wait / remote_wait (Section 5.2) |
| 1 | `NOTIFY_RECORD` | → local_record / remote_record |
| 2 | `REDUCE` | → memcpy / local_copy (reduction approximated, Section 5.7) |
| 3 | `MEM_CPY` | → memcpy / local_copy (determined by jettyId) |
| 4 | `CCU_GRAPH` | → CCU subgraph expansion (Chapter 6) |
| 5 | `AIV_GRAPH` | Not supported; skip with warning |
| 6 | `SYNC_STREAM` | → Skip, no node generated (Section 5.6) |
| 7 | `WRITE_WITH_NOTIFY` | → Single task mapping `Write_With_Notify` (network write + attached notify, not split, Section 5.2) |

#### 4.4.3 opDetails.opDetail (28 bytes)

`OpDetails` (`task_meta_defs.h:141-165`): `<HHH` header + union:

```
opType u16@0 | dataType u16@2 | reduceType u16@4
opV1: count u64@6 | dataType u16@14 | reduceOpType u32@16          (total 28B)
opV2: sendCount u64@6 | sendDataType u16@14 | recvCount u64@16
      | recvDataType u16@24 | extInfo u16@26                       (AllToallV type)
```

opType is `HcclCMDType` (ALLREDUCE=2, ALLGATHER=6, REDUCE_SCATTER=7, ALLTOALL=10, and so on;
for the complete enumeration, refer to `src/plugin/checker/header/external/hccl_types.h:162-189`), used for operator labeling in profiling reports.
`opExtInfo` BLOB (optional) stores V-operator counts/displs, AllToallV send matrix, and so on; this iteration only passes it through.

V1/V2 layouts are distinguished by opType (level0 stub writer-side convention, `hccl_op_stub.cc`):
**V2 applies only to AllToAll(10)/AllToAllV(8)/AllToAllVC(9)** (`BuildOpDetailsV2`, sendCount/recvCount
are the total element counts sent/received by the local rank); **AllGatherV(16)/ReduceScatterV(17) use V1 + opExtInfo**
(count = element count on the local rank side). During decoding, db_reader selects the layout by opType; V2 operators decode the send and receive
element counts and types on both sides (`send_count/recv_count/send_data_type/recv_data_type`),
while `data_size/data_type` follow the send side to maintain backward compatibility with old consumers, and `reduce_op` is set to 0.

#### 4.4.4 ccuChannels BLOB

| Field | Format | Description |
|-------|--------|-------------|
| `leid` / `reid` | 16B raw bytes each | Source/destination endpoint EID (`URMA_EID_LEN=16`) |
| `jettyId` | `jettyNum × u32` little-endian | Contiguous jetty number array (C struct reserves `u32[64]`; sample measurement: jettyNum=1, jettyId=(1024,)) |
| `protocol` | int column | CommProtocol: HCCS=0/TCP=1/ROCE=2/UB_CTP=3/UB_TP=4/PCIE=5/SIO=6 |

Note: The `srcRankId/dstRankId` columns are actually filled with device_id (channel establishment precedes communicator creation; see writer-side
`hccp_ccu_stub.cc:401-405` comments). The rank semantics require correction through opDetails.

#### 4.4.5 ccuInstrRes instrSpace

- `instrSpace` = a **complete microcode instruction space snapshot** for one die: `32K entries × 32B = 1 MiB`
  (`CCU_INSTRUCTION_NUM=32*1024`, `CCU_INSTRUCTION_SIZE=32`, `include/sim_common_defs.h:35-36`).
- The i-th instruction is at byte offset `i*32`; `instrCount = max(historical value, startId+instrCnt)`, **gaps are allowed**.
- During reading, only the first `instrCount` instructions are retrieved; the CCU task's `instStartId/instCnt` is directly used as the index range.
- The `ccuInstr` table is only for runtime awareness (the runner polls `COUNT(ccuInstr)+COUNT(ccuInstrRes)` changes);
  light_performance decoding does not depend on this table.

### 4.5 Intermediate Data Model (db_reader Output)

```python
@dataclass
class OpExecutionGroup:          # One operator instance (single-rank perspective: only target device task data)
    comm_id: int
    op_iter: int
    op_type: int                 # HcclCMDType
    rank_size: int
    main_stream_ids: dict[int, int]      # deviceId → main stream streamId (opDetails.streamId)
    tasks: list[RawTask]         # All SQEs for this operator, globally ordered
    mem_info: dict[int, MemInfo]         # deviceId → input/output/ccl ranges
    channels: dict[tuple, ChannelInfo]   # (srcDev, srcDie, channelId) → channel information
    instr_space: dict[tuple, list[bytes]]# (deviceId, dieId) → 32B instruction list
    algorithm_name: str          # Algorithm selection result (opDetails.algorithmName, captured and backfilled through DFX/log dual paths; empty string for old databases)
    ub_multi_channel_num: int    # UB multi-channel (jetty) count (opDetails.ubMultiChannelNum; old database/zero value treated as 1)
    send_count / recv_count: int # Send/receive element counts on both sides (V2 layout operators; consistent with count for V1)
    send_data_type / recv_data_type: int # Send/receive data types on both sides (same as above)

@dataclass
class RawTask:
    global_id: int               # Global order across opTask_P_* tables (sorted by table pid value + in-table id)
    op_detail_id: int
    device_id: int
    stream_id: int
    meta: HcclTaskMetaData       # 4.4.1 decoding result (dataclass form)
```

`meta` includes v2-added fields: header `lc_eid` (explicit local EID, used with priority for local eid resolution in Section 5.5);
NotifyTaskData/TransMemTaskData both carry `src_rank_id/dst_rank_id`
(unpacked from the rankId value carried in the deviceId field under the estimate collection convention; for details, refer to Section 4.4.1 rank field rollback).

### 4.6 Data Integrity Verification

For each `OpExecutionGroup`:

1. **Single-rank filtering (design premise)**: The performance plugin consumes only **single-rank task data** (the sample db contains only device 1,
   which is the design norm, not data loss). If the db contains multi-device data, filter opDetails and their tasks by target deviceId (`--rank`, defaulting to the first device);
   `rankSize` is passed through only as metadata; it is not required that `len(opDetails row count) == rankSize`.
2. **notify peer** (no backfill):
    - Estimate collection convention (`device_sqe_parse_stub.cc`, consistent for both task types): for notify,
      `srcDeviceId` (carrying rankId) = record side, `dstDeviceId` (carrying rankId) = wait side,
      both explicitly filled by the launching end; when the peer rank is unresolved, the value is `0xFFFFFFFF`.
    - The earlier `dstDeviceId == 0` backfill fallback was removed along with the rank field rollback (the 0 sentinel conflicts with the valid
      rank 0, and after the launching end always explicitly fills, no backfill is needed).
    - The notify carried by `WRITE_WITH_NOTIFY` (7) targets the **write peer** (the wait side is not in this db, per the single-rank
      simulation premise) and does not participate in local pairing; its `dstDeviceId` (carrying rankId) is the write destination.
    - Note: The runner/checker consumer side (`DecodeAndExpandTaskMeta`) splits 7 into
      [write, NOTIFY_RECORD] two sub-tasks at the decoding boundary; light_performance **does not split** — single task mapping
      `Write_With_Notify` (Section 5.2) ensures a one-to-one correspondence between simulation tasks and db tasks.
3. **Decoding length verification**: `len(opTaskMeta) >= 185` (auto-adapts v4/v3/v2 layout by length, Section 4.4.1),
   `len(opDetail) >= 28`,
   `len(instrSpace) >= instrCount*32`; if not satisfied, the row is discarded with a warning
   (the checker uses direct memcpy; this approach is more lenient).

### 4.7 Decoding Example (Measured from hccl_vm_data.db)

Sample db (REDUCE_SCATTER, opType=7, dataType=4/FP32, count=8, rankSize=2, devType=6/950)
measured decoding results:

- `opTaskMeta` length is consistently 188B (v3 collection sample; from v4 collection onward it is 189B, Section 4.4.1); taskType distribution: 4 (CCU_GRAPH) and 6 (SYNC_STREAM) alternating.
- opDetails contains only device 1 (rank 0) → single-rank data (design norm); main stream is stream 2.
- Task sequence (main stream per opIter):

```
stream 2 (main stream):  CCU_GRAPH(die=1, mission=0, instStartId=14, instCnt=100, argSize=13)
                          SYNC_STREAM(syncIdx=n)          # Microcode delivery + stream sync
```

- The CCU task args are item-by-item consistent with the data directory file `sqe_info_device_1_die_1_mission_0_startId_14.txt`
  (args[0]=281475405602881, args[1]=281475405602945, and so on);
  `args[0]/args[1]` are measured as the inputAddr/outputAddr from opMemInfo.
- `ccuInstrRes` (dev 1, die 1) instrCount=114 = instStartId(14) + instCnt(100).
- In early 169B samples containing MEM_CPY/NOTIFY, measurement shows: MEM_CPY srcOffset/dstOffset can both fall within
  the opMemInfo input/output/ccl ranges (`input ⊃ output`; for ReduceScatter, the output is the first
  1/rankSize segment of the input); for network copies with jettyId, the offset also manifests as "local range address"
  (communicator memory symmetric layout; the peer buffer has the same VA as the local side), so **direction determination cannot rely on address alone**
  (see Section 5.5). This conclusion also applies to v2 data.

#### 4.7.1 v2 AICPU Mode Sample (opTask_P_118800)

REDUCE_SCATTER, rankSize=8, count=32768 (FP32, output=128KB), 76 tasks,
stream 2 (main stream) / 6 (scheduling stream) / 7..13 (one stream per peer). Key differences from the CCU mode sample:

- taskType distribution: NOTIFY_WAIT×30, NOTIFY_RECORD×30, MEM_CPY×8 (7 network write
  jet=1..7 + 1 local), REDUCE×7 (local), SYNC_STREAM×1; no CCU_GRAPH, `ccuChannels` is empty.
- **Direction is entirely determined by the rank dimension** (Section 5.5 rule 1): all 7 network MEM_CPY tasks have
  `srcRank=0/dstRank=1..7` (write), task offsets do not intersect with opMemInfo ranges (range inference fails),
  `direction_inferred=0`.
- **notifyId reuse across ranks**: for example, `RECORD(nid=30, dst=dev2)` (wait side is on the peer, skipped) and
  local `WAIT(nid=30)` (record side is on the peer, not in this db) appear in pairs, 7×2 each over two rounds;
  16 locally paired (nid 3/13 main↔scheduling stream, 15..27 scheduling stream→peer stream, 6..12 peer stream→scheduling stream),
  14 unpaired local WAITs → charged at `remote_wait_delay` (Section 5.4).
- lcEid/rmEid match ranktable.json byte-for-byte (`…0040…01010001` → port `1/0`, and so on);
  all network task ports are resolvable.

### 4.8 eid→rank Index (`build_rank_index`, Peer Rank Resolution)

Peer rank resolution (Section 5.5 SQE fallback, Section 6.7 CCU primary path) shares a single eid(hex) → rank_id
index, constructed with the following priority and basis:

1. **RankEidMap table (primary path)**: The `RankEidMap` table in runnerdb `hccl_sim.db` (same directory as hccl_vm_data.db).
   In estimate mode, the Communicator table has only one row for the local rank; the peer rank
   uses this table as the authoritative source (`sim_models.h` comment; host precedent `GetRmtRankIdByEid`,
   `device_sqe_parse_stub.cc:889`). Row BLOB layout is consistent with `sim::RankEidMap`:
   `id(u64) + rankId(u32) + eid(16B) + pad(4B)`; the eid is "hexadecimal string byte order",
   same source and same order as `ccuChannels.leid/reid`. The Python side matches directly through `bytes.hex()` equality.
   This table is written by ranktable.json parsing (`topo_ascend_cluster_parser.cc:229`),
   and in the mock-comm scenario it is entirely rewritten along with communicator initialization, so it is authoritative.
2. **ranktable.json (fallback)**: When RankEidMap is empty / `hccl_sim.db` is missing / the table does not exist, expand
   `rank_list[].rank_id ↔ level_list[].rank_addr_list[].addr` (addr_type=EID)
   to build an isomorphic index. Measurement shows both path indexes are fully consistent (8 ranks × 9 eids = 72 entries).

Open in read-only mode (`file:…?mode=ro` URI); read failures (including table not existing) degrade to fallback rather than error.
Note that **`ccuChannels.srcRankId/dstRankId` cannot be directly used as communicator rank**: the writer side
(`hccp_ccu_stub.cc:401-405`) explicitly fills global endpoint deviceId (channel establishment precedes
communicator creation; the same channel can be reused by multiple domains), and there is an offset from the opDetails.rankId numbering system.
The checker has therefore deprecated this field (`g_allRankChannelInfo` stores only dstDeviceId/dstDieId).

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
① Per-SQE → Task node (Section 5.2 mapping table; CCU_GRAPH handed to Chapter 6 for expansion)
   │     stream_id: int (SQE stream uses original streamId directly; CCU skeleton tasks also belong to their SQE original stream, Section 6.5)
   ▼
② Stream serial chain: group by (deviceId, streamId), ascending global_id → dep_graph.add_stream_chain
   ▼
③ notify pairing: cross-rank tasks do not participate in pairing (nodes preserved); only local notify first-come-first-served → dep_graph.add_match (Section 5.4)
   ▼
④ Network task parameter completion: direction inference / eid resolution (Section 5.5)
   ▼
⑤ CCU subgraph expansion (Chapter 6): CCU_GRAPH placeholder → microcode instruction interpretation → internal nodes + Event pairing
   ▼
⑥ Output assembly: (SimConfig, DependencyGraph) → HcclSimulator(config, dep, topo)
```

### 5.1.1 SQE Batch Merge (Pre-Graph-Construction Preprocessing)

On the collection side, one UDMA doorbell (SQE) can carry multiple WQEs. Tasks are still stored per WQE in the database and marked with
`sqeGroupId` for batch attribution (circularly incrementing doorbell batch number 1..255, wraps at 255, never takes 0; 0 = construction
default "unmarked", Section 4.4.1). Before graph construction, split into batches by **(streamId, sqeGroupId)** — the split key explicitly
includes the stream, and all data tasks within a batch (MEM_CPY/REDUCE/WRITE_WITH_NOTIFY) are merged into a single entry, aligned with
the hcomm batch transfer profiling `ExecProfilingAll` caliber — all three entry points (`pipeline.py`/
`__main__.py`/`dump_graph.py`) call it uniformly, ensuring consistent simulation and dump calibers:

- **Isolation guarantee through stream grouping**: The split key includes streamId, so **cross-stream tasks are never merged into the same batch** (one doorbell
  inherently belongs to only one stream; a single doorbell batch is not split by the key); the group number (doorbell batch number) is unique within a 255 window
  — doorbells from other operators are not visible in the current operator's task list but also consume batch numbers, so **same stream and same number must be
  the same doorbell**, preventing cross-doorbell mis-merging (safe for multi-operator dbs).
- **Descriptive information** (type/address/endpoint/notifyId/reduction parameters/profInfo) takes the **last** data
  task within the batch, **with length as the full batch cumulative sum** — consistent field-by-field with `ExecProfilingAll(loc[insNum-1], rmt[insNum-1],
  totalSize, transferOp[insNum-1])`; the trace/dump display name similarly takes the last task's
  op name (Section 11.4.1). When a batch mixes write/read directions or different dataTypes, it likewise follows the last task
  (a single SQE batch is usually homogeneous; mixing is a rare scenario).
- **Not participating in merge**: notify/SDMA/synchronization tasks (notifyId is unique, pairing depends on order; local tasks
  are preserved as-is and do not affect batch boundaries); unmarked/no-group-number tasks (`sqeGroupId <= 0`: 0 = unmarked,
  -1 = v3 and earlier BLOBs without this field) are preserved as-is and **cut the current batch** (batch attribution cannot be determined, conservatively
  ending the current batch); UDMA non-data tasks (inline write notify) participate only in batch splitting, and interleaved entries within a batch are placed
  after the merged task (notify causality).
- The merged task occupies the slot of the first data task in the batch (batch head). The split key including stream ensures that tasks within the batch share the
  same stream and same jetty. The stream serial chain (Section 5.3) and notify pairing (Section 5.4) relative order is not
  affected; local (non-UDMA) tasks do not participate in merging and do not reset batch boundaries (preserved in place, not affecting batch continuity).

### 5.1.2 DPU Sequence Merge (Pre-Graph-Construction Preprocessing)

DPU (host CPU RoCE) path large message WriteWithNotify sequence merge, executed after 5.1.1
(`pipeline.py` unified call). Rules and semantics are described in Section 11.4.6: scan the DPU stream order for
MEM_CPY chains with **consecutive src/dst dual addresses** (only data tasks with `src_type==1` from DPU source participate;
other tasks are preserved as-is without cutting the chain), terminated and merged into a single entry by an address-consecutive WRITE_WITH_NOTIFY —
**the terminating task is a natural terminator**, and two independent WriteWithNotify calls cannot be incorrectly merged (the tail block of each call
necessarily closes the current chain). The only theoretical mis-merge scenario is "an independent pure write whose address happens to be contiguous with the WriteWithNotify start address"
(semantic equivalence to one continuous data segment + terminating notification; simulation results are identical). Chains without a terminator are not merged
(independent pure writes remain per-entry `RDMASend`); the merged task occupies the terminator task slot (notify causality), and merged front blocks are removed from their original positions while unmerged tasks maintain the original stream order.

**Single-rank graph construction strategy (design premise, primary difference from the checker)**: The db contains only single-rank task data.
The DependencyGraph describes only the tasks and dependencies for that rank. **Cross-rank pairing edges (post-wait relationships)
are all not built (task nodes are preserved, approximated with fixed remote latency)**:

- SQE level: RECORD with waitDev ≠ local device (cross-rank post), WAIT with recordDev ≠ local device (cross-rank wait)
  **preserve the task node** — graph-constructed normally in stream order (src≠dst expresses remote, using fixed remote latency), but **no pairing edge is established**
  — only post-wait pairing within a single rank is considered (Section 5.4).
- CCU level: instructions that post to a remote Event, waits that depend on remote Event being set **also generate task nodes**
  (Notify_Record/Notify_Wait, src=local, dst=INVALID → fixed remote latency), but no cross-rank pairing edge is established
  (Sections 6.4/6.6).
- **Cross-rank data transfers are preserved** (MEMCPY with direction=write/read): they are the main source of network
  bandwidth usage from the local rank perspective; only synchronization dependency edges are ignored, not the transfers themselves.
- makespan semantics: the operator end-to-end duration from a single-rank perspective = local computation/copy + locally initiated network transfers
  + local synchronization waits; when the remote side is ready is not modeled (equivalent to assuming the remote side is immediately ready).

**Simulation granularity**: Default is per-operator (one DependencyGraph per `(commId, opIter)`, one simulation, one record),
consistent with the checker's per-operator verification caliber; `--all-in-one` optional full-volume mode (multiple operators on the same stream are naturally serial).

### 5.2 taskType → hccl_sim Task Mapping

Aligned with the HCCLSIM new task model (`TaskType`: `NOTIFY_RECORD`/`NOTIFY_WAIT`/`REDUCE_INLINE`/
`MEMCPY`/`UB_INLINE_WRITE`/`UB_WRITE_OR_READ`/`WRITE_WITH_NOTIFY`; for details, refer to Section 7.3):

| db taskType | Condition | hccl_sim TaskType | Task Parameters | Latency Model |
|-------------|-----------|-------------------|-----------------|---------------|
| NOTIFY_RECORD (1) | waitRank == local rank | `NOTIFY_RECORD` | notify_id=int, src=dst=local_rank | Fixed + triggers Event |
| NOTIFY_RECORD (1) | waitRank != local rank | `NOTIFY_RECORD` (cross-rank post: node preserved, not paired) | notify_id=int, src=local_rank, dst=peer rank (INVALID if unresolved) | Fixed remote_record_delay, does not trigger Event |
| NOTIFY_WAIT (0) | recordRank == local rank | `NOTIFY_WAIT` | Same as above | Waits for Event |
| NOTIFY_WAIT (0) | recordRank != local rank | `NOTIFY_WAIT` (cross-rank wait: node preserved, not paired) | notify_id=int, src=peer rank (INVALID if unresolved), dst=local_rank | Fixed remote_wait_delay |
| MEM_CPY (3) | jettyId == INVALID | `MEMCPY` | size=len, src/dst_rank=taskData endpoint rank (falls back to local if unfilled) | HBM dynamic shared bandwidth |
| MEM_CPY (3) | jettyId valid, write direction | `UB_WRITE_OR_READ` | size=len, src_rank=local, dst_rank=peer rank, src/dst_eid, jetty_id, protocol, is_write=True | Port write direction bandwidth sharing |
| MEM_CPY (3) | jettyId valid, read direction | `UB_WRITE_OR_READ` | size=len, src_rank=peer rank, dst_rank=local, src/dst_eid, jetty_id, protocol, is_write=False | Port read direction bandwidth sharing |
| REDUCE (2) | jettyId == INVALID | `REDUCE_INLINE` | size=dataCount, data_type=task-level dataType enum name, src/dst_rank same as MEMCPY local branch | HBM dynamic shared bandwidth |
| REDUCE (2) | jettyId valid | Same as MEM_CPY network branch (Section 5.7 approximation) | Same as above (including data_type) | Same as above |
| WRITE_WITH_NOTIFY (7) | Always write direction (local src → remote dst, no direction inference) | `WRITE_WITH_NOTIFY` (**single task, not split**) | size=len, notify_id=notifyId, src_rank=local, dst_rank=peer rank (falls back through eid→rank index if unresolved), src/dst_eid, jetty_id, protocol, is_write=True; when isReduce=1, data_type=dataType enum name | Port write direction bandwidth sharing + fixed record delay for notify after write lands (remote) |
| CCU_GRAPH (4) | — | (Expanded, Chapter 6) | — | — |
| AIV_GRAPH (5) | — | Skipped | — | Warning |
| SYNC_STREAM (6) | — | **Skipped** (host-side stream sync point, Section 5.6) | — | — |

> waitRank/recordRank are the taskData endpoint rankIds (carried in the deviceId field, src=record side,
> dst=wait side, Sections 4.4.1/4.6), used only for **determining local/cross-rank**: cross-rank notify tasks
> preserve nodes during the mapping phase but do not enter pairing (Section 5.1 strategy);
> pairing occurs only among local notifies. The local/remote distinction for notify in the new model is expressed as `src_rank == dst_rank`
> (`is_local_notify`).
> **WRITE_WITH_NOTIFY (7) single-task mapping** (one-to-one correspondence with db tasks, not split into write + NOTIFY_RECORD
> two sub-tasks): the write direction is always write (launching-end convention: src local → dst peer, no Section 5.5 direction inference),
> using port write direction bandwidth sharing; after write data lands, a notify is sent to the peer, charged at a fixed record delay
> (the peer wait is not in this db, per the single-rank simulation premise, same semantics as cross-rank record; self-write src==dst edge
> case uses local record delay); `notifyId` is preserved with the task (displayed in trace/dump),
> and `isReduce=1` only affects the `data_type` display (same enum name mapping as the REDUCE branch).
> **Network tasks directly consume taskData rank** (deviceId field carries rankId; launching-end convention:
> write=src local/dst peer, read=src peer/dst local): write fills `dst_rank=peer rank`,
> read fills `src_rank=peer rank`; when the peer rank is unresolved (`0xFFFFFFFF` or equal to local), falls back to INVALID
> (Section 10.1). The direction is explicitly carried as `is_write`, and the simulator no longer depends on the `src_rank==0` heuristic
> (Section 10.3 #1 fixed; the task.txt old input path still uses the heuristic).
> The local eid preferentially uses the SQE explicit lcEid (added in v2 collection), falling back to jettyId → ccuChannels channel
> leid derivation when unavailable (Section 5.5, Section 10.1).
> The TaskType in this table is for internal latency model classification; the task display name in trace/dump has been switched to
> the profInfo.profTaskType op name (Section 11.4.1 mapping table, Section 10.3 #18 implemented),
> and the correspondence with the TaskType in this table follows the "Simulation TaskType" column in Section 11.4.1.

### 5.3 Stream Serial Chain Construction

- Group by `(deviceId, streamId)`, and within each group, order by ascending `global_id` (= SQE submission order) to form a chain.
- The simulator starts one simpy process per chain for serial execution (Chapter 7) — **the checker's mainStart virtual
  start point and main stream special handling are not needed**: simulation focuses on end-to-end makespan, and each chain starts in parallel from time 0;
  the WAIT at the head of a stream naturally forms the actual synchronization with the main stream.

### 5.4 Notify Pairing Algorithm (Local Only)

**All cross-rank post-wait edges are ignored** (Section 5.1 strategy): before pairing, classify by the Section 4.6 peer backfill results.
Cross-rank notify tasks (RECORD with waitDev ≠ local device, WAIT with recordDev ≠ local device) preserve nodes during the Section 5.2
mapping phase (graph-constructed in stream order) but do not enter pairing. Therefore, the pairing key is simplified from the checker's triple to a single notifyId:

```
checker semantics:   pairing key = (recordDeviceId, waitDeviceId, notifyId)   ← multi-rank
this plugin:         filter cross-rank first → pairing key = notifyId         ← single-rank
```

Consistent in principle with the checker's `AddLocalNotifyEdges` (submission order simulation + first-come-first-served consumption,
`task_graph_generator_v3.cc:847-946`), light_performance simplifies to
**two-pass scan + first-come-first-served**:

```
Input: All local notify tasks for an operator (cross-rank nodes preserved but not participating in pairing)
First pass: Scan in ascending global_id order
    RECORD → pending_records[notifyId].append(task_id)                        # queue
    WAIT   → if pending_records[notifyId] is not empty:
                  rec = pop from front; dep_graph.add_match(wait_id, rec_id)   # first-come-first-served
              else:
                  unmatched_waits.append(wait_id)                              # wait for "later" records
Second pass: Process unmatched_waits (cases where WAIT appears before RECORD)
    Replay in global_id order: unmatched WAIT is suspended; when RECORD arrives, pair with the earliest suspended WAIT
    (successfully paired RECORD is simultaneously removed from pending_records, ensuring diagnostic statistics are accurate)
Cleanup: Remaining WAITs → normal (see below); remaining RECORDs → warning, should not appear in normal data
```

Description:

- The db id order is the submission order, and the stream serial chain already ensures "successor tasks on the same stream after a WAIT do not execute first",
  so the checker's complete queue simulation and deadlock detection are not needed (deadlocks surface as makespan anomalies/unexecuted tasks in simulation).
- Multiple RECORDs/WAITs with the same notifyId (for example, n11/n15 repeatedly appearing on stream 6 in early 169B samples, main stream over two iterations)
  are naturally paired by the first-come-first-served rule; **notifyId reuse is allowed**.
- **Unpaired local WAIT is normal semantics**: in the v2 AICPU sample (Section 4.7), notifyId is reused across ranks —
  local `RECORD(nid=30, dst=dev2)` (wait side is on the peer, node preserved but not paired) and local `WAIT(nid=30)`
  (record side is on the peer, not in this db) appear in pairs; this WAIT is charged at the fixed `remote_wait_delay` delay in simulation
  (Section 7.6), and the warning is only informational.
- Cross-rank pairs in early 169B samples: `RECORD(n18, s1→d2)` (notifying dev2) and `WAIT(n18)` (backfilled recordDev=2)
  both preserve nodes, and no cross-rank pairing edge is built: graph-constructed in stream order, `src≠dst` uses fixed remote latency;
  data transfer tasks on stream 8 (MEMCPY(jet=3), and so on) are not affected.

### 5.5 Network Task Direction and eid Resolution

**Direction determination** — network data tasks are uniformly mapped to `UB_WRITE_OR_READ`. The direction determines the port charging
direction (is_write) and rank endpoint values. The taskData endpoint rankId is carried by the deviceId field
(Section 4.4.1 rank field rollback), with launching-end convention (`device_sqe_parse_stub.cc`):
write = src local/dst peer, read = src peer/dst local:

| Priority | Rule | Basis |
|----------|------|-------|
| 1 | srcRank == local and dstRank ≠ local → write; dstRank == local and srcRank ≠ local → read | Collection endpoint rankId (carried in deviceId field, Section 4.4.1); measured samples (Section 4.7) are all determined by this rule, no inference |
| 2 | When rank does not distinguish (equal or both unfilled), use memory range: dst falls within local output range → read; src falls within local input range → write | Measured sample induction (Section 4.7): fails when task offset does not intersect with opMemInfo range |
| 3 | Fallback: default write, diagnostic statistics `direction_inferred` count | — |

> Network tasks explicitly carry `is_write` on the Task (Section 10.3 #1 fixed; CCU path Section 6.7 synchronously fixed).
> The simulator uses this to select the write/read charging direction and the port query eid (`src_eid`/`dst_eid`), no longer depending on
> the `src_rank==0` heuristic; the task.txt old input path does not have this field and falls back to heuristic determination.
> `WRITE_WITH_NOTIFY` (7) **does not participate in direction determination**: semantics are always write (local src → remote dst),
> and `is_write=True` is set directly during graph construction.

**eid resolution** — network tasks look up the ranktable through eid to get physical ports (`TopoResolver.get_ports_by_eid`):

```
SQE:        Local eid preferentially uses SQE explicit lcEid (added in v2); falls back through jettyId
            ──ccuChannels(local device outbound channel)──> leid(local eid)
            Peer preferentially uses channel reid, falls back to SQE rmEid; when channel is missing, local eid degrades to any local channel leid
CCU microcode: channelId → channels[(dev,die,channelId)] → leid/reid + jettyId list
            When channel is missing, local eid preferentially uses the mission SQE lcEid, degrades to any channel leid on the same die
write direction: simulator queries task.src_eid → ports; read direction queries task.dst_eid → ports
```

- `protocol`: channel protocol column → HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO (CommProtocol name,
  display only); `link_type`/`transport_type`: task profInfo (DfxLinkType/
  ProfTransportType name, Sections 11.4.2/11.4.3, uniformly filled by `_add_task`, reconciled with msprof L1).
- When the local eid cannot be resolved, a warning is issued + `eid_unresolved` count; when network tasks exist but no ranktable is available,
  simulation is skipped entirely (TopoResolver throws an error directly for missing eids; for details, refer to Section 10.3).

**Peer rank resolution** — For SQE network tasks, the peer rank (`_peer_rank`): the taskData endpoint rankId
(carried in the deviceId field, already resolved by the launching end through `GetRmtRankIdByEid`) takes priority; when unresolved
(UINT32_MAX/local value), falls back through the eid→rank index (Section 4.8) by peer eid; if still unavailable,
INVALID + `rank_unresolved` count.

### 5.6 SYNC_STREAM Handling

Host-side stream synchronization point (appears once per iteration on the sample main stream), indicating that the host waits for the current stream to complete.
**Does not represent a device-side task**: no task node is generated, no time is consumed (makespan is the end time of the last task in the stream chain).

- Diagnostic statistics `skipped_sync` count;
- Previously approximated by "unpaired local `NOTIFY_RECORD` fixed delay placeholder", which introduced a spurious
  `local_record_delay` at the stream tail (and appeared in trace as a pseudo notify task with notify_id=invalid); this has been removed.

### 5.7 REDUCE Approximation

Reduction = network transfer + computation. The current model approximates with data transfer:

- Network (jettyId valid): charged as `UB_WRITE_OR_READ` for port bandwidth delay
  (direction determined as in Section 5.5).
- Non-network (jettyId INVALID): mapped to `REDUCE_INLINE`, charged by HBM bandwidth dynamic sharing
  (`size / hbm_bandwidth`); computation delay is not separately listed yet (can be calibrated by `dataType×dataCount` in the future,
  for the data_type field gap, refer to Section 10.1).

### 5.8 Comparison with Checker Graph Construction and Simplifications

| checker Step | light_performance | Rationale |
|--------------|-------------------|-----------|
| mainStart virtual node + main stream entry edge special handling | Removed | Simulation only needs makespan; the WAIT at chain head already expresses synchronization |
| AddLocalNotifyEdges queue simulation + deadlock detection (GRAPH_DEADLOCK) | Two-pass scan first-come-first-served, no deadlock detection | No verification requirement; anomalies surface as simulation results |
| AddInterRankNotifyEdges cross-rank pairing (full-graph BFS simulation) | **Completely removed**: all cross-rank post-wait pairing edges ignored (notify task nodes preserved, graph-constructed in stream order) | Single-rank graph construction (core difference) |
| MemSlice address resolution + memory layout table | Range attribution for direction determination only (optional) | Simulation only needs len, not a precise memory model |
| CheckSlaveTaskQueue stream structure verification | Removed | Not a verification tool |
| Memory conflict/semantic verification/sync conflict/reachability/single-task check | All removed | Not a verification tool |
| CompactSyncNodes (SYNC_STREAM merging) | Skip, no node generated (host-side sync point, not device-side task) | makespan is the end time of the last task in the stream chain (Section 5.6) |
| AIV subgraph expansion | Not supported (skip with warning) | Requirement explicitly states not considered at this time |
| CCU subgraph: queue parallelization, loop merge, resource conflict verification | Queue parallelization simplified implementation (Section 7.4.1: assembly-phase skeleton/unit splitting + fork/completion-set join and attach extra edges + `#v` virtual streams); loop merge equivalently implemented (Section 6.5); resource conflict verification removed | Parallelization eliminates queue serialization distortion (Section 7.4.1); no verification needed, node count is manageable |

### 5.9 Output Assembly

```python
def build(group: OpExecutionGroup, sim_config: SimConfig | None = None
          ) -> tuple[SimConfig, DependencyGraph]:
    dep = DependencyGraph()
    # ① SQE → Task (CCU_GRAPH temporarily stored as placeholder)
    # ② stream chain (int stream id)
    # ③ notify pairing (add_match)
    # ④ direction/eid completion
    # ⑤ CCU expansion (ccu_graph_builder.expand_ccu_graph(..., local_rank, alloc_stream))
    return sim_config, dep
```

- `SimConfig` source: JSON specified by `--config` / manifest setting (Section 7.2 format, units
  us/GBps), overlaid with bandwidth utilization configuration.
- Topology: `TopoResolver` is constructed by the CLI layer (ranktable location described in Section 3.3) and passed to the simulator.
- The output can immediately be sent to `HcclSimulator(config, dep_graph, topo).run()` (returns dict, as described in Section 7.1).

---

## 6. CCU Mode Task Graph Generation (Sub-Requirement 3, Focus of This Iteration)

### 6.1 CCU Data Organization in db

```
SQE with taskType=4 (CCU_GRAPH) in opTask_P_<pid>
    └─ taskData.ccu: {dieId, missionId, timeout, instStartId, instCnt, key, argSize, args[13]}
           │  instStartId/instCnt define the instruction execution window [instStartId, instStartId+instCnt)
           │  args[13] are consumed one by one by LOAD-type instructions (every 13 args consumed advances sqeIndex)
           ▼
ccuInstrRes (deviceId, dieId) → instrSpace: 32K × 32B microcode instruction space
ccuChannels (deviceId, dieId, channelId) → {dstDeviceId, dstDieId, leid/reid, jettyId[]}
    └─ Channel register values → peer location and jetty
```

- SQEs are grouped by `(dieId, missionId, queue)` (single-device graph construction, deviceId is constant; queue is the SQE stream)
  into dispatch contexts (the checker's `CcuMissionKey` + latest-node-append semantics,
  `task_meta_translator_v3.h:47-62`): SQEs with the same key are merged into the same context only when **contiguous with the previous one** (previous
  `instCnt==13` and `instStartId` connects end-to-end, `IsContiguousCcuSqe`,
  `task_meta_translator_v3.cc:213-231`); otherwise **a new context is created** —
  multiple dispatches of the same mission (instruction window wrap-around re-delivery, typically "head SQE (13 Load
  SqeArgs) + body SQE" sent twice in succession) each expand independently: each context fully replays its own instruction window
  (including the Load SqeArgs header, reinitializing loop counters and other registers with **this dispatch's args**),
  avoiding the case where the later dispatch uses residual register state from the previous one, causing data transfers to be lost (Section 10.3 #20).
  - **args consumption** (checker `CcuGraphStateV3::GetSqe` semantics): SQEs within a context are ordered by dispatch sequence,
    and args are consumed sequentially per slot — every 13 args consumed (reaching the last slot `sqeArgsId==12`) advances the cursor
    to the next SQE; in the typical layout, the 13 Load SqeArgs instructions in the head SQE consume head SQE args[0..12],
    and subsequent Load instructions within the body SQE window consume body SQE args.
  - **Register/Event bitmap shared globally across contexts** (checker `AllRankParamRecorder::Global()`
    semantics): waits from a later dispatch can be resolved by Event bits/residual post accounting left by an earlier dispatch,
    consistent with the checker's multi-graph shared recorder behavior.
  - **Dispatch contexts within the same queue advance strictly in dispatch order** (checker queue semantics: a single
    queue strictly follows instruction order; a later TaskCcuGraph on the same stream must wait for the previous one to complete); cross-queue polling
    advances jointly (Section 6.6).
- Subgraph attachment point: The position of the CCU_GRAPH SQE in the stream chain is the subgraph entry; after expansion, the placeholder is removed
  (no placeholder node is introduced in the task graph), and the subgraph attaches to SQE predecessor/successor tasks through extra edges (Section 7.4.1).

### 6.2 Microcode Instruction Format (32B Fixed Length)

Same source as `${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode`
(reference copy in this repository: `ai_skills/skills/convert-ccu-microcode/references/ccu_microcode_v1.h`):

```
CcuInstr = CcuInstrHeader (2B) + union { V1(A5) | V2(A6) | V3 } (30B)

CcuInstrHeader bit field (little-endian u16):
    code     : 11 bit   (header & 0x7FF)      Instruction code
    type     :  4 bit   ((header >> 11) & 0xF) Instruction category
    reserved :  1 bit

Instruction categories: LOAD_TYPE=0x0 | CTRL_TYPE=0x1 | TRANS_TYPE=0x2 | REDUCE_TYPE=0x3
```

Chip generation → instruction set version mapping (checker `GetCcuInstrVersion`, `ccu_task_transform_v3.cc:175-184`):
`Ascend950 → A5/V1`, `Ascend960 → A6/V2`, other devTypes fall back to A5; derived from Device soc_version
(hccl_sim.db) or `opDetails.devType` (`expand_ccu_graph` selects `_MissionContext`/`_MissionContextA6` interpreter by `dev_type`). **Both instruction set generations are implemented** (A6 ported from
checker `ccu_task_transform_instruct_a6_v3.cc`, Section 11.3.3); for an A5/A6 instruction set difference overview, refer to Section 6.8.

### 6.3 Instruction Type → Task Mapping

#### 6.3.1 Summary Table

| Instruction Category | Representative Instructions | light_performance Behavior | Produced Task |
|---------------------|------------------------------|---------------------------|---------------|
| LOAD (register write) | `LoadSqeArgsToXn/GSA`, LoadImd, LoadX, arithmetic-logic (A6 Add/Sub/...) | Updates register model, data plane **does not produce nodes**; A6 LOAD instructions generally carry a `setCKE` tail field (non-zero mask → local `NOTIFY_RECORD`, checker `ProcessSetMask` semantics); A6 Load/Store (HBM↔Variable data plane) does not model HBM content, registers are not backfilled (only `setCKE` synchronization semantics preserved, warning logged) | —(Notify_Record when A6 `setCKE` is non-zero) |
| CTRL: `SetCKE`/`SetCKBit` | Set Event bit | Local → `NOTIFY_RECORD` (register Event accounting, with mask); remote target (cross-rank post, `SyncCKE` and so on) → remote `NOTIFY_RECORD` (not paired, Section 6.6) | Notify_Record |
| CTRL: Wait/Jmp/Fence | Wait for Event bit / jump / fence | Event wait (A5 embedded `waitCKE` field): if bits insufficient, pause the queue pointer; if bits satisfied, generate `NOTIFY_WAIT` (split by contributing bits, with mask); waits depending on remote bits → remote `NOTIFY_WAIT` (not paired, fixed delay, Section 6.6). A6 independent Wait instruction (Variable value comparison spin, checker `TransformWaitInstr`): satisfied → pass through with no node; unsatisfied = depends on peer remote register write (`SyncWtX`/TransMem notify) → top-level stall resolution generates remote `NOTIFY_WAIT` (fixed `remote_wait_delay`), within loop body generate node directly and continue. Jmp: A5 evaluates by `Xn[conditionXnId] != expectData` (checker `TransformJmpInstrA5` semantics); A6 supports 6 condition codes (==/!=/>/>=/</<= , ≥6 unconditional) + relative/absolute two modes (jumpMode, target `Xn[relTarInstrXnId]`); target limited to current instruction window, out-of-range warning pass-through; pass-through within loop body; step budget defense against backward jump infinite loops. Fence (A6 only): checker does not support, warning pass-through (queue serial model naturally guarantees order) | Notify_Wait |
| CTRL: `ClearCKE` | wait + clear bit | wait semantics; A6 follows checker `ClearWaitMask` semantics to clear the consumed wait bits (`clearMask != 0` is a checker unsupported scenario, warning + tolerant clear of target bits) | Notify_Wait |
| CTRL: Loop/LoopGroup | Loop | loopCnt from register, **merge expansion** (Section 6.5, falls back to full expansion on failure); A6 Loop `mode==1` is conditional expansion (`wishCKEBit == Xp.loopCtxId + 1024` hit to execute, always takes full expansion path) | None (body instructions mapped per this table; under merge path, data task size × total iterations) |
| TRANS (transfer) | LocMem↔LocMS/RmtMem/RmtMS (A5), local 4 codes + unified TransMem dmaOpCode (A6) | Mapped by role (Section 6.3.2) | `Memcpy`/`Ub_Write_Or_Read` |
| TRANS: Sync (cross-rank synchronization) | `SyncCKE/GSA/Xn` (A5), `SyncXnWtx`/`SyncAtx` (A6) | Remote set/notify → remote `NOTIFY_RECORD` (not paired, Section 6.6); value synchronization is remote register write, no node built | Notify_Record (remote) |
| REDUCE | Add/Max/Min | Same as SQE REDUCE approximation (Section 5.7); CcuBuffer is rank-local buffer, always local HBM reduction | `Reduce_Inline` |

#### 6.3.2 TRANS Transfer Role Determination (Consistent with checker `ccu_loop_merge_v3.cc:73-144`)

By src/dst device attribution:

| src | dst | Role | hccl_sim Task |
|-----|-----|------|---------------|
| Local | Local | LOCAL_COPY | `Memcpy` (HBM dynamic sharing) |
| Local | Remote | WRITE | `Ub_Write_Or_Read` (src_rank=local, dst_rank=peer, is_write=True) |
| Remote | Local | READ | `Ub_Write_Or_Read` (src_rank=peer, dst_rank=local, is_write=False) |

> Cross-die/card read/write in both directions is uniformly mapped to `Ub_Write_Or_Read`, with direction expressed through `is_write`/trace
> `direction` field. Ports are charged independently for write/read directions; the peer rank is resolved through channel
> reid → eid→rank index (INVALID when unresolved). To preserve microcode-distinguishable buffer
> information, CCU-converted transfer tasks carry `src_buf_type`/`dst_buf_type` (mem/ms, determined by
> instruction source/destination operand); SQE/CLI path same-name tasks do not have this information and output invalid value N/A.

> Cross-rank **data transfers** (WRITE/READ roles) are preserved — they are the main source of network bandwidth usage;
> only cross-rank **synchronization edges** (post-wait, see Sections 6.4/6.6) are ignored.

Key field sources:

- `size`: length register (A5 `lengthXnId` direct read / A6 `xlId` through `GetXnId` — when bit15=1,
  the register number is offset by loop `xnIdOffset`) → look up register model.
- Peer location: A6 channel through **register value** (`Xn[GetXnId(xcId)]` value is channelId) →
  `channels[(dev,die,channelId)]` → `{dstDeviceId, dstDieId}`; A5 channel is instruction
  immediate field. A5 MS-type instructions `msId / 0x8000` = rank, `msId % 0x8000` = MS offset (MS page 4KB) — **only for
  addressing encoding**: MS is a rank-directly-accessible local buffer (checker `GenSliceFromMs` +
  `AddLocalBatchReduce` treats all msIds as always local); the msId high bits are **not used as** local/remote task type
  criteria; cross-rank data movement is expressed by Rmt* role instructions (through channels). A6 TransMem's
  `msIdMode==1` marks the MS end (write source / read destination is local MS,
  `src/dst_buf_type=ms`).
- eid: channel leid/reid (hex string); local eid preferentially uses the mission SQE `lcEid`
  (added in v2), falls back to same-die channel leid approximation.
- A6 `TransMem` `dmaOpCode` (implemented, checker `TransformTransMemInstr`):
  3=write, 5=write with notify (notify value written to remote Variable/Event, target type unresolvable in single rank →
  approximated as remote record), 6=read, 0x1A=remote atomic count (no data plane, approximated as remote record),
  0x19 (compressed write+atomic) and others unsupported (warning, skip data plane; `setCKE` local
  post still executes to maintain synchronization skeleton); `udfEnable` (network reduction) charged as ordinary UB transfer
  (Section 5.7 approximation, peer reduction not modeled); `udfType != 0` warning, skip data plane.

### 6.4 Execution Model: Register + Event Accounting (register_model.py)

Python simplification of the checker's `AllRankParamRecorder` (`ccu_all_rank_param_recorder_v3.h:27-110`):

```python
class CcuRegisterModel:
    xn:    dict[(dev, die, id)] -> int      # General registers (length/loop counter and so on)
    cke:   dict[(dev, die, ckeId)] -> int   # CKE bitmap (set/clear)
    gsa:   dict[(dev, die, id)] -> int      # A5 GSA registers (A6 has no GSA, only xn/cke)
    seen_post: dict[key, list[PostRecord]]  # CKE post accounting (for pairing)

    key = (waitDeviceId, dieId, ckeId)
    PostRecord = {record_device_id, cke_mask, remaining_mask, task_id, is_local}
```

- LOAD/arithmetic-logic instructions "execute" on this model (read/write registers); uninitialized reads → warning and treated as 0
  (the checker reports error and exits; here, tolerant handling is used to accommodate incomplete data). A6 arithmetic-logic instructions
  (Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt) execute with 64-bit wrap-around semantics (Mul takes
  lower 32-bit product, shift amount truncated to 0x3F, arithmetic shift interprets as signed); LoadX/StoreX
  indirect addressing target register number out of range (>16383) warning, skip write (checker reports error and aborts).
- **A6 in-loop register number offset** (checker `GetXnId`/`UpdateXnId`): when the register number
  field bit15=1 in the payload, within the expansion region (`curLoopIdx >= expandOffset`), offset the register
  number by `xnIdOffset × curExpandCnt`; when bit15=0, use the original number directly. Event/CcuBuffer id
  loop offset follows the same rule across both generations (`UpdateId`).
- **Event pairing rules** (consistent with checker `GenWaitNode`, `ccu_task_transform_v3.cc:56-121`):
  ```
  post matches wait ⇔ post.waitDeviceId == wait.waitDeviceId
                   && post.dieId == wait.dieId
                   && post.ckeId == wait.ckeId
                   && (post.ckeMask & wait.ckeMask) != 0        # bit intersection
  After match: one wait instruction aggregates into at most 1 local wait node (mask = OR of all contributing
            post bits) + 1 remote wait node (bits depending on remote set)
            post.remaining_mask &= ~wait.ckeMask                  # one post's bits can be consumed by multiple waits in parts
            emptied → remove from seen_post
  Node dependencies: aggregated wait node builds edges with all contributing posts — first contributing post uses
            notify pairing (add_match, Event semantics), remaining through extra edges
            (child=wait, parent=post, "wait for all parents to complete" pure timing constraint)
  ```
- **Event accounting registers only local rank posts** (single-rank graph construction): cross-rank posts generate remote
  `Notify_Record` nodes (src=local, dst=INVALID → fixed remote delay) but do not register
  `seen_post` (the set action occurs on the peer rank's Event); wait bits depending on remote set generate
  remote `Notify_Wait` nodes (fixed delay, not paired), as described in Section 6.6.
- When bits are not satisfied, the queue pointer pauses (waiting for posts from other queues within the local rank to unlock) — multi-queue joint advancement
  is described in Section 6.6 algorithm; when multiple dispatch contexts exist within the same queue (same mission, multiple dispatches, Section 6.1),
  the later context must wait for the earlier context to complete before advancing.
- Final verification: `seen_post` residuals → warning and counted in diagnostic statistics (checker reports GRAPH_UNMATCHED, here downgraded).

### 6.5 Queue Model and Loop Handling

- **queue**: `CcuSqeParam` groups by the SQE's stream (checker uses ccuParams[queueId]),
  each queue forming one serial chain. The new task model `stream_id` is **int**:
  SQE streams use the original streamId directly; **CCU queue chains also belong to their SQE original stream** — whichever stream a CCU-type SQE
  is on, the expanded skeleton tasks are on that stream (`queue` is the original stream number,
  `alloc_stream("ccu_d{die}_q{queue}")` resolves back to the original stream number through `ccu_queue_origin_stream`,
  same-name reuse merges multiple missions within the same queue); only virtual streams newly created by queue parallelization are allocated sequentially from base
  `1<<20`; logical names are preserved in `stream_labels` for dump/report display.
  Exception: when the same original stream is occupied by a queue on **another die**, the latter falls back to base allocation of a synthetic stream with a warning
  (chaining would serialize cross-die parallel execution); in this case, the stream's kind=ccu in `stream_meta`.
  Instructions within the same queue are strictly ordered (checker semantics: `ccu_task_transform_v3.cc:189`,
  "a single queue strictly advances in instruction order; cross-queue coordination is expressed only through generated wait/post edges").
- **Assembly-phase queue parallelization splitting** (Section 7.4.1): after interpretive execution completes, each CCU queue chain is further split into
  skeleton (wait / floating set / remote synchronization nodes — floating set refers to record nodes not immediately following data tasks)
  and asynchronous data transfer units two categories:
  skeleton stays on the queue serial chain (= SQE original stream chain, pieced back to the placeholder position in dispatch order) to express the timing framework,
  while asynchronous units are reattached to virtual streams
  `ccu_d{die}_q{queue}#v{n}` (sequentially allocated from base `1<<20`), reused across missions,
  not reused within a mission, connected to the skeleton through fork extra edges — the serial chain constrains only the skeleton and unit internals,
  eliminating the "entire queue per-task serial" distortion.
- **loop expansion** (merge-first, aligned with checker `ccu_loop_merge_v3.cc`):
  - The iteration count for `Loop`/`LoopGroup` instructions comes from registers, known at interpretive execution time:
    - A5: LoopGroup parameters are in Variable (LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41])
      and Xm (gsaOffset/MSOffset/ckeOffset); Loop iteration count is in `Xn[xmId]` lower 13 bits;
    - A6: LoopGroup parameters are carried by **Variable/Xm/Xp three registers** (checker
      `LoopGroupXnA6/XmA6/XpA6`) — Variable: `loopInsCnt:10 | expandOffset:9 |
      expandCnt:9`, Xm: `ckOffset:10 | msOffset:11 | gsaOffset:32`,
      Xp: `xnIdOffset:32` (GetXnId offset, Section 6.4); Loop iteration count is in
      `Xn[GetXnId(xmId)]` lower 13 bits, Address step size in `Xn[xnId]` lower 32 bits, context id
      in `Xn[xpId]` lower 9 bits; `mode==1` is **conditional Loop** — expand only when instruction
      `wishCKEBit == loopCtxId + 1024` (`A6_LOOP_BOARD_ENTRY`), otherwise
      this expansion is skipped (checker `ProcessLoopIns`), and conditional Loop always takes the full expansion path
      (checker `CollectLoopExpandA6` refuses merging).
    Each Loop within a LoopGroup first attempts the **merge path** (checker `TryProcessMergedLoopA5/A6`
    → `LoopIterationMerge` / `LoopExpandMerge` semantics); if verification fails, falls back to full iteration expansion.
  - **Merge precondition verification** (any failure triggers WARN fallback, same semantics as checker "Skip merged-loop
    optimization"):
    1. Loop body contains only mergeable instructions:
       - A5: simple TRANS / SYNC TRANS (`SyncCKE/GSA/Xn`) / REDUCE / `ClearCKE`
         (`CollectLoopCkeOpsA5` rules; `SetCKE`, LOAD, LOOP, JMP,
         TransMem with `setRmtCKEMask != 0`, `ClearCKE` with `clearMask != 0` are all rejected);
        - A6 (`CollectLoopCkeOpsA6` + `Collect*LoopInstrA6` rules): local TRANS
          4 codes / TransMem (`udfType==0` and `dmaOpCode ∈ {3,6}` — 5 write with notify
          triggers remote post per iteration, rejected; channel must be resolvable, `IsExistRemoteDieInfo`) /
          REDUCE (TransMem with `udfEnable` network reduction also folded by trans slot) /
          `ClearCKE` (`clearMask==0`); `SetCKBit`, LOAD, LOOP, JMP, Wait are all rejected;
    2. wait/set masks are **balanced** within the loop body (`ConsumeProducedMask`/`ProduceMask`:
       each wait's bits must be produced by earlier sets, and all produced bits are exhausted at sequence end);
    3. No cross-expansion Event/CcuBuffer resource conflicts (`CheckResourceConflict`: different expansions' Event/CcuBuffer
       IDs after ckOffset/msOffset offset must not be reused; A6 CcuBuffer conflict check covers
       local transfer/TransMem `msIdMode` CcuBuffer end and reduction msId lists);
    4. Transfer/reduction length > 0 (`ValidateLoopLen`) and expansion count × iteration count > 0;
    5. **A6 additional**: wait Event IDs within the loop body must not be reused (`CheckA6LoopWaitCkeIdUnique`,
       hardware constraint; checker reports error and aborts, here downgraded to fallback full expansion); each expansion's iteration count/
       length/channel values must be consistent (checker's per-expansion `loopCnt` equality + `InstrMerge`/
       `MemMerge` mergeability performance-side approximation — register numbers after xnIdOffset offset may
       point to different registers).
  - **Merge emission** (`EmitMergedLoopInstrsV3` semantics): the loop body is executed **once** with sample expansion
    (curExpandCnt=0), per slot: wait → data task → set → clear wait bits;
    data task size = single-iteration length × (expansion count × iteration count) (timing-equivalent folding, consistent with checker
    AddBatchTransMem/AddBatchReduce batch tasks); Event register
    bitmaps for other expansions are replayed with or/clear idempotency, ensuring post-loop register state is consistent with full expansion (only
    PostRecord task accounting is not duplicated). All tasks within the emission window (data tasks and folded
    notify tasks) carry `loop_cnt` (loop count after expansion = N+1 after N expansions) and
    `loop_iter_cnt` (single loop self-iteration count); total iterations = `loop_cnt` ×
    `loop_iter_cnt`, preserving pre-merge loop details for subsequent simulation use; straight-line instructions, fallback full expansion, and
    SQE/CLI path tasks have both fields as 0 (invalid, trace uniformly outputs N/A).
  - **Quantified effect** (REDUCE_SCATTER measured db, 128 expansions × 50 iterations × 13 instructions):
    task count 192,005 → 66 (10 data tasks, several local post/wait, 35 remote synchronization nodes
    — 21 remote posts + 14 remote waits, Section 6.6), graph construction + simulation end-to-end
    ~55 s → ~6 ms, makespan 70,113.650 us → 2,753.711 us (this section's merge eliminates
    128K task-level notify overhead and 64K small transfer per-task delays; final value after queue parallelization
    and remote synchronization node inclusion, Section 7.4.1).
  - A6 LoopGroup configuration verification (`loopInsCnt≤512`, `expandCnt≤511`,
    `expandOffset≤511` and `expandOffset≤loopInsCnt`, checker
    `TransformLoopGroupInstr`); when not satisfied, warning and skip the entire LoopGroup (defense against abnormal
    register values causing uncontrolled expansion); the "wait Event IDs within loop must not repeat" verification is described in the
    merge precondition #5 above (`ccu_task_transform_instruct_a6_v3.cc:1126-1146`).

### 6.5.1 Merged Loop Pipeline Concurrency Restoration (Loop Tag + Feature Chain Resolution)

**Problem — folded pseudo-serialization (merge is not timing-equivalent for notify structure)**: The Section 6.5 merge is
timing-equivalent folding for **data tasks** (size × total iterations; total time unchanged in bandwidth-limited scenarios), but not for Event
wait/set handshakes within the loop body. During real microcode execution, the N expansions of a LoopGroup are emitted concurrently, and each expansion
internally advances through M inner iterations in a pipelined manner (typically 4KB per iteration), where the loop body's
`TransMemToMs → setCKE → waitCKE → TransLocMsToRmtMem / TransLocMsToLocMem`
is **fine-grained per-iteration handshaking**: the k+1 iteration's HBM→MS prefetch (memcpy engine) overlaps with the k-th iteration's
network write + local copy-out (channel / memcpy engine), with steady-state throughput ≈ max(each phase total).
After merge folding, per-iteration handshaking degenerates into a **single pair** of wait/record nodes, with semantics becoming "all front-segment data
(× N × M iterations) completes → then back-segment can start" **full barrier** — pipeline concurrency is folded into serialization.

Taking the first loop template from `hccl_vm_install/data/hccl_vm_data.db` as an example (microcode 0–10:
`TransMemToLocMs(4KB, set 0:1) → waitCKE(0:1) → 7× TransLocMsToRmtMem(each 4KB,
set 0:1..0x40) → TransLocMsToLocMem(4KB, set 0x80) → waitCKE(0:ff)`; LoopGroup
128 expansions × 256 inner iterations = 32768 times × 4KB = 128MB/task):

```
Current graph (after 7.4.1 parallelization; AA = loopgroup predecessor task representative, that is,
                             the last node in the queue skeleton before the loop emission window):
  AA -> memcpy(mem->ms) -> record(0:1)                     ← prefetch unit (128MB, 95.95us)
  record(0:1) →[paired]→ wait(0:1) → 7× ub_write + memcpy(ms->mem)
                                    ↑ Full serialization: back-segment 8 units fork from wait(0:1),
                                      waiting for prefetch 128MB to fully complete through pairing edge
  Multiple { record(0:xx) / record(0x80) -> wait(0:ff) }    ← loop-end barrier (preserved)
```

- Real execution: steady state ≈ max(prefetch 95.95, 7-port write 2526.48) us;
- Current graph: serial 95.95 + 2526.48 us.
- **Quantification** (sample db ALLGATHER, measured before and after this section's fix): makespan **2657.52 →
  2561.54 us**. Eliminates 95.95 us serial prefetch phase; after prefetch and local copy-out run concurrently through HBM dynamic sharing,
  each takes 191.79 us (correct expression of pipeline steady state, Section 7.5), and the 7-port network write 2526.48 us becomes the sole
  critical path. Error upper bound: when two phases belong to different engines (HBM prefetch vs network write) and have comparable durations,
  loop body time is overestimated by nearly **2×** (serial t1+t2 vs steady-state max(t1,t2)); same-engine phases
  (such as pure HBM prefetch+copy-out) serialization error is absorbed by HBM dynamic sharing (total unchanged);
  error accumulates linearly with the number of merged loops.

**Correctness premise — merge verification invariant**: Section 6.5 merge precondition #2 (wait/set masks balanced within body)
ensures that within the merged loop body **all wait bits are produced by earlier sets within the body and exhausted at the end** — that is,
all Event dependencies within the merge emission window are intra-loop per-iteration pipeline handshakes, with no cross-loop
real data dependencies (otherwise verification fails and falls back to full expansion). Therefore, resolving "mid-body waits and their consumed
sets" does not break the loop's external synchronization: the **loop-end wait** (last wait within the emission window, after which
no same-loop data tasks exist) and the **mission completion set** (skeleton tail + all unit tails, Section 7.4.1) continue to
carry post-loop instructions / SQE successors / same-queue subsequent mission queue-level serial semantics.

#### Approach (Loop Tag + Feature Chain Identification + Deletion and Reconnection, Iterate to Convergence)

1. **Loop tag**: `_MissionContext` maintains the current emission window's `_loop_tag`, named
   `loopgroup_{i}_loop_{j}` (i = LoopGroup instruction execution sequence number within the dispatch context, starting from 0;
   j = cur_loop_idx within the group, starting from 0; when LoopGroup is nested within a loop body, each is assigned by emission window,
   not forming a hierarchy). Within the `_emit_merged_loop` window, bits are set, and `add_task` defaults to inheriting into the new field
   `Task.loop_tag` (same pattern as `loop_cnt`). **Only the merge path is tagged**; full expansion fallback /
   SQE / CLI / straight-line instructions are empty string (trace outputs N/A). The tag also serves to limit the resolution scope
   (preventing cross-loop / cross-mission mis-deletion) and for trace/dump tracing.
2. **Feature chain identification** (recorded during emission + determined before assembly, avoiding reverse lookup of graph results): `_gen_waits`
   records (wait_task_id → all contributing post task_ids) into `ctx.wait_posts` when pairing succeeds
   (aggregated wait's match + extra all contributors, Section 6.4). During the assembly phase
   (`expand_ccu_graph` assembly loop start, **before** skeleton/unit splitting), scan the ctx.nodes sequence.
   A feature chain match requires all of the following:
   - W is a local paired `Notify_Wait` with `loop_tag == T` (no remote waits exist within the merge window —
     verification #2 ensures all body wait bits are locally produced);
   - All contributing posts R* of W are `Notify_Record` with `loop_tag == T`, and each
     **immediately follows a same-tag data task M\*** in ctx.nodes (unit-tail record, that is, the folded producer
     task, such as memcpy folded from TransMemToMs; data tasks ⊆ `_QUEUE_DATA_TYPES`,
     not limited to Ub_Write_Or_Read — Memcpy/Reduce_Inline also exhibit this pseudo-serialization);
   - **After W (in ctx.nodes order), a same-tag data task exists** (mid-body determination; loop-end
     waits have no same-loop data tasks after them, naturally exempt; the next loop's data tasks have a different tag,
     also not misidentified);
   - R* does not immediately follow a non-data node (floating set / sync slot local set **does not match** — these
     wait for real dependencies such as remote synchronization, and resolution is prohibited).
3. **Deletion and reconnection** (pruning ctx.nodes, not post-assembly graph surgery):
   - **W always deleted**: removed from ctx.nodes and dep task table, its pairing edge (`remove_match`)
     and aggregated extra edges (`extra_edges[W]`) revoked — this is the essential operation to break the barrier;
   - **R\* depends on consumers**: if R does not appear in other waits' `wait_posts` values → delete along with
     (in the sample db, set24 is consumed only by wt25, directly deleted); otherwise **preserve R and delete only
     W** (its preserved edges are unaffected, loop-end waits continue as normal — timing-equivalent, semantically lossless
     conservative branch; taken when post bits are partially consumed by multiple waits);
   - **Successor unit reconnection happens automatically**: after pruning, the assembly logic rescans ctx.nodes, and units originally forked from W
     automatically fork to W's predecessor skeleton node (that is, the loopgroup predecessor task AA);
     when W is the skeleton head (no preceding skeleton), the unit becomes an entry unit, with entry dependencies expressed by SQE attachment edges
     (Section 7.4.1). **No edge reconnection graph surgery is needed** — fork parents / skeleton chain /
     completion set are naturally generated by existing assembly rules based on the post-pruning sequence.
   - Register / Event accounting **zero changes**: pairing consumption is already completed during emission; pruning is only a graph structure
     operation; post-loop instruction wait matching uses the unchanged accounting results (deleted post's bits
     have already been consumed by W, with no residual waiters).
4. **Iteration and statistics**: Repeat steps 2–3 until no new matches (handles multi-level chains within the body, such as
   prefetch→wait→mid-segment→wait→end-segment: after deleting the first level, the second level becomes exposed as a new chain head);
   `CcuBuildStats` adds `pipeline_chains_broken`, summary line presents as `pb=N`.

Fixed sample task graph after resolution (compare with the beginning of this section; deleted nodes are record(0:1)/wait(0:1), the original
"wait(0:1) → ub_write" gate is now handled by AA, timing-equivalent):

```
AA -> wait(0:ff)                                        ← loop-end barrier (skeleton)
AA -> memcpy(mem->ms)                                   ← prefetch unit (record deleted)
AA -> Multiple { ub_write_or_read -> record(0:xx) }      ← 7-way write, concurrent with prefetch
AA -> memcpy(ms->mem) -> record(0x80)                   ← copy-out unit, concurrent with prefetch
Multiple { record(0:xx) / record(0x80) -> wait(0:ff) }  ← loop-end waits for all data to complete
```

#### Implementation Details (Code Locations)

- `simulator/task_model.py`: `Task` adds `loop_tag` slot (default empty string),
  `to_trace_args` outputs `loop_tag` (empty string → `N/A`, same caliber as `loop_cnt`);
- `ccu_graph_builder.py`:
  - `_MissionContext.__init__`: `_loop_tag = ""`, `_lg_seq = 0`, `wait_posts = {}`;
  - `_exec_loop_group`: on entry, records `lg_tag = f"loopgroup_{self._lg_seq}_loop_{
    cur_loop_idx}"` (passed through to `_emit_merged_loop`), after LoopGroup processing completes,
    `self._lg_seq += 1`;
  - `_emit_merged_loop`: within the window, sets `self._loop_tag = lg_tag` (`finally` resets,
    same pattern as `_loop_cnt`);
  - `_gen_waits`: after successful pairing, `self.wait_posts[wait_task.task_id] =
    [p.task_id for p in posts]`;
  - `expand_ccu_graph` assembly loop start calls `_break_pipeline_chains(ctx, dep)`
    (returns deleted chain count, accumulated into `stats.pipeline_chains_broken`);
- `graph_dumper.py` / `simulator/report.py`: dump `nodes[]` and trace args
  add `loop_tag`; DOT/HTML legend synchronized (optional, presented alongside loop_cnt).

Trace presentation (`export_trace`, display layer only, does not affect simulation): tasks on virtual sub-streams carrying
`loop_tag` have their display line re-attached to the current stream's loop tag line (line name is the tag, such as
`loopgroup_0_loop_0`), indicating they are converted from in-loop instructions; remaining tasks on the same sub-stream
maintain the original display line — cross-mission reused mixed sub-streams are therefore split into original line + tag line.
Display lines are partitioned by `(stream_id, loop_tag)`; tasks from different virtual streams are not merged into one line (preserving
inter-sub-stream task partitioning); main stream tasks (including loop-end waits) all maintain the original line; task
`args.stream_id` preserves the real stream number.


```
_break_pipeline_chains(ctx, dep):
    broken = 0; changed = True
    while changed:
        changed = False
        for W in [n for n in ctx.nodes if type == Notify_Wait and n.loop_tag]:
            posts = ctx.wait_posts.get(W.task_id);  T = W.loop_tag
            if not posts:                              continue   # unpaired/remote
            if any R ∈ posts: R.loop_tag != T:        continue   # cross-loop source
            if any R ∈ posts: R's preceding ctx.nodes node is not a same-tag data task: continue
                                                                # floating set/sync
            if no same-tag data task after W:           continue   # loop-end barrier
            for R in posts:                                        # --- matched ---
                if R ∈ other wait's wait_posts values: continue   # has preserved consumers
                dep deletes R; ctx.nodes removes R                # unit tail now handled by M
            dep.remove_match(W); dep.extra_edges pops W
            dep deletes W; ctx.nodes removes W
            broken += 1; changed = True; break                    # rescan
    return broken
```

Acyclic guarantee: pruning only deletes nodes and edges, does not add edges, and deleted W/R are all nodes within the emission window;
fork/join edges are recalculated by the assembly phase based on the post-pruning sequence (Section 7.4.1 acyclic argument continues to hold);
`_drop_cyclic_matches` (Section 5.4) and simulator runtime deadlock detection remain as fallback.

#### Approach Evaluation (Pros and Cons) and Alternatives

**Pros**:

1. Does not alter Section 6.5 merge semantics or checker alignment logic — post-hoc resolution, risk isolated;
2. Prunes ctx.nodes before assembly (rather than performing graph surgery on the graph result): fork parents / skeleton chain /
   completion set are naturally generated by existing assembly rules, avoiding the "edge reconnection after node deletion" graph surgery correctness burden
   (post-assembly surgery would require manual reconnection of fork edges, skeleton chain, and completion set, which is error-prone);
3. Records wait→posts pairing during emission; identification does not depend on the graph result's dependency shape — insensitive to Section 7.4.1
   assembly rule evolution (fork parent selection, unit splitting adjustments);
4. Loop tag limits resolution scope (same loop, same mission), preventing cross-loop / cross-mission
   mis-deletion; the tag also feeds back into trace/dump tracing;
5. Deletes nodes rather than just deleting pairing edges: avoids preserving a semantically dangling unpaired local wait (although its
   measured `local_wait_delay` = 30ns has no timing difference, the trace statistics caliber is cleaner).

**Cons / Risks**:

1. Feature chain is a structural heuristic; identification conditions couple with emission / assembly logic and must be maintained as both evolve
   (coupling surface compressed to the ctx.nodes sequence layer through "emission-time recording + pre-assembly pruning");
2. `Task` model field expansion; trace/dump/JSON presentation must be synchronized (Chapter 11 caliber);
3. "Full concurrency + bandwidth manager arbitration" is a pipeline **steady-state** approximation: ignores pipeline fill/drain
   latency (single iteration 4KB scale, sub-microsecond, negligible) and per-iteration handshake **rate** overhead
   (if Event engine throughput is the bottleneck, a specialized rate model is needed; currently both folding / deletion count as single occurrences,
   same caliber as before the fix, logged in Section 10.3 #21);
4. The full expansion fallback path (`mf > 0`) does not go through this resolution — its per-iteration wait/record is real 4KB
   granularity handshaking, but inter-iteration overlap is also serialized by the queue chain; this path has a large node count, is not triggered by current samples,
   and remains as a known residual (Section 10.3 #21).

**Alternative Approach Comparison**:

| Alternative | Idea | Pros | Cons | Conclusion |
|-------------|------|------|------|------------|
| B. Emission-time direct judgment (no tag / no identification) | `_emit_merged_loop` directly identifies mid-body waits by slot structure (data slots remain after it) and skips emitting their wait/pairing | Simplest implementation, no Task field extension; judgment information (slot order) is at hand, no heuristics | Loses handshake node and loop tracing in trace; skipping emission changes register pairing accounting (loop-end wait absorbs prefetch set, timing harmless but widens behavioral difference with checker); floating set / sync scenarios need additional exclusion conditions | Feasible simplified variant; selectable if trace presentation is not needed |
| C. Segmented pipeline modeling | Folds data tasks into K segments (8~16), preserves wait/record between segments, explicitly models fill/drain | Includes pipeline fill effect, more precise | Node count × K, K value has no physical basis; under steady state differs from full concurrency by only one head/tail segment | Over-design, rejected |
| D. Simulator soft dependency edge | New edge type "pipeline edge" (child starts after parent completes X%) | Graph structure unchanged | Simulator semantics complicated, X has no physical basis; and serial chain structure (unit fork from wait) still requires graph-layer modification | Rejected |

Conclusion: Adopt the original approach (tag + feature chain + deletion), with the implementation locations and safety
conditions strengthened as described in steps 2/3 above (emission-time recording + pre-assembly pruning + mid-body determination + consumer conservative branch).

**Implementation status (implemented)**: `Task.loop_tag` field (task_model / trace args /
dump `nodes[]` / input_builder round-trip) + `DependencyGraph.remove_task()`
(including cleanup of extra edges and notify pairing) + `_break_pipeline_chains()` (called at assembly loop
start, counts `pipeline_chains_broken`, summary presents `pb=N`) + trace
loop sub-stream rename presentation (`export_trace`, see above). Verification:
sample ALLGATHER db cross-engine scenario makespan 2657.521 → 2561.541 us (56 → 54
tasks, pb=1, all 9 units fork and reattach to predecessor skeleton node rwt22, consistent item-by-item with manual prototype);
archive library regression — same-engine phase scenario (mv→mv all HBM) makespan unchanged (zero-error scenario,
theoretical expectation), SQE-only library pb=0 unaffected. `loop_tag` is preserved as a persistent task attribute
for subsequent performance simulation consumption (such as per-loop bandwidth/rate modeling, Section 10.3 #21).

### 6.6 Graph Construction Algorithm (Pseudocode)

```
expand_ccu(device_id, die_id, sqe_params, instr_space, dep_graph, reg_model):
    SQEs grouped into dispatch contexts by contiguous segments (Section 6.1: non-contiguous = new dispatch, independent window)
    Same-queue contexts serialized by dispatch order; each queue takes first context into active set
    while active set is not empty:
        progress = False
        for q in active set's each queue current context:        # cross-queue polling
            while ptr[q] has not reached window end:
                instr = decode(instr_space[ptr[q]])
                if instr is WAIT and reg_model.cke bits insufficient:
                    break                              # this queue pauses,轮到other queues
                nodes = transform(instr, reg_model)   # Section 6.3 mapping; LOAD only changes registers
                for node in nodes: attach to queue q chain tail
                if instr is SET_CKE: reg_model.seen_post register
                if instr is WAIT:     pair with seen_post → aggregated wait node (Section 6.4)
                if instr is LOOP:     merge path (Section 6.5 verification) or interpret loop body repeatedly by iteration count
                ptr[q] += 1; progress = True
            if ptr[q] reaches window end: current context complete, same-queue next context enters active set
        if not progress:
            # stall = exists wait depending on remote CKE (cross-rank, invisible in single-rank graph)
            for stalled queue: generate remote Notify_Wait node (notify_id+mask, fixed delay),
                then replay the instruction's remaining parts (local set / data transfer), continue advancing
    SQE stream attachment (Section 7.4.1): placeholder removed from SQE stream chain (task graph introduces no placeholder nodes);
        entry edges (SQE predecessor → queue skeleton head) / exit edges (mission completion set → SQE successor)
        expressed through extra edges
    Assembly splitting (Section 7.4.1): each mission queue sequence → skeleton (stays on queue chain) + async units
        (attached to #v virtual stream, stream_id changed to point to virtual stream), fork (predecessor skeleton→unit head) /
        completion set join (skeleton tail+all unit tails → SQE successor / same-queue next mission entry)
        extra edges connect
```

- **Cross-rank post-wait edges not built, but synchronization task nodes preserved** (single-rank graph construction, primary difference from the checker): the checker's `GenerateRawCcuGraphsV3` (`ccu_task_transform_v3.cc:1275-1406`)
  allows multi-rank CCU graphs to advance alternately, with cross-rank posts mutually unlocking; this plugin has only single-rank instruction data:
  - Instructions that post to remote Event / remote register notify → generate remote `Notify_Record` node
    (src=local, dst=INVALID → fixed `remote_record_delay`, not registered in local Event accounting,
    does not participate in pairing):
    - A5: SyncCKE's `rmtCKEId/locCKEMask`, SyncGSA/SyncXn's
      `setRmtCKEId/setRmtCKEMask`, TransLocMSToRmtMS's `setRmtCKE` (consistent with checker
      `AddPost` trigger conditions);
    - A6: TransMem `dmaOpCode=5` write-then-notify and `0x1A` remote atomic count,
      SyncXnWtx's `notifyValid=1` notify, SyncAtx's atomic sync — remote target type
      (Variable/Event) is resolved by peer address mapping (checker `GetXnAndTypeIdByAddr`), unobtainable in single rank,
      all approximated as remote record (mask takes the notification value's lower 16 bits for trace display only).
    `SyncGSA/SyncXn` (A5) and `SyncXnWtx` main write (A6) value synchronization is only a remote register write
    (checker builds nodes/writes state on the peer graph); the local side does not separately generate a task.
   - When bits required by wait cannot be satisfied by local rank posts (stall detection finds this) → the wait depends on remote → generate
     remote `Notify_Wait` node (fixed `remote_wait_delay`, not paired), and replay the instruction's remaining
     parts; unsatisfied waits within a loop body similarly generate remote wait nodes and continue. When partially satisfied: satisfied
     bits pair normally with local posts (aggregated into 1 local wait node, Section 6.4), unsatisfied bits generate one
      remote wait node. A6 independent Wait instruction (Variable value comparison) when unsatisfied uses the same mechanism: top-level stall →
     stall resolution generates remote `Notify_Wait` (task id tagged `xwt`) then passes through.
    - The task graph does not produce any nodes from nothing: all nodes come from original db data or microcode instruction conversion
      (attachment dependencies expressed through extra edges, see Section 7.4.1).
- Post-wait within the local rank pairs normally: Event across multiple queues of the same device, across multiple dies
  are all in the local graph, with edges built per Section 6.4 rules.

### 6.7 Simplified Comparison with Checker CCU Graph Construction

**CCU task rank/eid/direction semantics**:

- **Channel-type network tasks** (write/read): local rank takes opDetails.rankId; peer rank is looked up through
  channel peer eid (reid) via the eid→rank index (Section 4.8); if not found, fills INVALID +
  `rank_unresolved` count. Equivalent to checker `GetRemoteRankByChannel`
  (`g_allRankChannelInfo`: channelId → peer), but uses RankEidMap's communicator rank
  instead of the checker's physical dstDeviceId.
- **Reduction tasks** (REDUCE microcode): CcuBuffer is a rank-local buffer (checker treats all msIds as always
  `GenSliceFromMs` + `AddLocalBatchReduce`); reduction is always a local HBM operation, src/dst both filled with
  local rank; cross-rank data has already been charged in the preceding Rmt* role transfer, reduction does not repeat network charging.
- **Local tasks** (local transfer Memcpy, local reduction Reduce_Inline): src/dst both filled with local rank.
- **is_write**: UB tasks produced by CCU all explicitly carry (write=True / read=False),
  not depending on simulator rank heuristics.
- **Local rank self-verification**: if channel leid ↔ index-resolved rank does not match opDetails.rankId,
  opDetails takes precedence with a warning (eid numbering system offset protection, such as when deviceId and rankId have an offset).

| checker Step | light_performance | Rationale |
|--------------|-------------------|-----------|
| Two-phase (raw graph → queue parallelization → reconnection) | Single-phase interpretive execution + assembly-phase queue parallelization splitting (Section 7.4.1: `ApplyCcuQueueParallelizationV3` simplified port, skeleton/unit splitting + fork/join and attach extra edges + `#v` virtual streams) | No verification needed, no need to preserve a modifiable raw intermediate form; parallelization completed in one pass during assembly (reuse granularity and other differences described in Section 7.4.1) |
| Loop merge (BatchTransMem/BatchReduce, resource conflict check, producedMasks balancing) | **Equivalently implemented** (Section 6.5: precondition verification + folded emission + fallback to full expansion; memory slice merge not needed — simulation only cares about data volume and timing) | Aligns with checker semantics to eliminate node bloat |
| SQE grouping into TaskCcuGraph (`IsContiguousCcuSqe` contiguous-then-merge + `CcuMissionKey` latest-node-append) | **Equivalently implemented** (Section 6.1: contiguous segment splitting into dispatch contexts, each context independent window replay + sequential args consumption; registers/CKE shared across contexts, same-queue contexts serialized by dispatch order) | Aligns with checker multi-dispatch (window wrap-around re-delivery) semantics; previously, merging into a single context lost the second dispatch's data transfers (Section 10.3 #20 fixed) |
| GetRemoteRankByChannel (channelId → dstDeviceId/dstDieId, mapped to rank through Communicator table) | Peer eid (reid) → RankEidMap/ranktable index → communicator rank (Sections 4.8/6.7) | In estimate mode, the Communicator table has only one row for the local rank; ccuChannels.rankId field is endpoint deviceId and unusable |
| AllRankParamRecorder complete address space mapping (XN/CKE/PFE/CHANNEL/JETTY/MISSION/MS ...) + HBM data content + remote address→register reverse lookup (`GetXnAndTypeIdByAddr`) | Preserves only xn/cke/gsa/channel four categories; A6 Load/Store HBM data plane and remote target reverse lookup not modeled (Section 6.4, `_MissionContextA6` deviation list) | Simulation only needs length and peer; data content/remote register number does not affect timing |
| A6 checker error-abort scenarios (unsupported dmaOpCode/dstType, clearType!=1, clearMask!=0, LoopGroup config illegal, LoadX/StoreX indirect number out of range, and so on) | Downgraded to warning + skip data plane (synchronization semantics still execute) or skip instruction | Performance tool tolerates incomplete/abnormal data, maintains queue skeleton integrity |
| CcuTraceInfo tracing, ccu_convert_config.json generation | Removed | Verification/rewriting tool specific |
| CheckAllPostMatch error report | Downgraded to warning + diagnostic statistics | Adapts to single-rank data |
| Cross-rank post unlocking others' waits (multi-rank joint advancement) | **No cross-rank edges built**: cross-rank post/wait generates remote sync nodes (fixed delay), peer readiness approximated by fixed delay | Single-rank graph construction (core difference) |

### 6.8 A5 and A6 Instruction Set Difference Overview

| Dimension | A5 (V1/Ascend950) | A6 (V2/Ascend960) |
|-----------|-------------------|-------------------|
| LOAD type | 7 instructions: including Address series | 18 instructions: unified X register + `LoadX`/`StoreX`/`ClearX`/Nop + arithmetic-logic (Add/Sub/Mul/And/Or/Not/Xor/Shl/Shr/Popcnt) |
| CTRL type | 5 instructions: Loop/LoopGroup/`SetCKE`/`ClearCKE`/Jmp | 7 instructions: + `SetCKBit`/Wait/Fence |
| TRANS type | 12 instructions: Mem↔CcuBuffer all combinations + `SyncCKE/GSA/Xn` | 7 instructions: 4 local types + unified TransMem (distinguished by dmaOpCode) + `SyncWtX`/`SyncAtX` |
| REDUCE type | Add/Max/Min (msId list addressing, ≤8 CcuBuffer) | ReduceAdd/Max/Min (same structure, no embedded `waitCKE`) |
| Loop registers | v1.loop{xnId, startInstrId, endInstrId}, loopCnt in Variable | v2.loop{xpId(context), xnId, xmId(iteration count), mode}, mode==1 conditional expansion (Section 6.5) |
| LoopGroup parameters | Variable: LoopNum[61:55]/RepeatNum[54:48]/NoRepeatNum[47:41]; Xm: gsaOffset/MSOffset/ckeOffset | Variable: loopInsCnt:10/expandOffset:9/expandCnt:9; Xm: ckOffset:10/msOffset:11/gsaOffset:32; Xp: xnIdOffset:32 |
| Register number addressing | Payload field direct use | `GetXnId`: when bit15=1, offset by loop xnIdOffset (Section 6.4) |
| Embedded sync fields | TRANS/REDUCE generally carry `waitCKE` + `setCKE` + clearType | Local TRANS/REDUCE only `setCKE` (wait handled by independent `SetCKBit`/`ClearCKE`/Wait instructions) |
| Channel/length values | Channel is instruction immediate field | Channel/length are both Variable register values (`Xn[xcId]`/`Xn[xlId]`) |

Both instruction set generations are already implemented for graph construction (A6 interpreter `_MissionContextA6` reuses A5's task emission/
Event pairing/advancement/merge emission infrastructure, only overriding instruction decoding and loop organization); the A6 instruction →
task type summary table is in Section 11.3.3, and the established deviation list versus the checker is in the `_MissionContextA6`
class docstring and Section 6.7 comparison table.

---

## 7. Performance Simulation Module (Sub-Requirement 4, Consolidated with HCCLSIM Baseline)

> This chapter is the design description for the `simulator/` subpackage (ported from `docs/light_performance/HCCLSIM/hccl_sim/`).
> Interfaces use HCCLSIM as the authoritative baseline. Differences from the old version and residual items are in Chapter 10.

### 7.1 Overview

```python
sim = HcclSimulator(config, dep_graph, topo_resolver)
result = sim.run(trace_filepath=None, max_trace_events=0)   # → dict
```

The simulator receives three parameters:

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
    "sim_time_s": float,      # Software runtime (seconds)
    "tasks": list[Task],      # All tasks (including start/end_time)
    "tasks_by_stream": dict,  # int stream id → [Task] (chain order)
    "ports": dict,            # tuple(ports) → NetworkPort
}
```

The simulator does not accept task sequence lists and does not infer dependencies — all are obtained from `dep_graph`.

### 7.2 Parameter One: `config: SimConfig`

**Unit convention**: internal latency=seconds, bandwidth=B/s; **JSON configuration units latency=us, bandwidth=GBps**
(`from_dict`/`to_dict` auto-converts). Port identifiers are `"die_id/port_number"` strings
(such as `"1/0"`), consistent with ranktable.json.

#### Field Definitions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `local_record_delay` | float | `1e-6` | Local record fixed duration; JSON value unit us |
| `remote_record_delay` | float | `1e-6` | Remote record fixed duration; JSON value unit us |
| `remote_wait_delay` | float | `1e-6` | Remote wait / unmatched local wait fixed duration; JSON value unit us |
| `port_bandwidth` | dict | `{}` | `"die/port"` → bandwidth (B/s). One eid can correspond to multiple ports, configured per port individually, total bandwidth summed |
| `default_port_bandwidth` | float | `53.125e9` | Default bandwidth for unconfigured ports (425 Gbps = 53.125 GBps); JSON value unit GBps |
| `port_type` | str | `"x4"` | Port type (`x2`/`x4`), authoritative source is db `Server.port_type` (server/pod topology configuration `port_type` field stored through servers_info.json). Configured bandwidth is x4 baseline; for `x2`, non-d2h port bandwidth is halved (see "Port Type Scaling" below) |
| `hbm_bandwidth` | float | `1.4e12` | HBM bandwidth, used for Reduce_Inline/Memcpy; JSON value unit GBps |
| `bw_profile_set` | BandwidthProfileSet? | `None` | Bandwidth utilization configuration set; when None, utilization is 100% (including multi-jetty conversion step `jetty_utilization_step`, see below) |

#### Port Bandwidth Configuration Format

```json
{"port_bandwidth": {"1/0": 53.125, "0/7": 53.125}}
```

#### Port Type Scaling (`port_type`, x2/x4)

Port bandwidth in `sim_config.json` (`<topo_type>_port_bandwidth` and
`default_port_bandwidth`) is all **x4 baseline**. The server/pod topology configuration file's
`port_type` field (`x2`/`x4`, default `x4`, illegal values cause error) is written by
`generate_cluster_topo.sh` into the cluster-level `servers_info.json` at each **leaf node** entry
(in server form, one entry per server; in POD form, one entry per board, with explicit
`super_pod_id`/`server_id`/`board_id` triple id, for the parser to match by id rather than by position order;
this file is tool-specific and does not enter topo.json/rootinfo/ranktable and other hccl workload native format files),
then stored by the C++ cluster parser (`ParseServersInfoJson`) into the `Server.port_type` column
(`sim_models.h`/`db_sim_runner_ops.h`, RunnerDB v5; from v5, the `Server` table adds
`board_id` column, with one `Server` row per board in POD form, and `board_id` always 0 in server form).

During simulation, `pipeline`/`__main__`/`dump_graph` entry points call
`db_reader.load_port_type(db_path, device_id)` to read the port type from the `Server`
table in `hccl_sim.db` (locates the target device's server through `Device.id → server_id → Server.id`;
falls back to the sole value in the full table when not located; defaults to `x4` when the old database has no `port_type` column), then applies
`SimConfig.with_port_type()` to the configuration. `get_port_bandwidth()` multiplies **non-d2h**
ports by `port_type_scale` (`x4=1.0`, `x2=0.5`), that is, x2 port bandwidth is halved;
`d2h_bandwidth` and HBM bandwidth are unaffected. The command line `--port-type` can override the db value.

#### Bandwidth Utilization Configuration (`bw_profile_set`)

```
BandwidthProfileSet
├── profile_low:   BandwidthProfile  ← used when shared port concurrency < 7
├── profile_high:  BandwidthProfile  ← used when shared port concurrency ≥ 7
└── THRESHOLD: 7
```

Each `BandwidthProfile` is a data volume (32B~128MB) → utilization (0~1) mapping table. Queries round down to the
largest tier not exceeding the data volume (floor matching).

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

#### Multi-Jetty Conversion (`jetty_utilization_step`)

`multi_port_utilization_low` is calibrated for the standard configuration of **one channel, one jetty**.
When `HCCL_UB_MULTI_CHANNEL_NUM=n`, one channel has n jettys, and concurrent transfer capacity approximately increases linearly;
lightweight simulation does not build a jetty-level model (jetty is used only for eid/protocol resolution, see Section 5.5), instead converting
the jetty count into an overall uplift of the `profile_low` table:

```
util'(size) = min(1.0, util(size) + (n - 1) × jetty_utilization_step)
```

- `n` is taken from `opDetails.ubMultiChannelNum` (proxy level0 stub reads environment variable and stores, Section 4.2),
  applied per operator; when `n ≤ 1` (not configured/illegal, falls back to 1), the original table is used directly, no uplift applied;
- `jetty_utilization_step` value range `[0.0109, 0.044]`, defaults to mid-value `0.02745`,
  configured only through `bw_profile.json` (out-of-range produces WARNING then clamped to range boundaries);
- Only applies to `profile_low`: `profile_high` is the high-concurrency tier (ports already saturated by multi-path transfers, multi-jetty
  has no additional gain); single-port/`d2h` uses constant utilization (`single_port_utilization` /
  `d2h_utilization`), none are converted;
- Entry point: `SimConfig.for_jetty_num(n)` → `BandwidthProfileSet.for_jetty_num(n)`
  (`bandwidth_profile.py`); pipeline generates a converted config copy per operator;
  debugging can use `--jetty-num` to override the db value (`run_perf_sim.py` / `python -m light_performance`).

### 7.3 Parameter Two: `dep_graph: DependencyGraph`

#### Four Layers of Content

```
DependencyGraph
├── tasks: dict[str, Task]                       ← Task nodes (with all parameters)
├── stream_chains: dict[int, list[str]]          ← Serial dependencies (int stream id)
├── notify_matches: dict[str, str]               ← Notify dependencies (wait_task_id → record_task_id)
└── extra_edges: dict[str, list[str]]            ← Additional dependency edges (child_task_id → [parent_task_id, ...])
                                                  ← Source: CCU queue parallelization fork/join edges and SQE attachment edges (Section 7.4.1)
```

#### First Layer: Task Nodes (`Task`)

Each Task carries all parameters needed for execution (invalid values uniformly use `INVALID = 0xFFFFFFFF`,
eid is empty string, string attributes are `""`):

| Field | Type | Applicable Task Types | Simulator Purpose |
|-------|------|----------------------|-------------------|
| `task_id` | str | All | Dependency graph index |
| `task_type` | TaskType | All | Determines latency model (internal classification, Section 7.3) |
| `task_name` | str | For display | trace/dump display name: profInfo.profTaskType op name (uniformly filled by `_add_task`, Section 11.4.1); when empty, `display_name` falls back to `task_type.value` (CCU sub-tasks/CLI input) |
| `stream_id` | **int** | All | Serial chain attribution |
| `notify_id` | int? | Notify_Record/Wait/Write_With_Notify | Dependency pairing identifier (CCU path = cke id); Write_With_Notify is display only (its notify targets the peer, does not participate in pairing) |
| `mask` | int | Notify_Record/Wait | Notify bitmask (CCU path): notify_id + mask uniquely identifies one notify action; wait node carries aggregated mask (OR of all contributing post consumed bits for one wait instruction, Section 6.4); SQE path with 0=unfilled |
| `src_rank` / `dst_rank` | int? | notify/transfer | Local determination (src==dst); read direction local determination |
| `size` | int | Memcpy/Reduce_Inline/Ub_*/Write_With_Notify | Calculates delay + looks up utilization |
| `port` | str/list? | (reserved field) | Port resolved from eid in real time, not filled by db path |
| `src_eid` / `dst_eid` | str | Ub_*/Write_With_Notify | eid → ranktable → physical port |
| `transport_type` / `link_type` | str | For display | msprof reconciliation dimensions (profInfo transportType/linkType name, Sections 11.4.2/11.4.3, uniformly filled by `_add_task`; CCU sub-tasks not filled) |
| `data_type` | str | For display | trace/report labeling |
| `protocol` | str | For display | Channel CommProtocol name (Section 11.4.2 note; empty when no channel info) |
| `jetty_id` | int? | Ub_*/Write_With_Notify | trace labeling |
| `is_write` | bool? | Network tasks | Charging direction and port query eid selection (Section 10.3 #1) |
| `start_time` / `end_time` | float | All | Filled by simulator (-1=not executed) |

7 TaskTypes:

| TaskType | Latency Model | Description |
|----------|--------------|-------------|
| `NOTIFY_RECORD` | Fixed delay (local additionally triggers Event) | Local = `is_local_notify`(src==dst); unpaired local record uses fixed delay only |
| `NOTIFY_WAIT` | Waits for Event / fixed delay | matched→waits for record's Event; unmatched/remote→fixed `remote_wait_delay` |
| `REDUCE_INLINE` | HBM dynamic sharing | `size / hbm_bandwidth`, shares bandwidth with other HBM tasks |
| `MEMCPY` | HBM dynamic sharing | Same as above (HBM copy, not network transfer) |
| `UB_INLINE_WRITE` | Fixed delay (same as Notify_Record: local/remote record_delay) | Notify inline write; not produced by SQE graph construction path, used only by old input path (task.txt) |
| `UB_WRITE_OR_READ` | Port bandwidth sharing (write/read two directions charged independently) | Network data send/receive; `is_write` explicitly carried by graph construction (SQE network tasks in both directions uniformly mapped to this type) |
| `WRITE_WITH_NOTIFY` | Port write direction sharing + post-write notify fixed delay | Network write + attached notify single task, one-to-one correspondence with db `WRITE_WITH_NOTIFY` (7); notify uses fixed remote record delay (peer wait not in this db) |

#### Second Layer: Serial Dependencies (`stream_chains`)

```
int stream id → [task_id_0, task_id_1, ..., task_id_n]
```

- List order = execution order: task[i] must complete before task[i+1] can execute
- Each stream_id starts one simpy process, executing serially in chain order with `yield`
- Different streams are naturally parallel
- db path stream id addressing: SQE streams use original streamId directly, **CCU queue skeleton chains belong to their SQE original stream**
  (Section 6.5, whichever stream a CCU-type SQE is on, the expanded tasks are on that stream; only when the original stream is occupied by another die's
  queue is a synthetic stream allocated); virtual streams allocated from base `1<<20`
  (Section 6.5, virtual stream logical name carries `#v{n}` suffix), logical names preserved through `stream_labels`;
  `GraphBuilder.stream_meta` provides real/ccu/virtual three-category classification and virtual stream
  parent_sid on top of this (Section 7.4.1, pointing to its queue's stream id, normally the SQE original stream), for report/trace/dump labeling

#### Third Layer: Notify Dependencies (`notify_matches`)

```
wait_task_id → record_task_id
```

Explicit 1:1 pairing: which wait waits for which record. When multiple records exist for the same notify_id, only one
pairs with the wait; unpaired records use fixed delay.

#### Fourth Layer: Additional Dependency Edges (`extra_edges`)

```
child_task_id → [parent_task_id, ...]
```

- Semantics: child must wait for **all** parents to complete before starting execution (pure timing constraint, does not carry
  record/wait event semantics — event pairing is still expressed by `notify_matches`).
- Current sources: CCU queue parallelization fork edges (predecessor skeleton → unit head), mission completion set
  join edges (skeleton tail + unit tail → SQE successor / same-queue next mission entry), CCU subgraph
  attachment edges on the SQE stream (entry/exit, see Section 7.4.1), and aggregated wait node multi-parent edges
  (wait → all contributing posts, Section 6.4 — first contributing post uses `notify_matches`, remaining use this layer).
- Writer side `add_edge` auto-deduplicates (same child-parent pair recorded only once); simulator queries through
  `get_extra_parents` / `extra_parent_ids`.
- Acyclicity: the above edges always connect "forward" along the original instruction order (skeleton before unit, unit before join point,
  attachment at placeholder position before and after, post registered before the wait instruction that consumes it), graph construction direction guarantees acyclicity,
  and included in `_drop_cyclic_matches` cycle detection fallback (Section 5.4).

### 7.4 Simulator Consumption Flow

```python
def run(self, trace_filepath=None, max_trace_events=0):
    # 1. Retrieve all task nodes
    tasks = list(dep_graph.tasks.values())

    # 2. Create synchronization primitives: only matched waits create Events
    for wait_task_id in dep_graph.matched_wait_ids:
        notify_events[wait_task_id] = env.event()
    for parent_id in dep_graph.extra_parent_ids:  # all parent tasks from extra edges
        task_done_events[parent_id] = env.event() # must be pre-built before stream processes start:
                                                  # if parent completes first and child builds Event later, it waits forever (Section 7.4.1)

    # 3. HBM bandwidth manager (Reduce_Inline + Memcpy shared, utilization always 1.0)
    hbm_mgr = _BandwidthManager(env, config.hbm_bandwidth, empty utilization table)

    # 4. Start one simpy process per stream, execute serially in chain order
    for task in chain:
        for pid in dep_graph.get_extra_parents(task.task_id):
            yield task_done_events[pid]           # extra edges (Section 7.4.1), start_time does not include waiting
        if NOTIFY_RECORD:
            yield timeout(calc_notify_record_delay(task, config))
            if local and matched: trigger the matched wait's Event
        elif NOTIFY_WAIT:
            if local and matched: yield notify_events[task_id]   # wait for Event
            else:              yield timeout(remote_wait_delay)
        elif REDUCE_INLINE / MEMCPY:
            yield hbm_mgr.send(task_id, size)                 # HBM dynamic sharing
        elif UB_INLINE_WRITE / UB_WRITE_OR_READ / WRITE_WITH_NOTIFY:
            is_write = task.is_write  # explicit direction; None for old input path falls back to heuristic
            eid = src_eid if is_write else dst_eid
            ports = topo_resolver.get_ports_by_eid(eid)       # throws error if missing
            port = look up/create NetworkPort(ports, total bandwidth)
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

Latency formulas are unified and maintained at a single point in `task_timing.py`.

#### 7.4.1 CCU Queue Parallelization and Virtual Streams (Extra Edges)

**Problem** — The CCU queue chains produced by Sections 6.5/6.6 are per-task serial streams: all
data transfers, waits, and sets within the same queue execute in order. On real hardware, a queue's asynchronous transfer units execute concurrently.
Serialization causes three distortions:

| Distortion | Mechanism |
|------------|-----------|
| Makespan overestimation | Asynchronous transfers are separated one-by-one by skeleton (wait/set), unable to overlap |
| Utilization tier distortion | Concurrent transfers within the same queue degrade to sequential execution; `_BandwidthManager` observes port concurrency always as 1, always queries profile_low tier (Section 7.5) |
| Port peak concurrency statistics distortion | `peak_concurrency` always 1, unable to reflect concurrent bandwidth sharing behavior (Section 8.2 port-level metrics) |

**Checker reference** — `ApplyCcuQueueParallelizationV3` (`ccu_task_transform_v3.cc:1052`):
second-pass transformation after raw graph construction, `CollectAsyncBoundRecordNodesV3` collects asynchronous transfer nodes,
`QueueIdReuseAllocatorV3` reuses virtual queue ids by `[startNodeId, afterNode]` interval,
fork/join edges reconnect to skeleton.

**Simplified port** (`ccu_graph_builder.py` assembly loop, completed in one pass during the assembly phase, no raw
intermediate form) — splits each mission's queue node sequence:

1. **Unit identification**: `_QUEUE_DATA_TYPES` (`Memcpy`/`Ub_Inline_Write`/
   `Ub_Write_Or_Read`/`Reduce_Inline`) data tasks + immediately following consecutive
   `NOTIFY_RECORD` chain = one **asynchronous unit**; remaining nodes (wait / floating set /
   remote sync nodes) are **skeleton**, staying on the queue serial chain — the queue chain is the skeleton itself,
   no additional head/tail placeholder nodes are introduced.
2. **Fork edge**: `add_edge(unit head, predecessor skeleton node)` — unit must wait for queue to advance
   to its predecessor skeleton before starting (adjacent units share the same fork parent node); units appearing before the skeleton
   (mission first node is a data task) have no skeleton predecessor, and their entry dependencies are instead expressed by attachment edges.
3. **Completion set join**: mission completion set = skeleton tail + all unit tails, joined through extra edges to
   — same-queue next mission entry (skeleton head + pre-skeleton unit head, queue serial semantics;
   skeleton tail ordering already covered by queue chain serialization) and SQE stream successor tasks (exit edges,
   added by graph_builder during the attachment phase).
4. **Virtual stream reattachment**: each queue has one set of counters/reuse pools (`vstream_seq`/
   `vstream_free`/`vstream_busy`), unit tasks' `stream_id` changed to point to
   `ccu_d{die}_q{queue}#v{n}` (same `1<<20` base address space), unit chains registered into
   `result.virtual_chains` and `add_stream_chain` — serial chains constrain only skeleton and
   intra-unit task order.
5. **SQE stream attachment** (`graph_builder._expand_ccu` consumes `result.attach`):
   CCU placeholder removed from SQE stream chain (task graph produces no placeholder/from-nothing nodes),
   entry edges = SQE predecessor tasks (including immediately preceding CCU mission completion set) → skeleton head and
   pre-skeleton unit head; exit edges = mission completion set → SQE successor tasks. All expressed through extra edges,
   with timing semantics: SQE predecessor (and preceding mission completion set) must complete before subgraph can start,
   and subgraph mission completion set must all finish before SQE successor tasks can start.
   Empty chains are removed from the SQE stream for all-placeholder tasks.

**CKE timing preservation**: splitting does not alter `add_match` pairing — whichever unit a record falls into, or whether a wait
stays on the skeleton, does not affect event pairing (waits wait across streams through notify_events). In the measured sample,
the trailing memcpy unit's fork parent wt28 happens to be on the preceding read unit's tail set accounting wait,
and cross-stream events + fork edges together guarantee the "read complete → trailing copy" original timing. **Exception**: merged loop
emission window mid-body wait/record pairings are folded pseudo-serial barriers (per-iteration pipeline handshakes
folded into full barriers), resolved by Section 6.5.1 feature chain pruning before assembly — this section's splitting is based on the post-pruning
sequence, and skeleton/unit/fork structure is unaffected.

**Reuse rule differences versus checker**:

| Dimension | checker (V3) | light_performance |
|-----------|--------------|-------------------|
| Reuse granularity | afterNode interval (join successor node) level | Mission level (all predecessors released when entering a new mission) |
| Intra-mission reuse | Allowed (DAG explicit edges, no pseudo-dependencies) | Not reused (same-mission units are concurrent with each other; under chain model, reuse on the same stream would introduce pseudo-serialization) |
| Residual strictness | GRAPH_UNMATCHED error exit | Tolerated (performance tool, Section 6.4 downgraded to warning) |

Cross-mission reuse safety: units reusing the same virtual stream are serial within that stream chain (later mission's
unit chain follows earlier mission's unit chain), with no timeline overlap.

**Acyclic guarantee**: fork (skeleton → unit head) and completion set join edges (unit tail → SQE successor /
next mission entry) always connect "forward" along the original instruction order; splitting does not change the original order;
`_drop_cyclic_matches` cycle detection also accounts for extra edges (Section 5.4) as fallback; the simulator
additionally has runtime deadlock detection (reports error and exits when task `end_time<0`).

**Statistics and marking**: `CcuBuildStats` carries parallelization (`parallel_units`/`virtual_streams`/
`fork_edges`/`join_edges` — join_edges now counts completion set join and SQE attachment edges), remote sync
(`local_posts`/`remote_posts`/`wait_matched`/`remote_waits`) and other counts,
summary line presents as `pu/vs/lp/rp/wm/rw`;
`GraphBuilder.stream_meta` classifies by logical stream name — `s{n}` → real, contains `#` →
virtual (with parent_sid attached), others → ccu, consumed by three locations: report (`Stream count:
N (including virtual M)`, virtual stream lines annotated with `vir_stream {n} (label)`, n numbered from 0
in allocation order, detail folded when exceeding 64 streams), trace (virtual stream thread_name `vir_stream
{n} ({label})` + task args `virtual_stream: true`), dump (streams[] with
kind/virtual/parent_sid,
extra_edges list; DOT purple dashed cluster + blue dotted extra edges, HTML badge
+ dotted arcs). When stream_meta is absent (simulator standalone CLI old input), no annotation is made, backward compatible.

**Quantified effect** (sample db REDUCE_SCATTER iter0, 66 tasks): queue parallelization reduces makespan
from 20,404.20 → 5,149.36 us (≈3.96x); subsequent two corrections further reduce to **2,753.71 us**:
① Remote CcuBuffer reduction correction (Section 6.3.2, reduction always local HBM) — the original second phase 128MB reduction charged as
network read ~2526 us, corrected to HBM ~96 us; ② Cross-rank sync node completion (Section 6.6) —
14 waits depending on remote set were previously skipped (read units launched early at t≈0), corrected to simulate with fixed
remote delay. 10 asynchronous units reattached to 10 virtual streams (fork edges 10, attachment/join edges generated based on
SQE neighbors and queue serial conditions) — first phase 8 units concurrent (7 network read units each
~2526 us, belonging to 7 independent ports `1/0`..`1/6` + 1 local memcpy ~96 us), followed by
reduction ~96 us and trailing memcpy ~96 us in serial succession. Note that single-port read peak concurrency remains
1 — the benefit comes from **cross-port overlap**, not single-port concurrency improvement.

### 7.5 Bandwidth Sharing Mechanism (`_BandwidthManager`)

#### General Manager (Ports and HBM Reuse the Same Model)

- **Single manager process model**: when a new transfer joins, `interrupt` the manager process, which evenly divides bandwidth by current concurrency N:
  `bw_per = total bandwidth / N`.
- **Utilization query basis is `min_remaining`**: the same group of concurrent tasks queries the
  utilization table using "minimum remaining data volume" (concurrency <7 uses profile_low, ≥7 uses profile_high), same basis within group.
- **profile_low table already converted by jetty count**: before querying, the pipeline has already used
  `SimConfig.for_jetty_num(opDetails.ubMultiChannelNum)` to generate a config copy;
  when multi-jetty (`HCCL_UB_MULTI_CHANNEL_NUM>1`), table values are uplifted overall; the manager itself does not perceive
  jetty (Section 7.2 "Multi-jetty conversion").
- Effective bandwidth = `(total bandwidth / N) × utilization(min_remaining, N)`; calculates earliest completion time,
  timeout settles progress; recalculates when joins/completions occur.
- **HBM dynamic sharing**: `Reduce_Inline` and `Memcpy` share one HBM manager
  (utilization always 1.0) — when multiple HBM tasks run concurrently, they share HBM bandwidth evenly (different from the old version's per-task independent
  `size / hbm_bandwidth`, see Section 10.3).

#### Bidirectional Independence

Each `NetworkPort` contains write and read two independent managers, with independent non-interfering bandwidth:
when port bandwidth B is configured, the write direction can use B, and simultaneously the read direction can also use B.

#### Multi-Port Aggregation

When one eid corresponds to multiple physical ports, `port_key = tuple(port_ids)` serves as the logical port,
total bandwidth = sum of each port's bandwidth (`get_ports_total_bandwidth`).

### 7.6 Parameter Three: `topo_resolver: TopoResolver`

- Input `ranktable.json` (`TopoResolver(ranktable_path=...)`), providing
  **eid (16B hex string) → physical port list (`["die/port", ...]`)** resolution.
- Transfer task routing: write direction queries `task.src_eid`, read direction queries `task.dst_eid`.
- **Throws error directly when eid cannot find a port** (no jettyId fallback, no default port) — therefore the db path
  skips simulation entirely when "network tasks exist but no ranktable is available" (Section 3.3), and the db graph construction counts and warns about tasks with unresolvable
  eids (Section 5.5).
- Port bandwidth is looked up per resolved port individually in `SimConfig.port_bandwidth` and summed.

### 7.7 Build Methods and CLI

All input methods uniformly build `(config, dep_graph)`:

| Input Source | Build Function | Returns | Status |
|-------------|---------------|---------|--------|
| **hccl-vm db (newly added this iteration)** | `graph_builder.build(group)` | `(config, dep_graph)` | Chapters 5, 6, primary input |
| JSON task file | `load_dep_graph(filepath)` + `load_config(...)` | `(config, dep_graph)` | Preserved (hccl_sim compatible) |
| task.txt / algorithm-generated and other old input sources | — | — | **Removed with HCCLSIM version** (see Section 10.3) |

Build flow unified as:
1. Create `Task` nodes, call `dep_graph.add_task(task)`
2. Call `dep_graph.add_stream_chain(stream_id, task_ids)` in task order within each stream
3. Call `dep_graph.add_match(wait_task_id, record_task_id)` based on notify pairing results

CLI entry points:

```bash
# db input (light_performance main entry, for details refer to Section 3.3)
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

The trace file uses Chrome Trace Event format: `ts`/`dur` value units are **us** (format specification convention),
and the file top level carries `"displayTimeUnit": "ns"` to specify the viewer's (chrome://tracing and so on)
Start/Wall Duration **display** unit as ns (default ms).

`export_trace` display adaptation conventions:
- **tid sequential numbering**: thread lines mapped to tid `0,1,2,...` in ascending stream id order (when original
  streamId 2/6..13 is directly used as tid, string-sorting viewers would place 10..13 before 2);
  real stream id preserved in thread_name label and `args.stream_id`.
- **Endpoint rounding**: `ts`/`end` each independently rounded before subtraction to get `dur` — `ts` and `dur` each independently rounded
  would produce 0.001us spurious overlap, and same-tid overlap slices would be nested into sub-rows by the viewer (displayed on new lines).
- **Virtual stream annotation** (Section 7.4.1): db path passes stream metadata through `export_trace(..., stream_meta=...)`,
  virtual stream thread_name changed to `vir_stream {n} ({label})` (n numbered from 0 in ascending sid order, that is, allocation order,
 摆脱 large-value stream numbers starting from 2^20 base; real stream number see
  in-line task `args.stream_id`), and its task args appended with `virtual_stream: true`;
  real/ccu stream names unchanged (CCU queue chain logical names appear only in text report lines). When stream_meta
  is absent (simulator standalone CLI old input), no annotation is made, backward compatible.
- **Notify task args carry mask**: CCU path `Notify_Record`/`Notify_Wait`
  task args appended with `mask: "0x…"` (notify_id + mask uniquely identifies one notify action,
  Section 7.3); SQE path and mask=0 do not output this field.
- **Operator summary information first line (op_summary)**: pipeline calls per operator
  `export_trace(..., op_summary=...)`, and each operator (when multiple operators are merged for export, each
  pid process group) reserves the `op_summary` first line with tid 1 + `thread_sort_index` 0, and task line
  tid and sort_index follow in sequence (presentation layer only; tid avoids 0 — some viewers such as Perfetto treat
  tid 0 as idle thread, `thread_sort_index` is the Chrome Trace standard line order control),
  summary event placed at the front of the event array, triple guarantee that op_summary is always the first line. The first line places one
  `X` summary slice spanning the operator's start to end (start time and duration are the slice's own ts/dur, taking
  the earliest task start to latest end), line label   `"<op_name> | <algorithm_name>"`,
  args carrying: `algorithm_name` (algorithm selection result, opDetails.algorithmName log
  capture backfill (DFX primary path, Section 4.2 note)), `data_type`/`data_count` (element count, V2 layout operators have
  send/receive element counts and types on both sides, others have count and single-side type), `task_count` (total task count after graph construction,
  for CCU mode this is the count after microcode instruction translation to task graph), and op_type/
  comm_id/op_iter/rank/rank_size/input_bytes/output_bytes (opMemInfo's
  inputSize/outputSize, calculated from workload API parameters at collection time) context; summary events are not counted toward
  `max_events`.

### 7.8 Alignment Adaptation Completion Status

Differences between the old version (before modification in this repository) and the HCCLSIM baseline have been resolved in this alignment:

| # | Item | Handling |
|---|------|---------|
| 1 | Task model 6 types → new 6 types (local/remote record/wait merged into Notify_*, LOCAL_COPY split into Memcpy/Reduce_Inline, network transfer changed to Ub_*; later with db addition of WRITE_WITH_NOTIFY expanded to 7 types, Section 10.3 #19) | `task_model.py` replaced per HCCLSIM |
| 2 | stream_id str → **int**; notify_id str → **int** (INVALID=0xFFFFFFFF) | `dependency_graph.py`/graph construction side synchronized |
| 3 | Port specified by `port` int field → **eid real-time resolution** (write queries src_eid / read queries dst_eid) | `topo_resolver.py` + graph construction side fills eid |
| 4 | Unit system: JSON latency s → **us**, bandwidth B/s → **GBps**, port key int → **"die/port"** | `config.py` from_dict conversion |
| 5 | HBM delay `size/bw` independent → **dynamic sharing** (Reduce_Inline+Memcpy share manager) | `_BandwidthManager` reuse |
| 6 | `run()` returns makespan → **result dict**; simulator two params → **three params** (+topo_resolver) | `simulator.py` |
| 7 | Latency formulas scattered → `task_timing.py` single point | Consolidated |

Unmatched residual items (including `src_rank==0` hardcoding, eid without fallback, broadcast pairing triggering only the first wait, and so on)
are uniformly moved to Chapter 10 for tracking.

---

## 8. Performance Profiling Output (Sub-Requirement 5, Overview)

### 8.1 Output Location and Files

```
<installRoot>/plugin/light_performance/data/<timestamp>/
├── perf_profiling.json      # Main product: performance metrics + task graph summary
├── trace_<commId>_<opIter>.json   # (optional) Chrome tracing, visualizable with chrome://tracing
└── light_performance.log    # Runtime log
```

- Directory naming: the `hccl-vm plugin run @estimate` one-shot execution channel does not pass data_id (Section 3.2),
  using runtime timestamp as substitute; standalone debugging mode (run_perf_sim.py) specifies output path with `-o`.
- Output fixed in plugin data directory rather than `data/`: the host clears `data/` at each start (`src/main.cc:32`),
  products under the plugin directory are preserved across sessions; `archive.sh` archiving covers only logs/ and data/, plugin products
  do not participate in automatic archiving (Section 10.3 #15).

### 8.2 Metric Definitions (Draft)

| Level | Metric | Source |
|-------|--------|--------|
| Operator level | opType/opIter/rankSize/dataSize, makespan, software simulation time | opDetails + `sim.run()` result dict |
| Operator level | Operator summary first line (algorithm_name/data_type/data_count, start/end and duration carried by slice ts/dur, Section 7.5 op_summary) | opDetails.algorithmName (DFX/log dual-path capture) + simulation task timeline |
| Operator level | Total transfer volume, effective bandwidth utilization (total transfer / (makespan×port bandwidth)) | Summarize result["tasks"] |
| Stream level | Each stream start/end time, duration, task count + virtual stream annotation (Section 7.4.1) | result["tasks_by_stream"] + stream_meta |
| Graph level | Async unit count, virtual stream count, fork edge count, completion set/attachment edge count (Section 7.4.1); local/remote post count (lp/rp), local wait match count (wm), remote wait count (rw) | CcuBuildStats (parallel_units / virtual_streams / fork_edges / join_edges / local_posts / remote_posts / wait_matched / remote_waits, summary line `lp/rp/wm/rw`) |
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

> `cross_rank_notify` is the count of preserved cross-rank notify nodes (diagnostic information): these nodes are graph-constructed normally in stream order,
> use fixed remote delay, but do not participate in local pairing (no cross-rank record-wait edges); this count is used to troubleshoot
> data/classification anomalies (such as local notify being misidentified as cross-rank). `kind`/`virtual_stream_count`/
> `extra_edge_count` are fields added by CCU queue parallelization (Section 7.4.1): stream classification real/ccu/virtual,
> virtual stream count, and extra edge (fork/join/attachment) count.

---

## 9. Appendix

### 9.1 Checker Graph Construction Module → light_performance Graph Construction Module Overall Comparison

| Aspect | checker (C++) | light_performance (Python) |
|--------|---------------|---------------------------|
| Goal | Correctness verification DAG (mainStart reachable, supports various checks) | Simulation DependencyGraph (three layers: tasks / stream_chains / notify_matches) |
| Input | db (through StorageManager/sim::Query) | Same db (sqlite3 direct read + struct decoding) |
| Graph construction scope | **Multi-rank joint graph construction** (all rank tasks enter the same DAG) | **Single-rank graph construction** (cross-rank pairing edges not built, remote sync task nodes preserved) |
| SQE translation | TaskMetaTranslatorV3 (including MemSlice address resolution) | Section 5.2 mapping table (only takes len/jettyId/notify fields) |
| WRITE_WITH_NOTIFY consumption | Split at decoding boundary: `DecodeAndExpandTaskMeta` expands to [write, NOTIFY_RECORD] two sub-tasks | **Not split**: single task mapping `Write_With_Notify` (Section 5.2), one-to-one correspondence with db tasks, consistent trace/dump/statistics granularity |
| Dependency edges | Stream chain + local/cross-rank notify edges (queue simulation) | Stream chain + local-only notify first-come-first-served (two-pass scan) + extra edges (CCU fork/join/attachment, Section 7.4.1) |
| CCU | Two-phase expansion + queue parallelization + loop merge + Event accounting | Single-phase interpretive execution + assembly-phase queue parallelization (Section 7.4.1) + loop merge (Section 6.5) + Event accounting (Chapter 6) |
| Post-verification processing | Memory conflict/semantic/sync conflict/reachability/single-task check | None |
| Output | Verification conclusion + dump | (SimConfig, DependencyGraph) → makespan → profiling json |

### 9.2 Glossary

| Term | Description |
|------|-------------|
| SQE | Submit Queue Element, task unit submitted by AICPU/SDMA side, that is, one row in the opTask table |
| WQE | Work Queue Element (UB/DB mode), not involved in this iteration |
| notify | AICPU-side event synchronization primitive, record sets / wait waits, paired by (recordDev, waitDev, notifyId) |
| Event | CCU-side event synchronization register (bitmap), post sets / wait waits, paired by (waitDev, dieId, ckeId, mask) |
| jetty | Network card (communication engine) port number, determines memcpy bandwidth attribution |
| die | Chip die, CCU microcode space and Event both organized by (device, die) |
| mission | One CCU task delivery (instStartId/instCnt defines instruction window, args[13] passes parameters) |
| makespan | Task graph end-to-end duration (latest task end time) |
| Main stream/Sub stream | Operator main stream marked by opDetails.streamId; others are sub streams (first WAIT, last RECORD structure) |

---

## 10. HCCLSIM Alignment Residual Items and Pending Match List

> This chapter records the **field gaps, task type mismatches, and other conflict points** confirmed during the alignment of the "db graph construction path + design document"
> with the HCCLSIM baseline. Current implementations have all been landed with "default fill invalid values / approximate mapping",
> and are documented here for future joint resolution with the collection side and simulation module.

### 10.1 Task Field Availability Table (db Graph Construction Path)

New model Task field sources and default values in the db graph construction path (invalid value convention: int fields `INVALID=0xFFFFFFFF`,
eid empty string, other strings `""`):

| Task Field | db Source | Availability | Default/Approximate Value | Future Match Path |
|------------|----------|--------------|--------------------------|-------------------|
| `task_id` / `stream_id` | Global order / streamId+synthetic base | Available | — | — |
| `notify_id` | opTaskMeta `NotifyTask.notifyId` | Available | — | — |
| `size` | MEM_CPY `len` / REDUCE `dataCount` / CCU length register | Available | — | — |
| `jetty_id` | SQE table column / ccuChannels.jettyId array | Available | INVALID (non-network tasks) | — |
| `link_type` | opTaskMeta `HcclTaskProfInfo.linkType` (v3, enum name mapping Section 11.4.2) | Available (v3) | "" (v2 old BLOB without profInfo displays `Unknown`) | — (Section 11.4 consumed) |
| `protocol` | ccuChannels.protocol (CommProtocol enum name mapping) | Available | "" (when no channel info, trace displays `N/A`) | — (migrated from `link_type`, Section 11.1 approach a) |
| `src_eid` (local, write) | SQE `lcEid` (added in v2 collection, Section 4.4.1) preferred; fallback jettyId → local device outbound channel `leid` derivation | Available (v2) / Approximate (fallback) | Empty→any local channel leid fallback | — (v2 already supplemented explicit field) |
| `dst_eid` (peer) | Channel `reid` preferred, fallback to SQE `rmEid` | Approximate | "" (tasks without channel info) | Collection side supplements explicit peer eid |
| `src_rank` / `dst_rank` | taskData endpoint rankId (carried in deviceId field, Section 4.4.1 rank field rollback) directly consumed: write=src local/dst peer, read=src peer/dst local; CCU/unresolved falls back through eid→rank index (Section 4.8) | Available / index fallback | INVALID + `rank_unresolved` count when both explicit and index are unresolved | — (Sections 5.2/5.5/6.7) |
| `transport_type` | opTaskMeta `HcclTaskProfInfo.transportType` (v3, enum name mapping Section 11.4.3) | Available (v3) | "" (v2 old BLOB without profInfo displays `Unknown`) | — (Section 11.4 consumed) |
| `data_type` | REDUCE task-level `dataType` (added in v2, enum name mapping int8..bfp16); MEM_CPY task-level has no source | Partially available (v2) | REDUCE fills enum name; MEM_CPY "" | MEM_CPY collection side supplement |
| `port` | (reserved field) | Not filled | None | Simulator resolves ports through eid in real time (Section 7.6) |

**Instruction-level gaps (CCU microcode expansion)**: channel-type network task peer rank is already resolved by eid→rank index
(Sections 4.8/6.7); reduction tasks are always local (MS is rank-local buffer, Section 6.3.2), with no peer eid
requirement. UB tasks produced by CCU all explicitly carry `is_write` (Section 6.7), and direction determination does not depend on rank
values.

### 10.2 Task Type Mismatch Table

| db taskType / Old Type | New Model TaskType | Mismatch Handling |
|------------------------|-------------------|-------------------|
| SYNC_STREAM (taskType=6) | **No corresponding type** | Skip, no node generated (host-side stream sync point, not device-side task, Section 5.6); count `skipped_sync` |
| AIV_GRAPH (taskType=5) | None | Skip + warning (new and old models consistent, both unsupported) |
| WRITE_WITH_NOTIFY (taskType=7, db addition) | `WRITE_WITH_NOTIFY` (new addition, 7th type) | Single task mapping (**not split** into write + NOTIFY_RECORD): write always write direction uses port bandwidth sharing, attached notify uses fixed record delay (Section 5.2, Section 10.3 #19 supported) |
| Old `REMOTE_RECORD` / `REMOTE_WAIT` | Merged into `NOTIFY_RECORD` / `NOTIFY_WAIT` (src≠dst expresses remote) | Single-rank db path cross-rank notify generates remote node with `src≠dst` (Section 5.2, node preserved not paired) |
| Old `MEMCPY` (network transfer) | Semantic change: new `Memcpy` = HBM copy | Network transfer now expressed by `Ub_Write_Or_Read` (write/read direction distinguished); `Ub_Inline_Write` preserves only notify inline write semantics (fixed delay) |
| Old `LOCAL_COPY` | Split into `Memcpy` (pure copy) / `Reduce_Inline` (reduction) | REDUCE non-network → Reduce_Inline |
| Old `MemcpyDirection` enum (direction field) | Deleted | Direction expressed by task type + rank/eid values (Section 5.5) |
| Old `sync_stream_delay` config item | Cancelled with type gap | Merged into local_record_delay reuse |

### 10.3 Other Conflict Points (Future Matching)

1. ~~**`src_rank == 0` local determination hardcoding** (`simulator.py` TRANSFER branch): read-direction tasks where
   the peer happens to be rank 0 and is explicitly filled would be misidentified as write.~~ **Fixed**: Task adds explicit
   `is_write` field (filled by db graph construction path, task.txt old path None falls back to heuristic), simulator
   prioritizes consumption; taskData's peer rankId (carried in deviceId field) also directly fills src/dst_rank (Sections 5.2/5.5).
2. **TopoResolver has no fallback**: eid not found in ranktable throws error directly (no jettyId→port mapping,
   no default port). The db path avoids this through "network tasks exist and no ranktable → skip simulation, only dump"
   (Section 3.3); future work could consider integrating ccuChannels/JettyMaps to build a fallback mapping.
3. **Broadcast pairing triggers only the first wait** (matched record's `break`): when multiple waits pair with the same
   record, subsequent waits' Events are not triggered, end_time remains -1 (tasks silently lost).
4. **`is_matched_record` linear scan**: `dependency_graph.py` performs O(R×M)
   traversal over matches, with performance risk for large-scale dbs (tens of thousands of tasks); recommended to maintain a record→wait reverse index (simultaneously
   resolving #3).
5. **SimConfig unit system and port key changes**: JSON latency s→**us**, bandwidth B/s→**GBps**,
   port key int→**"die/port"** string; old configuration files are incompatible (sample already synchronized this iteration
   `src/plugin/light_performance/sim_config.json`).
6. **HBM dynamic sharing semantic change**: old version per-task independent `size / hbm_bandwidth`, new version
   Reduce_Inline+Memcpy concurrent shares HBM bandwidth evenly (manager with utilization always 1.0); result caliber
   differs.
7. **Utilization query basis `min_remaining`**: same-group concurrent tasks query utilization tier using minimum remaining data volume,
   different from old version's "each task queries by its own data_size" — small tasks within a group drag down large tasks' utilization tier.
8. **Old input sources removed**: task.txt parsing, AllToAll generator, built-in demo removed from input_builder
   with the HCCLSIM version; restoration needed for log replay debugging.
9. **TopoResolver has no net_type filtering**: no priority rules when the same eid has multiple network type entries.
10. **profiling json not implemented** (Sub-Requirement 5): current output is task graph dump (DOT/JSON/HTML)
    + text report + Chrome trace; perf_profiling.json format see Chapter 8 draft.
11. ~~**local_rank fallback convention**: when opDetails.rankId is missing, falls back by `deviceId−1`
    (following the task.txt path convention); deviceId and rank numbering have no inherent relationship, ranktable can
    provide precision.~~ Peer rank is already resolved by eid→rank index (RankEidMap primary path + ranktable
    fallback, Section 4.8), no longer dependent on numbering conventions; local side still uses opDetails.rankId as authoritative
    (channel leid cross-verification, Section 6.7).
12. ~~**CCU queue parallelism not modeled**: CCU queue chains execute as independent synthetic streams in parallel, real hardware
    queue concurrent resource constraints not modeled (checker side has queue parallelization transformation, this plugin Section 6.7 simplified).~~
    **Queue parallelization implemented** (Section 7.4.1: assembly-phase skeleton/unit splitting + fork/join/attachment extra edges +
    `#v` virtual streams reused across missions). Still not modeled: real hardware queue concurrent resource upper limits —
    virtual stream count is unlimited (checker V3 also unlimited), future work can cap by chip generation.
13. **@estimate one-shot channel does not pass data_id**: the `plugin run @estimate` special-case branch directly executes
    entry (`cmd_base_utils.cc:626-663`), without going through the general message path's `DumpData` (which generates
    `data_id`, currently a fixed value "1234"), so the output directory is named by runtime timestamp (Section 8.1); if
    switched to resident model in the future, data_id semantics can be aligned.
14. **`plugin install @estimate` semantic degradation**: install goes through `RegisterPlugin` resident startup
    (fork+pipe), main.py finishes one-shot execution and exits after stdin EOF, MonitorThread reclaims —
    behavior is harmless but does not match install's resident semantics; a stdin command loop must be implemented for true residency
    (Section 3.2 evolution direction).
15. **Plugin products do not participate in automatic archiving**: `archive.sh` archives only `logs/` and `data/`,
    `plugin/light_performance/data/` perf_profiling.json and other products do not enter the archive directory
    (benefit: not affected by `remove_all("data")` at start); archive.sh must be extended for archiving.
16. **simpy deployment dependency**: target environment must have simpy pre-installed (not distributed with hccl_vm_install as offline wheel),
    main.py should give a clear error and installation prompt when import fails at startup (Section 3.4).
17. ~~**db_reader.py not synchronized with v3 collection layout**: C-side `HcclTaskMetaData` has been extended to 188B (
    `HcclTaskProfInfo` 3 bytes added after taskType, Section 4.4.1), `db_reader.py` still decodes at 185B
    (`TASK_META_LEN=185`, `<BQIQQI16s16s`) — length gate `len(blob) >= 185` cannot block
    188B BLOBs, and reading v3-collected dbs causes **field-wide misalignment** (from commId onward). Needs synchronization to
    188B/`<BBBBQIQQI16s16s` and decoding of profInfo three fields (value ranges/names see Section 11.4).~~
    **Synchronized**: `db_reader.py` auto-identifies v3 (188B/`<BBBBQIQQI16s16s`,
    profInfo three fields into `HcclTaskMetaData`) and old v2 (185B) layouts by BLOB length, see Section 4.4.1 implementation status.
18. ~~**JSON task types not consuming profInfo** (remaining items narrowed): trace args
    `transport_type`/`link_type` already switched to profInfo-driven, CommProtocol migrated to new field
    `protocol` (Sections 11.4.2/11.4.3, Section 11.1 approach a landed); **remaining item** is `task_type`
    field — currently still derived from TaskType enum + jettyId/direction conditions (Section 11.2), which has **semantic misalignment** with collection-side
    profInfo's hcomm op names (such as `Ub_Inline_Write` bidirectionally referring to different
    tasks, Section 11.4.4); after switching to `profInfo.profTaskType` driven, can correspond one-to-one with msprof timeline
    op names, target state comparison (including correspondence with simulation TaskType) see Section 11.4.1.~~
    **Landed (all remaining items completed)**: `Task` adds `task_name` display name (`task_model.py`,
    `display_name` = `task_name or task_type.value`), `graph_builder._add_task`
    uniformly injects `prof_task_type_name(profInfo.profTaskType)`; trace event name/`cat`
    (`report.py export_trace`), `args.task_type` (`to_trace_args`), verbose
    detail and dump (DOT/JSON/HTML legend) all switched to op name. Internal `TaskType` retired as
    latency model classification (Section 11.4.5 #3), latency/pairing/charging zero changes. v2 old BLOB
    (profInfo=0xFF) displays `Unknown` (same caliber as link/transport); CCU microcode sub-tasks
    have no profInfo, fall back to internal name (Section 11.3). Misalignment elimination comparison see Section 11.4.4 implementation record.
19. ~~**WRITE_WITH_NOTIFY (taskType=7) decoding missing**: `db_reader.py`'s
    `HccLTaskMetaType` IntEnum only 0~6, `decode_op_task_meta` encountering 7 directly raises and discards
    (counted as warning).~~ **Supported (single task mapping)**: `db_reader.py` decodes
    `WriteWithNotifyTask` (Section 4.4.1) and passes through as single task, `graph_builder` maps to
    `WRITE_WITH_NOTIFY` TaskType (trace display name `Write_With_Notify`, Section 5.2),
    **one-to-one correspondence** with db tasks; latency = port write direction bandwidth sharing + post-write notify
    fixed record delay (Sections 7.3/7.4). **Split strategy not adopted** (checker/runner side's
    "split into [write, NOTIFY_RECORD] two sub-tasks", and this plugin's early decoding-boundary split both
    deprecated): splitting would break task granularity, making trace/dump/statistics unable to correspond one-to-one with db tasks.
    Collection-side profInfo=TASK_WRITE_(REDUCE_)WITH_NOTIFY (op name `Write_With_Notify` /
    `Reduce_With_Notify`, hcomm source has the latter misspelled as `Reuce_With_Notify`, see Section 11.4.5),
    both profInfo values map to the same `Write_With_Notify` task (isReduce only affects
    data_type display).
20. ~~**Same mission multiple dispatches' SQEs merged into a single context**: CCU expansion merges all SQEs with the same
    `(dieId, missionId, queue)` into one dispatch context (single instruction window
    [first instStartId, end), single SQE args consumption) — does not satisfy checker
    `IsContiguousCcuSqe` "contiguous-then-merge" constraint (`task_meta_translator_v3.cc:213-231`).
    Typical victim scenario: instruction window wrap-around re-delivery (head SQE 13 Load SqeArgs + body SQE sent twice in succession,
    second time with new args), under old implementation the second dispatch: ① window start counted by head SQE, end JMP
    jumps over entire data transfer instructions; ② SQE args already consumed by first pass, subsequent Loads take 0 value
    (Address all 0); ③ loop count/Event retains first pass state — second dispatch's
    Ub/Memcpy/local post entirely lost, network data volume only 3/10 of checker.~~
    **Fixed (contiguous segment splitting)**: split into multiple dispatch contexts by `instStartId`/`instCnt` continuity
    (Section 6.1), each context independently replays its window (including Load SqeArgs header) + independently sequentially
    consumes args (`GetSqe` semantics); registers/Event bitmap shared across contexts (`AllRankParamRecorder`
    semantics), same-queue contexts serialized by dispatch order, multi-segment task ids disambiguated with `_s{sequence}` suffix
    (`ccu_graph_builder.expand_ccu_graph`). After fix, aligned stream-by-stream with checker single-pass graph:
    record local/remote 54/54, wait local/remote 27/36, Ub 39/39, Memcpy 6/6,
    network+local data volume 1088MB (2333 verification environment ALLGATHER dev1).
    Same root-cause **wait node emission deviation** also fixed: old implementation split by "one wait node per contributing post"
    (one wait instruction could produce N nodes, not matching checker `GenWaitNode`'s "per instruction at most
    1 local + 1 remote node", manifesting in this environment as local wait 48 vs 27); now aggregated into a single node
    (mask = OR of all contributing post bits, first contributing post uses notify pairing, remaining through extra edges,
    Section 6.4), after fix local wait 27/27 aligned.
21. **Merged loop folded pseudo-serialization residual gaps**: Section 6.5.1 feature chain resolution covers the merge path's
    mid-body wait/record (folded pseudo-serial barriers), but two residuals remain: ① full expansion fallback path
    (`mf > 0`) per-iteration wait/record is real 4KB granularity handshaking, and its inter-iteration overlap is also
    serialized by the queue chain (large node count, not triggered by current samples, not processed for now); ② per-iteration handshaking
    **rate** overhead not modeled — both folding and resolution count as single occurrences (if Event engine throughput is the bottleneck, a specialized
    rate model is needed, such as ns/occurrence × 32768 occurrences scale evaluation).

---

## 11. db Task Type and JSON Product Task Type Mapping Table

> This chapter is a type mapping quick reference, answering three questions:
> ① How tasks in the db database (`opTask_P_<pid>` table taskType column) are ultimately converted to
> which task types in JSON products (Sections 11.1, 11.2); ② The correspondence between microcode instructions and JSON task types in CCU mode
> (Section 11.3); ③ The correspondence and target state between collection-side profiling reconciliation information (`HcclTaskProfInfo`) and JSON
> display types (Section 11.4). Implementation basis: `src/plugin/light_performance/
> graph_builder.py` (SQE mapping, `_map_tasks`/`_map_transfer`),
> `src/plugin/light_performance/ccu_graph_builder.py` (CCU instruction mapping, `transform`
> series), `src/plugin/light_performance/simulator/task_model.py` (TaskType enum).
> Same source as Section 5.2 (SQE mapping + latency model) and Section 6.3 (instruction category mapping), reorganized here by
> "db → JSON" correspondence, with reverse index added.
>
> **profInfo background**: From v3 collection onward, db tasks' opTaskMeta BLOB carries `HcclTaskProfInfo`
> sub-structure (Section 4.4.1, single-source definition in `include/sim_profiling_defs.h`, treated as opaque whole by checker/runner for
> pass-through), with three fields `profTaskType`/`linkType`/`transportType` whose value ranges and
> names align one-to-one with hcomm `TaskParamType`/`DfxLinkType`/`DfxTransportType` — that is,
> msprof timeline op names and L1 detail linkType/transportType fields. Sections 11.1~11.3 of this chapter describe
> **current TaskType implementation** (profInfo not yet consumed), and Section 11.4 provides the complete three-field comparison and switchover
> target state (to-do items see Section 10.3 #17~#18).

### 11.1 Task Type Carriers in JSON Products

The current implementation has three types of JSON products, all of which take task types from the `TaskType` enum values
(string form, `task_model.py:20-26`):

| JSON Product | Format | Task Type Field Location | Generation Module |
|-------------|--------|------------------------|-------------------|
| `perf_profiling.json` (main product, Section 8.1) | Chrome Trace Event | `traceEvents[].name` / `.cat` / `.args.task_type` | `simulator/report.py` (`export_trace`, event name is task display name: profInfo op name, Section 11.4) |
| `trace_comm<commId>_iter<opIter>.json` (optional per-operator trace) | Chrome Trace Event | Same as above | Same as above |
| Task graph dump `graph_comm<commId>_iter<opIter>_dev<dev>.json` (optional, `--dump-dir`) | `light_perf.graph_dump/2` | `nodes[].type` | `graph_dumper.py` |

> Dump JSON notify nodes carry `nodes[].mask` when mask is non-zero (Section 7.3 Task.mask),
> HTML view tooltip displays synchronously; `perf_profiling.json`/trace task args also contain
> `mask` (Section 7.7).

Complete set of task types appearing in JSON (7 types):

| TaskType Value (Presented in JSON) | Latency Model | Semantics |
|-----------------------------------|--------------|-----------|
| `Notify_Record` | Fixed delay (local additionally triggers Event) | notify set (record / CKE post) |
| `Notify_Wait` | Waits for Event / fixed delay | notify wait (wait / CKE wait) |
| `Reduce_Inline` | HBM dynamic sharing | Local reduction (computation approximated as transfer, Section 5.7) |
| `Memcpy` | HBM dynamic sharing | Local copy (HBM, not network) |
| `Ub_Inline_Write` | Fixed delay (same as Notify_Record) | Notify inline write (old input path only) |
| `Ub_Write_Or_Read` | Port bandwidth sharing (write/read directions charged independently) | Network data send/receive (SQE network tasks in both directions uniformly mapped) |
| `Write_With_Notify` | Port write direction bandwidth sharing + post-write notify fixed delay | Network write + attached notify (one-to-one with db `WRITE_WITH_NOTIFY`, Section 5.2) |

> The `perf_profiling.json` structured format (meta/ops array) in Section 8.3 is a draft; the current main product
> is actually in Chrome tracing format (Section 10.3 #10); regardless of format, task types are the above
> 7 TaskType values, and the mapping tables in this chapter apply uniformly.

**Type-related fields in trace args (current state and profInfo consumption)**: Task's `to_trace_args`
(`task_model.py`) outputs four type-related fields; all four fields have been switched to profInfo-driven
(Section 10.3 #18 all landed, Section 11.4):

| trace args Field | Current Source | Status |
|-----------------|---------------|--------|
| `task_type` | `profInfo.profTaskType` op name (`prof_task_type_name`, uniformly filled by `graph_builder._add_task` as `Task.task_name`; `display_name` falls back to internal TaskType name when empty) | **Landed** (one-to-one correspondence with msprof timeline op name, Section 11.4.1; v2 old BLOB is `Unknown`) |
| `transport_type` | `profInfo.transportType` name (`ProfTransportTypeName`, Section 11.4.3, uniformly filled by `graph_builder._add_task`) | **Landed** |
| `link_type` | `profInfo.linkType` name (`ProfLinkTypeName`, Section 11.4.2, uniformly filled by `graph_builder._add_task`) — **note this is a different concept from `protocol` (CommProtocol)** | **Landed** (approach a: overridden to DfxLinkType name) |
| `protocol` (new field) | Channel `CommProtocol` enum name (HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO, filled by `graph_builder._resolve_eids` / CCU `_channel_eids`) | **Landed** (CommProtocol migrated from `link_type`, `N/A` when no channel info) |

> The `link_type` override (approach a) has been implemented: semantics aligned with msprof L1 detail field name, avoiding same-name-different-meaning;
> existing CommProtocol information is not lost, moved to the `protocol` field. CCU microcode sub-tasks are simulator
> semantic products, have no profInfo (Section 11.3), and their `link_type`/`transport_type` are `N/A`,
> while `protocol` retains the channel CommProtocol name.

### 11.2 SQE Path: db taskType → JSON Task Type Mapping Table

Forward mapping (db task → JSON task display name). The "JSON task type" column is the trace/dump displayed
op name (switched to `profInfo.profTaskType` driven, Section 10.3 #18 landed); "internal TaskType"
(latency model classification, determines charging/pairing) maintains the original Section 11.2 judgment conditions (jettyId/direction/recordDev),
with correspondence to op names see Section 11.4.1 "Simulation TaskType" column:

| db taskType (opTask.taskType column) | Condition | JSON Task Type (op name) | Collection-side profInfo.profTaskType (op name) | Description |
|--------------------------------------|-----------|-------------------------|------------------------------------------------|-------------|
| `NOTIFY_WAIT` (0) | recordDev == local device | `Notify_Wait` | `TASK_NOTIFY_WAIT` → `Notify_Wait` ✓ | Local wait: participates in first-come-first-served pairing (Section 5.4), waits for Event |
| `NOTIFY_WAIT` (0) | recordDev != local device | `Notify_Wait` | `TASK_NOTIFY_WAIT` → `Notify_Wait` ✓ | Cross-rank wait: node preserved not paired, src≠dst → fixed `remote_wait_delay` |
| `NOTIFY_RECORD` (1) | waitDev == local device (SDMA/local notify) | `Notify_Record` | `TASK_NOTIFY_RECORD` → `Notify_Record` ✓ | Local record: participates in pairing, triggers Event |
| `NOTIFY_RECORD` (1) | waitDev != local device (SDMA/local notify) | `Notify_Record` | `TASK_NOTIFY_RECORD` → `Notify_Record` ✓ | Cross-rank post: node preserved not paired, src≠dst → fixed `remote_record_delay` |
| `NOTIFY_RECORD` (1) | UBDMA WQE `inlineEn==1` (notify inline write) | `Ub_Inline_Write` | `TASK_UB_INLINE_WRITE` → `Ub_Inline_Write` ✓ | Notify inline write (name already returned, Section 11.4.4); internal TaskType still Notify_Record (fixed record delay) |
| `REDUCE` (2) | jettyId == INVALID | `Reduce_Inline` | `TASK_REDUCE_INLINE` → `Reduce_Inline` ✓ | Local reduction: HBM dynamic sharing charging |
| `REDUCE` (2) | jettyId valid + write direction | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` → `Ub_Reduce_Inline` ✓ | Network reduction approximation (Section 5.7), internal TaskType=Ub_Inline_Write, port write direction charging |
| `REDUCE` (2) | jettyId valid + read direction | `Ub_Reduce_Inline` | `TASK_UB_REDUCE_INLINE` → `Ub_Reduce_Inline` ✓ | Same as above, internal TaskType=Ub_Write_Or_Read, read direction charging |
| `MEM_CPY` (3) | jettyId == INVALID (same device) | `Memcpy` | `TASK_SDMA` → `Memcpy` ✓ | Local copy: HBM dynamic sharing charging; profInfo.linkType=ONCHIP |
| `MEM_CPY` (3) | jettyId == INVALID (cross-device SDMA SQE) | `Memcpy` | `TASK_SDMA` → `Memcpy` ✓ | Cross-die SDMA transfer; same Memcpy, difference expressed by profInfo.linkType=PCIE / transportType=SDMA (Section 11.4) |
| `MEM_CPY` (3) | jettyId valid + write direction | `Ub_Write_Or_Read` | `TASK_UB` → `Ub_Write_Or_Read` ✓ | Network send (direction not encoded in type name, expressed by direction/is_write); internal TaskType=Ub_Inline_Write |
| `MEM_CPY` (3) | jettyId valid + read direction | `Ub_Write_Or_Read` | `TASK_UB` → `Ub_Write_Or_Read` ✓ | Network receive; internal TaskType=Ub_Write_Or_Read |
| `WRITE_WITH_NOTIFY` (7) | Always write direction (no direction inference, Section 5.5) | `Write_With_Notify` (single task, not split); `udfFlag==1` produces `Reduce_With_Notify` | `TASK_WRITE_WITH_NOTIFY` → `Write_With_Notify`; `udfFlag==1` produces `TASK_WRITE_REDUCE_WITH_NOTIFY` → `Reduce_With_Notify` (hcomm source has misspelling `Reuce_With_Notify`, Section 11.4.5) | Write+notify composite task (`WriteWithNotifyTask`, Section 4.4.1): port write direction bandwidth sharing + post-write notify fixed record delay; both profInfo values map to the same internal WRITE_WITH_NOTIFY (latency model identical, display name distinguished by isReduce) |
| `CCU_GRAPH` (4) | — | (Placeholder, not directly produced) | `TASK_CCU` → `Ccu` | Expanded into CCU subgraph, final products see Section 11.3 |
| `AIV_GRAPH` (5) | — | None (skipped) | `TASK_AIV` → `AivKernel` | Warning + `skipped_aiv` count (not supported at this time) |
| `SYNC_STREAM` (6) | — | None (skipped) | `UNKNOWN` → `Unknown` | Host-side stream sync point, not device-side task (Section 5.6), hcomm profiling has no corresponding type |

> Cross-rank determination (waitDev/recordDev) see Section 4.6; network direction determination (write/read) see Section 5.5.

Reverse index (JSON task display name ← all sources, including CCU path and profInfo dimension; SQE path
display name = profInfo op name, CCU path display name = internal TaskType name, Section 11.3):

| JSON Task Display Name | SQE Source (db taskType) | Collection-side profInfo Source (op name) | CCU Source (microcode instruction, Section 11.3) |
|----------------------|-------------------------|------------------------------------------|------------------------------------------------|
| `Notify_Record` | `NOTIFY_RECORD` (1), local/cross-rank (SDMA/local notify) | `Notify_Record` (TASK_NOTIFY_RECORD ✓) | SetCKE/SetCKBit local post; TRANS/REDUCE/LOAD instruction embedded `setCKE` field (A6 LOAD type also carries); cross-rank set/notify (A5 Sync series setRmtCKE; A6 TransMem notify/0x1A atomic, SyncWtX notify, SyncAtx — remote approximation); subgraph attachment launch-record / completion-record |
| `Notify_Wait` | `NOTIFY_WAIT` (0), local/cross-rank | `Notify_Wait` ✓ | Various instructions' embedded `waitCKE` field; ClearCKE; A6 independent Wait instruction (Xn value comparison, unsatisfied → remote wait); subgraph attachment completion-wait / launch-wait |
| `Reduce_Inline` | `REDUCE` (2), jettyId invalid | `Reduce_Inline` ✓ | REDUCE type instructions (msId list all local) |
| `Memcpy` | `MEM_CPY` (3), jettyId invalid (including cross-die SDMA) | `Memcpy` ✓ | TRANS local transfer instructions (A5 4 codes: 0x0/0x2/0x5/0xA; A6 4 codes: 0x0/0x2/0x5/0x6) |
| `Ub_Inline_Write` | `NOTIFY_RECORD` (1) + `inlineEn==1` (notify inline write) | `Ub_Inline_Write` (TASK_UB_INLINE_WRITE ✓, name returned) | — (CCU network transfer uniformly mapped to `Ub_Write_Or_Read`, Section 11.3.1) |
| `Ub_Write_Or_Read` | `MEM_CPY` (3) / `REDUCE` (2), jettyId valid (write/read direction) | `Ub_Write_Or_Read` (TASK_UB ✓, direction expressed by direction/is_write) | TRANS network transfer instructions (A5 write role 3 codes: 0x3/0x7/0x9, read role 4 codes: 0x1/0x4/0x6/0x8; A6 unified TransMem 0x10: dmaOpCode 3/5=write, 6=read, including udfEnable network reduction approximation); REDUCE type with remote MS |
| `Ub_Reduce_Inline` | `REDUCE` (2), jettyId valid (network reduction) | `Ub_Reduce_Inline` (TASK_UB_REDUCE_INLINE ✓) | — |
| `Write_With_Notify` | `WRITE_WITH_NOTIFY` (7), isReduce=0 | `Write_With_Notify` (TASK_WRITE_WITH_NOTIFY ✓) | — |
| `Reduce_With_Notify` | `WRITE_WITH_NOTIFY` (7), udfFlag==1 | `Reduce_With_Notify` (TASK_WRITE_REDUCE_WITH_NOTIFY ✓) | — |

### 11.3 CCU Path: Microcode Instruction → JSON Task Type Mapping

The `CCU_GRAPH` (4) SQE in the db is itself only a placeholder (`CcuPlaceholder`, Section 6.1); the final
JSON tasks are produced by microcode interpretation from the `ccuInstrRes` instruction space (Chapter 6). **A5 (V1,
Ascend950, devType=6) and A6 (V2, Ascend960, devType=8) both instruction set generations are implemented**
(selected by `opDetails.devType`, other devTypes fall back to A5): A5 summary table see Section 11.3.1,
A6 summary table see Section 11.3.3.

> profInfo correspondence: CCU placeholder SQE's `HcclTaskProfInfo` is entirely
> `TASK_CCU`→`Ccu` (linkType/transportType are UNKNOWN, microcode has no link dimension). The expanded
> microcode sub-tasks (`Memcpy`/`Ub_Write_Or_Read`/`Notify_*` and so on) are
> simulator semantic products, **have no corresponding ProfTaskParamType**, and display names fall back to internal TaskType names
> (established trade-off at Section 10.3 #18 implementation time, Section 11.4.4); msprof side CCU tasks through
> microcode-level detail (`CcuProfilingInfo`: CCU_TASK/WAITCKE/LOOPGROUP three types, reported with TASK_CCU)
> express equivalent granularity information, and reconciliation uses the whole SQE's `Ccu` as alignment anchor.

#### 11.3.1 A5 Instruction Summary Table (Instruction → JSON Task Type)

Microcode instruction = 2B header (type:4bit category + code:11bit instruction code, Section 6.2) + 30B payload:

| Instruction Category (type) | Instruction (code) | JSON Task Type | Description |
|---------------------------|-------------------|---------------|-------------|
| LOAD (0x0) | `LoadSqeArgsToGSA` (0x0), `LoadSqeArgsToXn` (0x1), `LoadImdToGSA` (0x2), `LoadImdToXn` (0x3), `LoadGsaXn` (0x4), `LoadGsaGsa` (0x5), `LoadXx` (0x6) | **None** (no node generated) | Only reads/writes register model (xn/gsa), for subsequent instructions to get length, loop count, and so on |
| CTRL (0x1) | `SetCKE` (0x2) | `Notify_Record` | Local Event post (ckeId as notify_id, ckeMask as mask, src=dst=local); embedded `waitCKE` field can additionally produce `Notify_Wait` |
| CTRL (0x1) | `ClearCKE` (0x4) | `Notify_Wait` | wait semantics (embedded `waitCKE` field); clear action itself does not produce a node |
| CTRL (0x1) | Loop (0x0) | **None** | Must be triggered by LoopGroup; appearing alone is abnormal, warning and skip |
| CTRL (0x1) | LoopGroup (0x1) | (No direct node) | Loop control: merge/expand loop body (Section 6.5), body instructions mapped per this table; under merge path, data task size × total iterations, and emitted tasks carry loop_cnt (loop count after expansion) / loop_iter_cnt (single loop self-iteration count), total iterations = product of the two fields |
| CTRL (0x1) | Jmp (0x5) | **None** | Conditional jump: when `Xn[conditionXnId] != expectData`, pc jumps to instruction pointed by `Xn[dstInstrXnId]` (checker `TransformJmpInstrA5` semantics, target limited to current instruction window, out-of-range warning pass-through); pass-through within loop body (merge precondition verification already rejected loop bodies containing JMP); step budget defense against backward jump infinite loops |
| TRANS (0x2) | LocMemToLocMs (0x0), LocMsToLocMem (0x2), LocMsToLocMs (0x5), LocMemToLocMem (0xA) | `Memcpy` (src/dst_buf_type labeled mem/ms) | Local transfer (LOCAL_COPY role, Section 6.3.2), HBM dynamic sharing |
| TRANS (0x2) | LocMsToRmtMem (0x3), LocMemToRmtMem (0x9), LocMsToRmtMs (0x7), RmtMemToLocMs (0x1), RmtMsToLocMem (0x4), RmtMsToLocMs (0x6), RmtMemToLocMem (0x8) | `Ub_Write_Or_Read` (is_write distinguishes direction, src/dst_buf_type labeled mem/ms) | Cross-die/card transfer (write/read roles unified type, Section 6.3.2); peer rank resolved through channel reid → eid→rank index (Sections 4.8/6.7) |
| TRANS (0x2) | `SyncCKE` (0xB), `SyncGSA` (0xC), `SyncXN` (0xD) | Local post portion → `Notify_Record`; remote post portion → `Notify_Record` (remote, not paired) | Remote set fields (`SyncCKE`'s `rmtCKEId/locCKEMask`, `SyncGSA/SyncXN`'s `setRmtCKEId/setRmtCKEMask`) generate remote record nodes (src=local, dst=INVALID → fixed `remote_record_delay`, not registered in local accounting), counted in `remote_posts` (Section 6.6); `SyncGSA/SyncXN` carrying second remote post also generates; embedded `waitCKE` can produce `Notify_Wait` |
| REDUCE (0x3) | Add (0x0), Max (0x1), Min (0x2) | `Reduce_Inline` | Reduction computation approximated as transfer (Section 5.7); CcuBuffer is always rank-local buffer (msId high bits are addressing encoding only, Section 6.3.2), local HBM charging |
| (Subgraph attachment) | — (synthesized at graph construction phase, not microcode) | None (no task node produced) | Attachment dependencies all expressed through extra edges (Section 7.4.1): entry edges SQE predecessor→skeleton head/pre-skeleton unit head, exit edges mission completion set→SQE successor; task graph does not produce any nodes from nothing |

#### 11.3.2 Embedded Synchronization Fields (Cross-Instruction Common)

TRANS / REDUCE / `SetCKE` / Sync type instruction payloads generally carry synchronization fields, **additively producing**
JSON tasks with the main semantics:

| Payload Field | JSON Task Type | Behavior |
|--------------|---------------|---------|
| `waitCKEId` / `waitCKEMask` | `Notify_Wait` | When bits are satisfied, pairs with local post accounting (aggregated into single node: mask=OR of all contributing post consumed bits, first post add_match, remaining extra edges, Section 10.3 #20); when bits insufficient: top-level instruction blocks the queue (stall detection finds and generates remote wait node then replays instruction remainder, `remote_waits`); within loop body cannot block, directly generates remote wait node and continues |
| `setCKEId` / `setCKEMask` | `Notify_Record` | Local Event post, registers seen_post accounting (Section 6.4) |
| `clearType` bit / SetCKE clear flag | None (register clear only) | Clears bits after wait, no node produced |

> Therefore one TRANS instruction can produce 1 data transfer task (`Memcpy`/`Ub_Write_Or_Read`),
> and additionally produce several `Notify_Wait` (by bit pairing,
> 0..n) and 1 `Notify_Record` based on embedded fields. Under merged loops (Section 6.5), each slot similarly preserves
> wait/set tasks, only data tasks are folded to size × total iterations.
>
> **A6 difference**: embedded `waitCKE` exists only in SetCKBit/ClearCKE (CTRL type); local
> TRANS/REDUCE and TransMem/Sync types carry only tail `setCKEId/setCKEMask` (@26/28,
> additionally producing `Notify_Record`); wait semantics are handled by independent Wait instructions (Xn value comparison).

#### 11.3.3 A6 (Ascend960) Instruction Summary Table (Instruction → JSON Task Type)

devType=960 (`DEV_TYPE_960`) corresponds to V2/A6 instruction set, **implemented** (ported from checker
`ccu_task_transform_instruct_a6_v3.cc`, interpreter `_MissionContextA6`). Microcode instruction
= 2B header (isomorphic with A5) + 30B V2 payload (`ccu_microcode_v1.h` CcuV2 struct,
pack(1)); register numbers generally through `GetXnId` (bit15 addressing mode, Section 6.4).

| Instruction Category (type) | Instruction (code) | JSON Task Type | Description |
|---------------------------|-------------------|---------------|-------------|
| LOAD (0x0) | LoadSqeArgsToX (0x1), LoadImdToX (0x2), LoadX (0x6), StoreX (0x7), ClearX (0x8), Nop (0x9), arithmetic-logic Add (0xD)/Sub (0xE)/Mul (0xF)/And (0x10)/Or (0x11)/Not (0x12)/Xor (0x13)/Shl (0x14)/Shr (0x15)/Popcnt (0x16) | Data plane **none**; `setCKEMask != 0` additionally produces `Notify_Record` | Unified Xn register read/write (no GSA): SQE args sequentially consumed (Section 6.1), immediate load, indirect read/write (LoadX/StoreX, oMode 0=immediate offset/1=register offset), range clear (ClearX), 64-bit wrap-around arithmetic-logic (Mul takes lower 32-bit product, shift amount 0x3F truncated, Shl/Shr supports logical/arithmetic/circular three types); all except Nop carry tail setCKE field (checker `ProcessSetMask`, non-zero mask → local post node) |
| LOAD (0x0) | Load (0xA), Store (0xB) | Data plane **none**; setCKE additionally produces `Notify_Record` | HBM ↔ Xn data plane: HBM content not modeled (checker reads/writes data through AllRankParamRecorder), registers not backfilled, warning logged; only synchronization semantics preserved |
| CTRL (0x1) | `SetCKBit` (0x2) | `Notify_Record` (+ embedded `waitCKE` additionally produces `Notify_Wait`) | wait → local post → clearType==1 clears wait bits (checker supports only clearType=1, others warning skip clear) |
| CTRL (0x1) | `ClearCKE` (0x4) | `Notify_Wait` (embedded `waitCKE`) | After wait, clears consumed wait bits per checker `ClearWaitMask` semantics; `clearMask != 0` is checker unsupported scenario (warning + tolerant clear of target bits) |
| CTRL (0x1) | Loop (0x0) | **None** | Must be triggered by LoopGroup; appearing alone triggers warning skip. `mode==1` conditional Loop: expands only when `wishCKEBit == Xp.loopCtxId + 1024` is hit (Section 6.5) |
| CTRL (0x1) | LoopGroup (0x1) | (No direct node) | Loop control: parameters from Variable/Xm/Xp three registers (loopInsCnt:10/expandOffset:9/expandCnt:9 and so on, Section 6.5); merge/expand same as A5 (merge verification includes A6 additional rules: wait Event ID not reused, cross-expansion iteration count/length/channel consistent); illegal configuration (loopInsCnt>512 and so on) warning skip |
| CTRL (0x1) | Jmp (0x5) | **None** | Conditional jump: conditionType 0~5 (==/!=/>/>=/</<=), ≥6 unconditional; jumpMode 0=relative pc (mod 0x10000)/1=absolute instruction number, target takes `Xn[relTarInstrXnId]`; limited to current instruction window, out-of-range warning pass-through; pass-through within loop body; step budget defense against infinite loops |
| CTRL (0x1) | Wait (0x7) | Remote `Notify_Wait` when unsatisfied | Variable value comparison spin (conditionType same as Jmp, default equals): satisfied pass-through no node; unsatisfied = depends on peer remote register write → top-level stall resolution/within loop body directly generates remote wait node (fixed `remote_wait_delay`, task id tagged `xwt`, counted in `remote_waits`) |
| CTRL (0x1) | Fence (0x8) | **None** | Checker does not support (error abort); here warning pass-through (queue serial model naturally guarantees order) |
| TRANS (0x2) | TransLocMemToLocMS (0x0), TransLocMSToLocMem (0x2), TransLocMSToLocMS (0x5), TransLocMemToLocMem (0x6) | `Memcpy` (src/dst_buf_type labeled mem/ms) | Local transfer (length `Xn[GetXnId(xlId)]`); no embedded `waitCKE`, tail `setCKE` additionally produces `Notify_Record` |
| TRANS (0x2) | TransMem (0x10) | `Ub_Write_Or_Read` (dmaOpCode 3/5=write, 6=read); notify/atomic → remote `Notify_Record` | Unified network transfer: channel = `Xn[GetXnId(xcId)]` register value → peer rank/eid resolution same as A5 (Sections 4.8/6.7); when `msIdMode==1`, write source / read destination is local CcuBuffer (buf_type=ms); dmaOpCode 5 post-write notify and 0x1A remote atomic count (no data plane) approximated as remote record (mask=notification value lower 16 bits); `udfEnable` network reduction charged as ordinary UB transfer (Section 5.7 approximation); 0x19 and other dmaOpCode / `udfType != 0` warning skip data plane (`setCKE` post still executes) |
| TRANS (0x2) | `SyncXnWtx` (0xD), `SyncAtx` (0xE) | Remote `Notify_Record` (approximation); `SyncAtx` also has `setCKE` → local `Notify_Record` | Cross-rank register sync: `SyncWtX`'s `notifyValid==1` notify and `SyncAtx`'s atomic write approximated as remote record (remote target type unresolvable in single rank, Section 6.6); main value sync is remote register write, no node built (same caliber as A5 `SyncXn`); `SyncWtX` tail `setCKE` field not consumed by checker (local does not post); unresolvable channel warning skip (`eid_unresolved`) |
| REDUCE (0x3) | ReduceAdd (0x0), ReduceMax (0x1), ReduceMin (0x2) | `Reduce_Inline` | Local CcuBuffer batch reduction (msId[8] list + `Xn[XnIdLength]` length); no embedded `waitCKE`, tail `setCKE` additionally produces `Notify_Record`; CcuBuffer always local HBM charging (Section 6.3.2) |
| (Subgraph attachment) | — (synthesized at graph construction phase, not microcode) | None (no task node produced) | Same as A5 (Section 7.4.1 extra edge attachment) |

> Established deviations between A6 and checker (HBM data plane, remote target type resolution, checker error scenario
> tolerant downgrades, and so on) summarized in `_MissionContextA6` class docstring and Section 6.7 comparison table;
> loop merge/pipeline concurrency restoration (Sections 6.5/6.5.1), queue parallelization (Section 7.4.1), SQE grouping
> and stall resolution (Sections 6.1/6.6) and other framework mechanisms are shared across both generations.

### 11.4 HcclTaskProfInfo Three-Field Comparison (profInfo → JSON Display)

From v3 collection onward, db tasks carry `HcclTaskProfInfo` (Section 4.4.1, single-source definition
`include/sim_profiling_defs.h`). This section provides the complete "value → enum → JSON display string"
comparison for all three fields and the correspondence between each value and the simulation TaskType (latency model classification, Section 7.3);
C++ side name functions (`ProfTaskParamTypeName`/`ProfLinkTypeName`/
`ProfTransportTypeName`) are the sole authoritative source for display strings.

> **Implementation status**: `task_type`/`transport_type`/`link_type`/`protocol` four fields are all landed per
> this section's comparison (Python-side display name mapping in `db_reader.py`'s `PROF_*_NAMES`, character-by-character consistent with C++
> name functions, Section 11.4.5; `graph_builder._add_task` uniformly fills — `task_type`
> through `Task.task_name`/`display_name`, Section 10.3 #18 completed).

#### 11.4.1 profTaskType (ProfTaskParamType, 0~22 → msprof timeline op name)

This table supplements the correspondence between each value and the simulation TaskType (`task_model.py`, latency model classification) — previously
db tasks and simulation task types were matched through the db taskType column (Section 11.2); after profInfo consumption, this table's
`profTaskType → simulation TaskType` serves as the second matching dimension (judgment conditions unchanged, same db taskType
routed to different TaskType by jettyId/direction as noted inline):

| Value | Enum | JSON Display String (op name) | Simulation TaskType (latency model classification) | light_perf Occurrence Scenario |
|------|------|------------------------------|--------------------------------------------------|------------------------------|
| 0 | TASK_SDMA | `Memcpy` | `Memcpy` (jettyId invalid, HBM charging) / `Ub_Inline_Write`·`Ub_Write_Or_Read` (jettyId valid by direction, port charging) | Local copy, cross-die SDMA SQE (linkType distinguishes, Section 11.4.2) ✓ |
| 1 | TASK_RDMA | `RDMASend` | `Memcpy` (jettyId invalid, HBM charging approximation) | DPU host ibv ordinary read/write (`RDMASend`, same caliber as hcomm UbMemTransport::Write/Read) ✓ |
| 2 | TASK_REDUCE_INLINE | `Reduce_Inline` | `Reduce_Inline` (jettyId invalid, HBM charging) / `Ub_Inline_Write`·`Ub_Write_Or_Read` (network approximation, Section 5.7, port charging) | Local reduction ✓ |
| 3 | TASK_REDUCE_TBE | `Reduce_TBE` | — (does not appear) | 910 generation reserved, does not appear |
| 4 | TASK_NOTIFY_RECORD | `Notify_Record` | `Notify_Record` | SDMA/local notify record ✓ |
| 5 | TASK_NOTIFY_WAIT | `Notify_Wait` | `Notify_Wait` | notify wait ✓ |
| 6/7 | TASK_SEND_NOTIFY / TASK_SEND_PAYLOAD | `Send_Notify` / `Send_Payload` | — (does not appear) | hcomm reserved, does not appear |
| 8 | TASK_WRITE_WITH_NOTIFY | `Write_With_Notify` | `Write_With_Notify` | WRITE_WITH_NOTIFY (7) whole task (single task mapping, Section 10.3 #19 supported) ✓ |
| 9 | TASK_WRITE_REDUCE_WITH_NOTIFY | `Reduce_With_Notify` (tool-side corrected spelling; hcomm source/msprof has misspelling `Reuce_With_Notify`, Section 11.4.5) | `Write_With_Notify` (isReduce=1 reduction write payload, data_type distinguishes display) | WRITE_WITH_NOTIFY (7) isReduce=1 |
| 10 | TASK_CCU | `Ccu` | (CCU placeholder whole SQE, expanded into Chapter 6 subgraph, Section 11.3) | CCU placeholder whole SQE ✓ |
| 11/12 | TASK_AICPU_KERNEL / TASK_AICPU_REDUCE | `AicpuKernel` / `Aicpu_Reduce` | — (not consumed) | AICPU kernel delivery (through device process, currently no collection point/not consumed) |
| 13 | TASK_AIV | `AivKernel` | — (skipped) | AIV kernel task (skipped) |
| 14 | TASK_UB_INLINE_WRITE | `Ub_Inline_Write` | `Notify_Record` (notify fixed record delay; display name returned to op name ✓) | UBDMA `inlineEn==1` notify inline write |
| 15 | TASK_UB_REDUCE_INLINE | `Ub_Reduce_Inline` | `Ub_Inline_Write` (write) / `Ub_Write_Or_Read` (read) (network reduction approximation, port charging) | UBDMA `udfFlag==1` network reduction |
| 16 | TASK_UB | `Ub_Write_Or_Read` | `Ub_Inline_Write` (write direction) / `Ub_Write_Or_Read` (read direction) (direction expressed by direction/is_write) | UBDMA ordinary read/write WQE (write/read both directions same value) |
| 17~22 | TASK_DPU_KERNEL / THREAD_FENCE / CHANNEL_FENCE / INLINE_WRITE / NOTIFY_WAIT / WRITE_WITH_NOTIFY | `DpuKernel` / `Dpu_ThreadFence` / `Dpu_ChannelFence` / `Dpu_Notify_Record` / `Dpu_Notify_Wait` / `Dpu_Write_With_Notify` | `Dpu_Notify_Record`→`Notify_Record`, `Dpu_Notify_Wait`→`Notify_Wait` (cross-rank fixed delay), `Dpu_Write_With_Notify`→`Write_With_Notify` (single task port charging, Section 11.4.6); `DpuKernel`/Fence types not produced by current collection path | host CPU RoCE (ibv_dpu_stub) tasks ✓ (Section 11.4.6) |
| 0xFF | UNKNOWN | `Unknown` | — (SYNC_STREAM and so on already skipped, no node produced) | SYNC_STREAM, level1 channel API tasks, v2 old BLOB and other unlabeled sources |

> `task_type` has been switched to op name (Section 10.3 #18 completed); this table's "Simulation TaskType" column is now
> an internal latency model classification index (Section 11.4.5 #3); ✓ rows have op name consistent with TaskType value.
> After SQE batch merge (Section 5.1.1), the merged task's op name and link/transport take the **last**
> data task's profInfo within the batch (description follows last task, length takes full batch cumulative sum); v3 and earlier BLOBs without
> sqeGroupId are not merged, displayed per task.

#### 11.4.2 linkType (ProfLinkType, 0~9 → hcomm DfxLinkType Enum Name)

| Value | Enum | JSON Display String | light_perf Occurrence Scenario |
|------|------|--------------------|-------------------------------|
| 0 | LINK_ONCHIP | `ONCHIP` | Local Memcpy/Reduce_Inline/Notify_* |
| 1 | LINK_HCCS | `HCCS` | Reserved (current collection path does not produce) |
| 2 | LINK_PCIE | `PCIE` | Cross-die SDMA SQE (same-device determination classified as local ONCHIP, Section 4.4.1 collection rules) |
| 3 | LINK_ROCE | `ROCE` | host CPU RoCE (DPU, ibv_dpu_stub) tasks ✓ (Section 11.4.6) |
| 4/5/6 | LINK_SIO / HCCS_SW / STANDARD_ROCE | `SIO` / `HCCS_SW` / `STANDARD_ROCE` | Reserved |
| 7 | LINK_UB | `UB` | Network tasks (UBDMA WQE all types) |
| 8 | LINK_UBoE | `UBoE` | UBoE channel (collection side does not yet distinguish UB/UBoE, does not appear yet) |
| 9 | LINK_RESERVED | `RESERVED` | Reserved |
| 0xFF | UNKNOWN | `Unknown` | CCU/AIV/SYNC_STREAM, level1 channel API tasks |

> **Implementation status**: trace args `link_type` field filled per this table (`prof_link_type_name`,
> uniformly injected by `graph_builder._add_task`, Section 11.1 approach a); this table and the `CommProtocol` channel protocol previously carried by
> `link_type` (HCCS/TCP/ROCE/UB_CTP/UB_TP/PCIE/SIO) are **two different
> concepts**: CommProtocol describes channel establishment protocol, DfxLinkType describes task execution link (profiling
> dimension). CommProtocol name has been migrated to the `protocol` field (`N/A` when no channel info).

#### 11.4.3 transportType (ProfTransportType, 0~4 → msprof L1 Detail transportType Value Range)

| Value | Enum | JSON Display String | light_perf Occurrence Scenario |
|------|------|--------------------|-------------------------------|
| 0 | TRANSPORT_SDMA | `SDMA` | Cross-die SDMA SQE |
| 1 | TRANSPORT_RDMA | `RDMA` | Reserved (910 generation RDMA semantics) |
| 2 | TRANSPORT_LOCAL | `LOCAL` | Local tasks (Memcpy/Reduce_Inline/Notify_*) |
| 3 | TRANSPORT_UB | `UB` | Network tasks (**latency model anchor**: port bandwidth sharing charging, Section 5.5) |
| 4 | TRANSPORT_ROCE | `ROCE` | host CPU RoCE (DPU, ibv_dpu_stub) tasks ✓ (Section 11.4.6) |
| 0xFF | UNKNOWN | `Unknown` | Unlabeled tasks |

> **Implementation status**: trace args `transport_type` field filled per this table
> (`prof_transport_type_name`, uniformly injected by `graph_builder._add_task`).
> Same-named `Memcpy` tasks' local/cross-die difference is expressed by linkType+transportType:
> ONCHIP+LOCAL (local) vs PCIE+SDMA (cross-die) — trace args can now losslessly distinguish these two types of
> previously indistinguishable tasks in JSON.

#### 11.4.4 Ub_Inline_Write Bidirectional Semantic Misalignment and Switchover Implementation Record

**Historical problem**: Before the switchover, JSON's `Ub_Inline_Write` (network write direction send) and hcomm
profiling's `Ub_Inline_Write` (TASK_UB_INLINE_WRITE, notify inline write) **referred to
different tasks**, with mutual reverse misalignment:

| Task (Collection Semantics) | db taskType | ProfTaskParamType (op name) | Pre-Switchover JSON Type |
|---------------------------|-------------|---------------------------|------------------------|
| UBDMA ordinary write WQE (network send, write direction) | MEM_CPY (3) | `TASK_UB` → `Ub_Write_Or_Read` | `Ub_Inline_Write` ← name misalignment |
| UBDMA network reduction (write/read direction) | REDUCE (2) | `TASK_UB_REDUCE_INLINE` → `Ub_Reduce_Inline` | `Ub_Inline_Write` / `Ub_Write_Or_Read` ← name misalignment |
| UBDMA inline write notify (`inlineEn==1`) | NOTIFY_RECORD (1) | `TASK_UB_INLINE_WRITE` → `Ub_Inline_Write` | `Notify_Record` ← name misalignment |

**Implemented (Section 10.3 #18 completed)** — trace/dump display name switched to `profInfo.profTaskType`
op name (`Task.task_name`, uniformly injected by `graph_builder._add_task`), all three misalignments eliminated:
1. Network data tasks (TASK_UB) write/read uniformly display `Ub_Write_Or_Read`; **direction no longer encoded
   in type name**, expressed by trace args existing `direction` (write/read) and `is_write` fields
   — latency model unchanged, still charged by port write/read direction bandwidth sharing respectively (Section 5.5).
2. `Ub_Inline_Write` name **returned** to notify inline write tasks (db taskType=
   NOTIFY_RECORD + inlineEn==1), one-to-one correspondence with msprof timeline.
3. Network reduction displays `Ub_Reduce_Inline` (TASK_UB_REDUCE_INLINE), preserving distinguishable reduction semantics.
4. `Notify_Record`/`Notify_Wait`/`Reduce_Inline`/`Memcpy` four types already consistent with op names,
   switchover produces no change (`Memcpy` corresponds to TASK_SDMA).

**Implementation trade-offs (established decisions)**:
- v2 old BLOB (no profInfo, profTaskType=0xFF) displays `Unknown` — same caliber as
  `link_type`/`transport_type` v2 behavior (strict semantics, no internal name fallback).
- CCU microcode sub-tasks (Section 11.3) have no profInfo, display name falls back to internal TaskType name — simulator
  semantic products have no reconciliation requirement, maintaining status quo (network transfer unified as `Ub_Write_Or_Read`, direction expressed by
  `direction`/`is_write`, Section 11.3.1).
- Internal `TaskType` enum retired as latency model classification (Section 11.4.5 #3), simulation/pairing/charging zero changes.

#### 11.4.5 String Consistency Convention (Reconciliation Basis)

1. **Python-side display strings must be character-by-character consistent with C++ side three name function outputs**
   (`sim_profiling_defs.h` single source), including hcomm source code's existing spellings
   (`Ub_Write_Or_Read`, `Dpu_Notify_Record`, and so on) — this is the basis for field-by-field reconciliation with msprof timeline /
   L1 detail.
   **Sole exception**: `TASK_WRITE_REDUCE_WITH_NOTIFY`'s op name unified on tool side as corrected spelling
   `Reduce_With_Notify` (hcomm source has misspelling `Reuce_With_Notify` in two locations:
   `profiling_common.h:42` and `dfx_profiling_handler_lite.cc:67`, msprof raw
   data displays the misspelled version); cross-system reconciliation encountering this string requires
   `Reuce_With_Notify ↔ Reduce_With_Notify` difference mapping.
2. trace args field-level switchover (**all landed**, Section 10.3 #18 completed):
   `transport_type` ← transportType name, `link_type` ← linkType name
   (uniformly injected by `graph_builder._add_task`, value range tables see Sections 11.4.2/11.4.3), CommProtocol
   migrated to new field `protocol` (approach a, Section 11.1); `task_type` ← profTaskType op name
   (`Task.task_name`/`display_name`, Section 11.4.4 implementation record).
3. `TaskType` enum retired as internal latency model classification (`task_name` handles display duties),
   trace event name (`traceEvents[].name`) and `args.task_type` both switched to op name;
   CCU microcode sub-tasks/CLI input have no profInfo, `display_name` falls back to internal name.

#### 11.4.6 DPU (host CPU RoCE) Task Collection and Merge

Host DPU transfers (hcomm `HostCpuRoceChannel`, pod networking cross-node path) ibv hijack tasks
(`ibv_dpu_stub.cc`) previously had no profInfo labeling (displayed `Unknown`), WriteWithNotify split into two
tasks, and peer resolution looped back to local. This section is the alignment implementation record for this path.

**profInfo labeling** (aligned with hcomm `host_cpu_roce_channel.cc` DFX caliber, link/transport
always ROCE/ROCE):

| ibv Operation | db taskType | profTaskType (op name) |
|--------------|-------------|----------------------|
| Pure imm notify (len=0 WRITE_WITH_IMM / hybrid) | NOTIFY_RECORD | `TASK_DPU_INLINE_WRITE` → `Dpu_Notify_Record` |
| Write+imm (len>0 WRITE_WITH_IMM) | **WRITE_WITH_NOTIFY (7) single task** | `TASK_DPU_WRITE_WITH_NOTIFY` → `Dpu_Write_With_Notify` |
| Pure RDMA write/read (no imm) | MEM_CPY | `TASK_RDMA` → `RDMASend` (aligned with hcomm `UbMemTransport::Write/Read` precedent) |
| poll_cq pairing | NOTIFY_WAIT | `TASK_DPU_NOTIFY_WAIT` → `Dpu_Notify_Wait` |

**WriteWithNotify merged single task**: The whole operation produces only one taskType=7 task
(`WriteWithNotifyTaskData`: src/dst rank, offsets, len, notifyId, protocol),
aligned with hcomm whole operation reporting only one `Dpu_Write_With_Notify` caliber; `DpuPendingNotify`
pairing mechanism unchanged. Simulation side `_map_write_with_notify` uses port write direction bandwidth sharing
(local RoCE eid → ranktable `d2h` port; falls back to
`default_port_bandwidth` when `d2h` port bandwidth is not configured).

**Real peer resolution (phantom echo problem, socket timing binding)**: In estimate mode, the socket layer constructs
phantom peers (local Send mirrored back as Recv, `hccp_ra_socket_stub.cc`), and the QP handshake exchange returns
"remote gid" which is actually the local self; `RaTypicalQpModify`'s gid→IP resolution returns to local. Fix
(zero-byte read, relies only on API call timing):
- `RaGetSockets` resolves real peer EndPoint from `SocketInfoT.remoteIp` (hcomm link configuration, both client/server roles
  carry it) and stores in FdInfo (peerDeviceId/peerEid/localEid);
- `RaGetQpAttr` anchor (hccp_stub, executed only within estimate branch) records QP pending binding
  (`DpuRecordPendingBindQp`, `bindAttempted` prevents Modify from repeatedly querying attr and mis-binding);
- Same thread's next `RaSocketSend` (that is, hcomm ExchangeData's attr pack→Send timing) through
  `BindDpuPendingQps` binds the QP to that socket's real peer: `DpuQpInfo` adds
  `peerRankId`/`lcEid` (peer rank resolved from FdInfo's peerEid through `RankEidMap`, estimate
  communicator table has only local rank; lcEid for WRITE_WITH_NOTIFY port charging).
`RaTypicalQpModify` maintains `remoteDeviceId`'s loopback plane semantics
(`DpuPendingNotify` pairing dependency, under estimate = local), task metadata uniformly takes binding fields.

**NotifyWait precise synthesis (hcomm primitive interception)**: Correct semantics are "only pending notifications inserted by peer post
generate NotifyWait for me"; under phantom, the peer does not run, and this notification has no real source. Furthermore, hcomm
`NotifyWait` performs imm hard verification on poll results (`wc.imm_data == dpuNotifyId`, immediately reports error on mismatch).
Implementation (`hcomm_api_stub.cc` intercepts 5 hcomm northbound primitives, real function pointer static
one-time dlsym resolution, **normal mode first line short-circuit pure forwarding, zero overhead**):
- post-type primitives (`HcommChannelNotifyRecordOnThread` / `HcommWriteWithNotify(±Nbi)
  OnThread`, input carries `remoteNotifyIdx`) → `InsertDpuPendingNotifyEstimate`
  records `channelSlots[channel][idx] = (imm, notifyId)` slot mapping — under echo,
  `remoteDpuNotifyIds_ == localDpuNotifyIds_`, post-side idx→imm mapping is the wait-side
  expected mapping (same idx, same slot value);
- wait-type primitives (`HcommChannelNotifyWaitOnThread(±WithDefaultTimeout)`, input carries
  `localNotifyIdx`) → `ConsumeDpuPendingNotifyAndEmit` precise path: expected
  imm = `channelSlots[channel][idx]` **precisely known**, directly synthesizes "virtual peer post" to construct
  wc return, imm necessarily passes hcomm hard verification; NotifyWait node records the local real wait event,
  only the "notification arrival" external fact is virtualized (isomorphic with socket echo virtualizing "peer response");
- Design constraint: estimate mode wait consumption **only goes through the precise path** — all wait entries must be intercepted
  (single-card phantom FIFO loopback consumption is unreliable, imm mismatch/timeout under asymmetric send/receive, old mechanism removed);
  unintercepted entry → `pollCq::miss` **ERROR** (deduplicated once per qpn+idx combination, preventing busy-loop
  log flooding) → hcomm timeout exposes,据此 add interception in `hcomm_api_stub.cc`.

**taskSrc Source Marker (BLOB v5)**: `HcclTaskMetaData` adds `taskSrc`
(0=default, 1=DPU); tasks emitted by DPU stub set to 1, for Python-side precise splitting by source
(not relying on transport==ROCE and other link semantic criteria; future sources such as AICPU RoCE are unaffected).

**DPU Sequence Merge Pass (Section 5.1.2, `sqe_group_merge.py::merge_dpu_write_sequences`)**:
hcomm `WriteWithNotify` splits large messages with len > maxMsgSize into N-1 ordinary RDMA_WRITE +
1 trailing WRITE_WITH_IMM (whole operation still reports only one); collection side stores per WR as N-1 MEM_CPY +
1 WRITE_WITH_NOTIFY. The pass scans DPU stream order for src/dst dual-address-consecutive MEM_CPY chains,
terminated and merged into a single entry by address-consecutive WRITE_WITH_NOTIFY (len accumulated across full chain, description follows trailing task including
notifyId, occupies trailing slot); chains without termination are not merged (independent pure writes remain per-entry `RDMASend`);
only `src_type==1` data tasks participate, other tasks preserved as-is without cutting the chain.

> **Verification** (222 networking 8MB all_reduce): First round (20260921) 15 `Unknown` → 0
> (`Dpu_Notify_Record` ×5 / `Dpu_Notify_Wait` ×9 / `Dpu_Write_With_Notify` ×4,
> link=ROCE), peer rank=2/4/6 (real cross-pod peers), `eid_unresolved=0`,
> WRITE_WITH_NOTIFY enters `d2h` port charging. Final version (20260922, after primitive interception):
> estimate (-np 1) makespan 166.596us, primitive interception 16 times, precise synthesis 11 times, miss=0;
> normal (-np 8) checker All Success (primitive pure forwarding, interception log 0 entries); single machine 8p
> (118) SQE path regression unchanged.

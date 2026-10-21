# HVRM Insight V3 User Guide

## 1. Overview

HVRM Insight V3 currently supports `DAGView` task graph viewing. This document covers data preparation, page access, dataset selection, DAG graph browsing, and node searching.

Other pages and advanced linkage capabilities are not yet implemented.

---

## 2. Prerequisites

Before using Insight V3, confirm that Checker has completed execution and has generated Insight data.

To configure Checker data output, confirm that Insight dump is enabled in the Checker configuration file:

```json
# The configuration file is located at /pathto/hccl_vm_install/plugin/checker/manifest.json

{
  ...
  "setting": {              // Checker plugin configuration
      ...
      "enable_insight_dump": true,         // Whether to output Checker V3 Insight data (disabled by default)
      "enable_memory_snapshot_dump": false  // Whether to output memory snapshot data (not yet supported, disabled by default, requires Insight data output enabled first)
  }
}
```

DAGView requires at least the following data files:

```text
<dataset_name>/
├── manifest.json
└── graph/
    ├── graph.msgpack
    └── layout.msgpack
```

The `manifest.json` file provides dataset metadata. The `graph.msgpack` and `layout.msgpack` files are used to render the DAG task graph.

---

## 3. Compile and Install the Insight Plugin

The Insight V3 frontend source code is located at:

```text
{hccl_vm directory}/src/plugin/insight/frontend_v3
```

Before installing the Insight plugin, compile the frontend first. Then run the HCCL VM build and installation workflow.

Note: Compiling the Insight visualization frontend requires `Node.js` and `npm`. Use `Node.js 20.19.0` or above and `npm 10.x` or above.

The recommended steps are as follows:

1. Enter the frontend directory:

```bash
cd {hccl_vm directory}/src/plugin/insight/frontend_v3
```

2. Install frontend dependencies:

```bash
npm install
```

3. Build the frontend artifacts:

```bash
npm run build
```

After the frontend compilation completes, the `src/plugin/insight/dist/` directory is generated.

4. Return to the HCCL_VM root directory and run `build.sh` to compile and install hccl-vm:

```bash
cd {HCCL_VM directory}
bash build.sh --package-path <ASCEND_CANN_PATH> --hcomm-path <HCOMM_CODE_PATH>
```

To package the installation directory, append `--pkg`. If the current build scenario requires AICPU / AIV / FULL mode, append `--aicpu`, `--aiv`, or `--full` as in the standard workflow.

---

## 4. Insight Plugin Configuration and Start/Stop

The Insight plugin configuration file is located at:

```text
hccl_vm_install/plugin/visualization/insight/manifest.json
```

The current default configuration is as follows:

```json
{
    "name": "insight",
    "version": "1.0.0",
    "entry": "python3 server.py",
    "dependency": {
        "min_core_version": "1.0.0"
    },
    "setting": {
        "dist_path": "./dist",
        "data_path": "../../checker/data/insight",
        "topo_config_path": "../../../../../asset/cluster_model/config/cluster",
        "port": 8080
    }
}
```

Common field descriptions are as follows:

| Field | Description |
|-------|-------------|
| `entry` | Insight plugin startup command. Currently starts the service through `python3 server.py` |
| `setting.dist_path` | Frontend static resource directory. Reads `./dist` by default |
| `setting.data_path` | Insight data directory. Points to the Checker output `data/insight` by default |
| `setting.topo_config_path` | Cluster topology configuration directory |
| `setting.port` | Insight service port. Default is `8080` |

To modify the port or data directory, edit the `setting` field in `manifest.json` directly.

> Warning: Insight binds to `127.0.0.1` by default through `python3 server.py`. To listen on a specified IP instead of the default `localhost(127.0.0.1)`, enter the `hccl_vm_install/plugin/visualization/insight` directory and execute manually:
>
> ```bash
> python3 server.py --host <ip> --port <port>
> ```
>
> For example:
>
> ```bash
> cd hccl_vm_install/plugin/visualization/insight
> python3 server.py --host 0.0.0.0 --port 8080
> ```

The plugin installation and uninstallation commands are as follows:

```bash
# Install and start the Insight plugin
hccl-vm plugin install @insight

# Uninstall the Insight plugin
hccl-vm plugin uninstall @insight
```

---

## 5. Open Insight

If the Insight service is already running, open the service address in a browser directly, for example:

```text
http://localhost:8080
```

To start Insight through the plugin, execute:

```bash
hccl-vm plugin install @insight
```

After the plugin starts, the terminal outputs the access address. Open that address in a browser to enter the Insight page.

![Launch Insight and Open Page](insight_image/V3_image/launch.gif)

---

## 6. Select a Dataset

After opening Insight, the `Overview` page loads by default. Follow these steps to select the dataset for analysis:

1. View the dataset list in the dataset table in the center.
2. Click the target dataset.
3. Confirm the Ranks to view in the Rank tree on the left.
4. If the DAG graph is large, reduce the Rank selection range first.
5. Click `Enter Correlation View` on the right.

![Select Dataset and Enter Correlation View](insight_image/V3_image/dagView_V3.gif)

When analyzing for the first time, select a small number of Ranks first. Confirm the analysis direction before expanding the Rank range gradually.

---

## 7. Enter DAGView

After entering the `Correlation` page, focus on the `DAGView Task View` in the lower half.

The main page areas are as follows:

| Area | Function |
|------|----------|
| Left panel | Select Rank, search nodes |
| Center bottom | Display the DAG task graph |
| Right detail panel | View current node details |

If the upper memory view is empty, it does not affect DAGView viewing and usage below.

---

## 8. View the DAG Task Graph

The DAG task graph consists of swim lanes, nodes, and arrows:

| Element | Description |
|---------|-------------|
| Swim lane | Represents an execution queue. Typically displayed as `Rank / Stream / Queue` |
| Node | Represents a task, for example data transfer, Reduce, Record, Wait, and so on |
| Arrow | Represents dependency relationships between tasks |
| Loop dashed box | Represents a Loop region. A dashed box surrounds related nodes within that Loop, with a `Loop` label outside the box |

Common node types are as follows:

| Node Type | Meaning |
|-----------|---------|
| `TRANS_MEM` | Data transfer task |
| `REDUCE` | Reduce task |
| `RECORD` | Notify record task |
| `WAIT` | Notify wait task |
| `CCU_GRAPH` | CCU graph task |
| `AIV_GRAPH` | AIV graph task |

Recommended viewing approach:

1. Follow the arrow direction to view the task dependency order.
2. Click the task node of interest.
3. View the Rank, Stream, Queue, and detailed node information in the right detail panel.
4. Continue viewing parent and child nodes to trace upstream and downstream dependencies.

For Loop display, pay additional attention to the following information:

1. If a Loop exists in the DAG, the page automatically overlays a Loop dashed box on the task graph.
2. The dashed box covers the Loop start, end, and loop body internal nodes, making it easy to identify loop boundaries.
3. Click the Loop capsule to navigate to the corresponding Loop Start node directly.
4. After selecting a node inside a Loop, the right detail panel displays the Loop information and nesting chain for that node.

![Browse Task Graph and View Node Details in DAGView](insight_image/V3_image/dagView_V3_2.gif)

---

## 9. Canvas Operations

The DAGView canvas supports the following operations:

| Operation | Description |
|-----------|-------------|
| Move canvas | Drag a blank area |
| Zoom in / Zoom out | Use the mouse scroll wheel, or click `+` / `-` in the lower right corner |
| Reset zoom | Click the percentage button in the lower right corner |
| Select node | Click the target node |

When the graph is large, reduce the Rank selection range first, then zoom in on local areas to view dependencies.

---

## 10. View Node Details

After clicking a DAG node, the right detail panel displays node information. Common information is as follows:

| Section | Description |
|---------|-------------|
| Node overview | Node ID, task type, Rank, Stream, Queue |
| Loop information | The Loop the current node belongs to, Loop count, instruction range, Loop boundaries, and so on |
| Parent nodes | Upstream nodes that the current node depends on |
| Child nodes | Downstream nodes that depend on the current node |
| Node semantics | Notify, Task metadata, Memory Slices, and so on |
| Raw JSON | Complete raw information for the current node |

When investigating dependency relationships, view `Parent nodes` and `Child nodes` first. When investigating loop structures, focus on `Loop information`. When investigating data transfer issues, focus on `Memory Slices`.

The `Memory Slices` section has the following display rules:

1. A regular memory Slice displays `rank / type / offset / size`.
2. If the Slice belongs to `MS_CCU`, Insight V3 displays `MSID` first instead of the underlying abstract offset.
3. For batch-type tasks, if multiple `MS_CCU` Slices exist, they are automatically merged into one summary card.
4. The summary card displays a description such as `Total 8 MSIDs used`.
5. After expanding `MSID Details`, view the `id` and `size` for each `MSID` individually.

---

## 11. Search Nodes

The left `Search` panel enables quick DAG node location.

Supported search fields are as follows:

| Field | Use Case |
|-------|----------|
| `taskId` | Use when the node ID is known |
| `taskType` | Search by task type, for example `TRANS_MEM` |
| `notifyId` | Search by notify id |

Operation steps:

1. Select a search field in the left search panel.
2. Enter the complete keyword.
3. Click the search result.
4. DAGView automatically locates and selects the corresponding node.

The current search uses exact matching. If no result is found, confirm that the keyword is complete and that the target Rank is selected.

---

## 12. Recommended Analysis Workflow

Follow this workflow for DAGView analysis:

1. Select a dataset on the `Overview` page.
2. Select a small number of Ranks in the left Rank tree.
3. Click `Enter Correlation View`.
4. View overall nodes and dependency arrows in DAGView.
5. Click the node of interest and view the right detail panel.
6. Trace upstream and downstream dependencies through parent and child nodes.
7. Use the search function to locate known nodes quickly.

---

## 13. Frequently Asked Questions

### 13.1 No Data on the Page

Confirm that the Insight service address is correct and that Checker has generated Insight data.

### 13.2 DAG Is Empty After Entering the Correlation Page

Confirm that the following files exist in the dataset:

```text
graph/graph.msgpack
graph/layout.msgpack
```

### 13.3 DAG Graph Is Too Dense

Reduce the number of selected Ranks on the left first. View only a subset of Ranks, then zoom in on local areas for analysis.

### 13.4 Node Not Found in Search

Confirm that the search keyword is complete and that the Rank where the target node resides is selected.

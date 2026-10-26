# HCCL-VM MariaDB 数据库部署

[返回主指南](../../../README.md)

部署架构（一台模拟器对应一套数据库实例）：

```mermaid
flowchart LR
    SIM["HCCL-VM 模拟器<br/>"]
    subgraph MARIADB ["MariaDB Server"]
        DATA[("hcclvm_* 数据库<br/>")]
    end
    SIM -->|" hccl_vm 账号连接"| MARIADB
```

以下命令中的源码路径基于 `HCCL_VM_SOURCE_HOME`（根据实际CheckerL2 源码根目录配置）：

```bash
export HCCL_VM_SOURCE_HOME=/home/workspace/CheckerL2
```

部署与运行流程：

- **首次部署**：第 1 章（部署数据库、创建账号与连接配置文件），一次性完成
- **日常运行**：首次部署完成后，日常运行直接从第 2 章开始

## 1. 首次部署参考
> **📌 提示：** **一台模拟器对应一套数据库实例**

### 1.1 部署mariadb环境

> **📌 提示：** 本文档配置已在 Ubuntu 24.04 / MariaDB 10.11 验证，其他版本未验证；WSL 需启用 systemd 才能使用 systemctl，未启用时以 `sudo service mariadb start` 代替

```bash
sudo apt install -y mariadb-server mariadb-client
sudo systemctl start mariadb
sudo mariadb-admin ping
```

### 1.2 创建数据库账号

以下 SQL 创建 `hccl_vm` 账号，使用示例密码 `password`，替换为实际密码后执行；账号已存在时会先删除重建，密码与权限重置为本次设置的值。
```bash
sudo mariadb <<'SQL'
DROP USER IF EXISTS 'hccl_vm'@'127.0.0.1';
DROP USER IF EXISTS 'hccl_vm'@'localhost';
CREATE USER 'hccl_vm'@'127.0.0.1' IDENTIFIED BY 'password';
CREATE USER 'hccl_vm'@'localhost' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON *.* TO 'hccl_vm'@'127.0.0.1' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON *.* TO 'hccl_vm'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
SQL
```

执行后用该账号登录验证（密码替换为实际密码），输出 1 即通过：

```bash
mariadb -h 127.0.0.1 -P 3306 -u hccl_vm -ppassword -e "SELECT 1;"
```

在数据库服务器复制执行以下命令（将仓库内示例配置安装为服务端配置）：

```bash
sudo cp ${HCCL_VM_SOURCE_HOME}/src/storage_backends/mariadb/config/z-hccl-vm.cnf.example /etc/mysql/mariadb.conf.d/z-hccl-vm.cnf
sudo chmod 644 /etc/mysql/mariadb.conf.d/z-hccl-vm.cnf
```

***不同规格数据库服务器参数配置调整***  
编辑 `/etc/mysql/mariadb.conf.d/z-hccl-vm.cnf`（需 sudo 权限），将 `thread_pool_size`（示例值 224）按数据库可用的逻辑 CPU 数填写，修改后重启。如16/32/64 个可用逻辑 CPU 的实例分别填 16/32/64。设置了 CPU 亲和性时以限制后的额度为准。
[MariaDB 线程池说明](https://mariadb.com/docs/server/ha-and-performance/optimization-and-tuning/buffers-caches-and-threads/thread-pool/thread-pool-in-mariadb)。

```bash
sudo systemctl restart mariadb
sudo mariadb-admin ping
```

### 1.3 重启后核对生效值
> **📌 提示：** 账号密码需要替换为1.2章节实际创建的账号密码  
> **📌 提示：** 核对 SQL 中 `thread_pool_size` 的期望值 224 为示例值，替换为 1.2 节写入 cnf 的本机 CPU 数（如 64 核实例为 64）；该项 status=1 表示实际生效值与 cnf 设置一致，status=-1 表示两者不一致

```bash
mariadb -u hccl_vm -ppassword --table <<'SQL'
SELECT * FROM (
    SELECT 'transaction_isolation' AS parameter, @@global.tx_isolation AS actual_value, 'READ-COMMITTED' AS expected_value, IF(@@global.tx_isolation = 'READ-COMMITTED', 1, -1) AS status
    UNION ALL SELECT 'thread_handling', @@global.thread_handling, 'pool-of-threads', IF(@@global.thread_handling = 'pool-of-threads', 1, -1)
    UNION ALL SELECT 'thread_pool_size', CAST(@@global.thread_pool_size AS CHAR), '224', IF(@@global.thread_pool_size = 224, 1, -1)
    UNION ALL SELECT 'innodb_buffer_pool_size', CONCAT(ROUND(@@global.innodb_buffer_pool_size / 1073741824, 0), 'G'), '8G', IF(ROUND(@@global.innodb_buffer_pool_size / 1073741824, 0) = 8, 1, -1)
    UNION ALL SELECT 'max_connections', CAST(@@global.max_connections AS CHAR), '8192', IF(@@global.max_connections = 8192, 1, -1)
    UNION ALL SELECT 'back_log', CAST(@@global.back_log AS CHAR), '4096', IF(@@global.back_log = 4096, 1, -1)
    UNION ALL SELECT 'max_prepared_stmt_count', CAST(@@global.max_prepared_stmt_count AS CHAR), '1048576', IF(@@global.max_prepared_stmt_count = 1048576, 1, -1)
    UNION ALL SELECT 'connect_timeout', CONCAT(@@global.connect_timeout, 's'), '30s', IF(@@global.connect_timeout = 30, 1, -1)
    UNION ALL SELECT 'net_read_timeout', CONCAT(@@global.net_read_timeout, 's'), '600s', IF(@@global.net_read_timeout = 600, 1, -1)
    UNION ALL SELECT 'net_write_timeout', CONCAT(@@global.net_write_timeout, 's'), '600s', IF(@@global.net_write_timeout = 600, 1, -1)
    UNION ALL SELECT 'wait_timeout', CONCAT(@@global.wait_timeout, 's'), '28800s', IF(@@global.wait_timeout = 28800, 1, -1)
    UNION ALL SELECT 'innodb_flush_log_at_trx_commit', CAST(@@global.innodb_flush_log_at_trx_commit AS CHAR), '0', IF(@@global.innodb_flush_log_at_trx_commit = 0, 1, -1)
    UNION ALL SELECT 'innodb_flush_method', @@global.innodb_flush_method, 'O_DIRECT_NO_FSYNC', IF(@@global.innodb_flush_method = 'O_DIRECT_NO_FSYNC', 1, -1)
    UNION ALL SELECT 'innodb_doublewrite', CAST(@@global.innodb_doublewrite AS CHAR), 'OFF', IF(@@global.innodb_doublewrite = 0, 1, -1)
    UNION ALL SELECT 'innodb_autoinc_lock_mode', CAST(@@global.innodb_autoinc_lock_mode AS CHAR), '2', IF(@@global.innodb_autoinc_lock_mode = 2, 1, -1)
    UNION ALL SELECT 'innodb_flush_neighbors', CAST(@@global.innodb_flush_neighbors AS CHAR), '0', IF(@@global.innodb_flush_neighbors = 0, 1, -1)
    UNION ALL SELECT 'innodb_purge_threads', CAST(@@global.innodb_purge_threads AS CHAR), '8', IF(@@global.innodb_purge_threads = 8, 1, -1)
    UNION ALL SELECT 'performance_schema', CAST(@@global.performance_schema AS CHAR), 'OFF', IF(@@global.performance_schema = 0, 1, -1)
    UNION ALL SELECT 'query_cache_type', @@global.query_cache_type, 'OFF', IF(@@global.query_cache_type = 'OFF', 1, -1)
    UNION ALL SELECT 'query_cache_size', CAST(@@global.query_cache_size AS CHAR), '0', IF(@@global.query_cache_size = 0, 1, -1)
    UNION ALL SELECT 'general_log', CAST(@@global.general_log AS CHAR), 'OFF', IF(@@global.general_log = 0, 1, -1)
    UNION ALL SELECT 'slow_query_log', CAST(@@global.slow_query_log AS CHAR), 'OFF', IF(@@global.slow_query_log = 0, 1, -1)
    UNION ALL SELECT 'skip_name_resolve', CAST(@@global.skip_name_resolve AS CHAR), 'OFF', IF(@@global.skip_name_resolve = 0, 1, -1)
) t;
SQL
```

### 1.4 模拟器配置

首次部署创建连接配置文件，日常运行仅需设置环境变量：

```bash
# 首次部署：创建连接配置文件。存放目录可自选（示例用 /home/workspace）；文件含数据库密码，注意权限
cp ${HCCL_VM_SOURCE_HOME}/src/storage_backends/mariadb/config/storage_mariadb.yaml.example /home/workspace/storage_mariadb.yaml
chmod 600 /home/workspace/storage_mariadb.yaml

# 日常运行：每个 shell 会话设置（必须在启动模拟器前设置）
export HCCL_VM_STORAGE_CONFIG=/home/workspace/storage_mariadb.yaml
```

> **📌 提示：** **必须在执行 `hccl-vm start` 启动模拟器之前设置该变量**。

将 /home/workspace/storage_mariadb.yaml 内 URI 的地址替换为数据库实际地址，账号、密码替换为[1.2 节](#12-创建数据库账号)创建的账号密码：

```yaml
uri: mariadb://hccl_vm:password@127.0.0.1:3306
```

日志调整：
```bash
cp ${HCCL_VM_SOURCE_HOME}/config/log_config.yaml /home/workspace/log_config.yaml 
```

模拟器 `hccl-vm start` 后，可用该账号确认存储已使用 MariaDB（密码替换为实际密码），输出应包含模拟器自动创建的 `hcclvm_` 开头数据库：

```bash
mariadb -h 127.0.0.1 -P 3306 -u hccl_vm -ppassword -e "SHOW DATABASES"
```

## 2. 环境配置

日常运行前，可先验证数据库服务是否正常存活；若连接异常或服务未就绪，请参考[第 1 章](#1-首次部署参考)重新排查或部署：

```bash
mariadb-admin ping
```

> **📌 重要：** `HCCL_VM_STORAGE_CONFIG` **必须在执行 `hccl-vm start` 之前设置该变量**。

```bash
# 启动模拟器前设置（指向 1.4 节创建的配置文件）
export HCCL_VM_STORAGE_CONFIG=/home/workspace/storage_mariadb.yaml
```

### 2.1 MPICH 配置

```bash
# 使用 MPICH 时，在启动模拟器前设置
export MPICH_NO_LOCAL=1
```

### 2.2 PMI socket 缓冲区配置

在运行 MPI 的主机上完成环境配置。

```bash
sudo tee /etc/sysctl.d/99-hccl-vm-socket.conf >/dev/null <<'CONF'
net.core.wmem_max=16777216
net.core.rmem_max=16777216
net.core.wmem_default=4194304
net.core.rmem_default=4194304
net.unix.max_dgram_qlen=10240
CONF

sudo sysctl -p /etc/sysctl.d/99-hccl-vm-socket.conf
```

## 3. hccl_test 1024p checkonly模式参考配置
> **📌 提示：** 1024p场景需要服务器可用资源大于500G，***小于该规格时，可扩大swap使可用内存满足该规格要求，但swap受磁盘等硬件因素影响，执行可能变慢或者超时***  
> **📌 提示：** 1024p场景需要服务器可用核大于100，***小于该规格可能由于CPU调度导致执行变慢或者超时***

配置示例：
```bash
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_DFS_CONFIG=cluster_heartbeat:off
export HCCL_VM_COMMUNICATOR_WAIT_TIMEOUT_MS=7200000
export HCCL_VM_STORAGE_CONFIG=/home/workspace/storage_mariadb.yaml
export HCCL_VM_LOG_CONFIG_PATH=/home/workspace/log_config.yaml 
export ASCEND_GLOBAL_LOG_LEVEL=3
```

HCCL_VM_LOG_CONFIG_PATH日志级别调优示例：
``` bash
# 低日志级别配置：仅保留 warn+error（info/debug/trace 全部关闭）
# 级别: 0-trace 1-debug 2-info 3-warn 4-error 5-critical 6-off
hccl_vm:
    console_level: 4
    file_level: 3
    max_file_size: 52428800
    max_files: 65535
    file_path: logs
    file_base_name: app_log
    file_suffix: .log
    enable_compress: false

proxy:
    console_level: 4
    file_level: 3
    max_file_size: 52428800
    max_files: 65535
    file_path: logs/proxy
    file_base_name: app_log
    file_suffix: .log
    enable_compress: false

runner:
    console_level: 6
    file_level: 3
    max_file_size: 52428800
    max_files: 65535
    file_path: logs/runner
    file_base_name: app_log
    file_suffix: .log
    enable_compress: false

checker:
    console_level: 2
    file_level: 3
    max_file_size: 52428800
    max_files: 65535
    file_path: logs/checker
    file_base_name: app_log
    file_suffix: .log
    enable_compress: false
    
# 拓扑生成脚本(generate_cluster_topo.sh/generate_server_topo.sh/allocate_eid.sh)日志配置
# console_level: 脚本控制台日志级别, 与C++日志级别定义一致:
#   0-trace 1-debug 2-info 3-warn 4-error 5-critical 6-off
#   debug(1): 输出全部日志(含逐server/逐die/逐端口明细)
#   info(2):  输出步骤与汇总信息(默认)
#   warn(3):  仅输出警告与错误
#   error(4): 仅输出错误
#   off(6):   不输出任何日志
script:
    console_level: 3    
```

## 第三方依赖

本模块专属第三方依赖：

| 开源软件 | 版本 | 许可证 (SPDX) | 下载地址 |
| --- | --- | --- | --- |
| mariadb-connector-c | 3.3.8 | LGPL-2.1-or-later | [v3.3.8-src.tar.gz](https://gitee.com/src-openeuler/mariadb-connector-c/raw/openEuler-24.03-LTS-SP2/mariadb-connector-c-3.3.8-src.tar.gz) |

许可原文位于 [licenses/](licenses/)。

[返回主指南](../../../README.md) · [继续主指南使用示例](../../../README.md#32-使用示例)

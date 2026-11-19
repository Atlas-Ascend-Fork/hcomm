# HCCL-VM MariaDB Database Deployment

[Back to the main guide](../../../README.md)

Deployment architecture (one simulator corresponds to one database instance):

```mermaid
flowchart LR
    SIM["HCCL-VM simulator<br/>"]
    subgraph MARIADB ["MariaDB Server"]
        DATA[("hcclvm_* databases<br/>")]
    end
    SIM -->|" hccl_vm account connection"| MARIADB
```

The source code path in the following commands is based on `HCCL_VM_SOURCE_HOME` (configure it based on the actual CheckerL2 source code root directory):

```bash
export HCCL_VM_SOURCE_HOME=/home/workspace/CheckerL2
```

Deployment and running procedure:

- **First deployment**: Chapter 1 (deploying the database and creating the account and connection configuration file). This is a one-time task.
- **Daily running**: After the first deployment is complete, start directly from Chapter 2.

## 1. First Deployment Reference
> **📌 NOTE:** **One simulator corresponds to one database instance**

### 1.1 Deploying the MariaDB Environment

> **📌 NOTE:** The configuration in this document has been verified on Ubuntu 24.04 / MariaDB 10.11, and other versions are not verified. For WSL, systemd must be enabled before systemctl can be used. If systemd is not enabled, use `sudo service mariadb start` instead.

```bash
sudo apt install -y mariadb-server mariadb-client
sudo systemctl start mariadb
sudo mariadb-admin ping
```

### 1.2 Creating a Database Account

The following SQL creates the `hccl_vm` account with the example password `password`. Replace the password with the actual password before execution. If the account already exists, it is deleted and re-created, and the password and permissions are reset to the values set this time.

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

After the execution, log in with the account to verify the result (replace the password with the actual password). The verification passes if 1 is output:

```bash
mariadb -h 127.0.0.1 -P 3306 -u hccl_vm -ppassword -e "SELECT 1;"
```

Run the following commands on the database server (this installs the example configuration in the repository as the server configuration):

```bash
sudo cp ${HCCL_VM_SOURCE_HOME}/src/storage_backends/mariadb/config/z-hccl-vm.cnf.example /etc/mysql/mariadb.conf.d/z-hccl-vm.cnf
sudo chmod 644 /etc/mysql/mariadb.conf.d/z-hccl-vm.cnf
```

***Parameter Configuration Adjustment for Database Servers of Different Specifications***
Edit `/etc/mysql/mariadb.conf.d/z-hccl-vm.cnf` (sudo permission required), set `thread_pool_size` (example value 224) to the number of logical CPUs available to the database, and restart the service after the modification. For example, instances with 16/32/64 available logical CPUs are set to 16/32/64 respectively. If CPU affinity is set, the quota after the restriction applies.
For details about the MariaDB thread pool, see [MariaDB thread pool description](https://mariadb.com/docs/server/ha-and-performance/optimization-and-tuning/buffers-caches-and-threads/thread-pool/thread-pool-in-mariadb).

`innodb_buffer_pool_size` defaults to 32G. When the simulator runs on the same machine, reserve memory for the simulator and the system. In environments with insufficient memory, reduce this value and adjust the expected value in the verification SQL below accordingly.

```bash
sudo systemctl restart mariadb
sudo mariadb-admin ping
```

### 1.3 Verifying Effective Values After the Restart
> **📌 NOTE:** Replace the account and password with those actually created in section 1.2.
> **📌 NOTE:** The expected value 224 of `thread_pool_size` in the verification SQL is an example value. Replace it with the number of local CPUs written into the cnf file in section 1.2 (for example, 64 for a 64-core instance). For this item, status=1 indicates that the actual effective value is the same as the value set in the cnf file, and status=-1 indicates that the two values are different.

```bash
mariadb -u hccl_vm -ppassword --table <<'SQL'
SELECT * FROM (
    SELECT 'transaction_isolation' AS parameter, @@global.tx_isolation AS actual_value, 'READ-COMMITTED' AS expected_value, IF(@@global.tx_isolation = 'READ-COMMITTED', 1, -1) AS status
    UNION ALL SELECT 'thread_handling', @@global.thread_handling, 'pool-of-threads', IF(@@global.thread_handling = 'pool-of-threads', 1, -1)
    UNION ALL SELECT 'thread_pool_size', CAST(@@global.thread_pool_size AS CHAR), '224', IF(@@global.thread_pool_size = 224, 1, -1)
    UNION ALL SELECT 'innodb_buffer_pool_size', CONCAT(ROUND(@@global.innodb_buffer_pool_size / 1073741824, 0), 'G'), '32G', IF(ROUND(@@global.innodb_buffer_pool_size / 1073741824, 0) = 32, 1, -1)
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

### 1.4 Simulator Configuration

During the first deployment, create the connection configuration file. For daily running, you only need to set the environment variable:

```bash
# First deployment: create the connection configuration file. You can select the storage directory (this example uses /home/workspace). The file contains the database password, so pay attention to permissions.
cp ${HCCL_VM_SOURCE_HOME}/src/storage_backends/mariadb/config/storage_mariadb.yaml.example /home/workspace/storage_mariadb.yaml
chmod 600 /home/workspace/storage_mariadb.yaml

# Daily running: set this in each shell session (this must be done before starting the simulator)
export HCCL_VM_STORAGE_CONFIG=/home/workspace/storage_mariadb.yaml
```

> **📌 NOTE:** **This variable must be set before you run `hccl-vm start` to start the simulator.**

Replace the address in the URI in /home/workspace/storage_mariadb.yaml with the actual database address, and replace the account and password with those created in [section 1.2](#12-creating-a-database-account):

```yaml
uri: mariadb://hccl_vm:password@127.0.0.1:3306
```

Log configuration adjustment:

```bash
cp ${HCCL_VM_SOURCE_HOME}/config/log_config.yaml /home/workspace/log_config.yaml 
```

After `hccl-vm start` is run for the simulator, use the account to confirm that MariaDB is used as the storage (replace the password with the actual password). The output should contain the databases automatically created by the simulator, whose names start with `hcclvm_`:

```bash
mariadb -h 127.0.0.1 -P 3306 -u hccl_vm -ppassword -e "SHOW DATABASES"
```

## 2. Environment Configuration

Before daily running, verify that the database service is alive. If the connection is abnormal or the service is not ready, refer to [Chapter 1](#1-first-deployment-reference) to troubleshoot or deploy again:

```bash
mariadb-admin ping
```

> **📌 IMPORTANT:** `HCCL_VM_STORAGE_CONFIG` **must be set before you run `hccl-vm start`.**

```bash
# Set this before starting the simulator (pointing to the configuration file created in section 1.4)
export HCCL_VM_STORAGE_CONFIG=/home/workspace/storage_mariadb.yaml
```

### 2.1 MPICH Configuration

```bash
# Set this when MPICH is used, before starting the simulator
export MPICH_NO_LOCAL=1
```

### 2.2 PMI Socket Buffer Configuration

Complete the environment configuration on the host machine that runs MPI.

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

## 3. Reference Configuration for the hccl_test 1024p check-only Mode
> **📌 NOTE:** The 1024p scenario requires more than 500 GB of available resources on the server. ***If the resources are below this specification, you can expand the swap space to meet the memory requirement. However, the swap space is affected by hardware factors such as disks, and the execution may slow down or time out.***
> **📌 NOTE:** The 1024p scenario requires more than 100 available cores on the server. ***If the number of cores is below this specification, the execution may slow down or time out due to CPU scheduling.***

Configuration example:

```bash
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_DFS_CONFIG=cluster_heartbeat:off
export HCCL_VM_COMMUNICATOR_WAIT_TIMEOUT_MS=7200000
export HCCL_VM_STORAGE_CONFIG=/home/workspace/storage_mariadb.yaml
export HCCL_VM_LOG_CONFIG_PATH=/home/workspace/log_config.yaml 
export ASCEND_GLOBAL_LOG_LEVEL=3
```

HCCL_VM_LOG_CONFIG_PATH log level tuning example:

``` bash
# Low log level configuration: only warn+error are retained (info/debug/trace are all disabled)
# Levels: 0-trace 1-debug 2-info 3-warn 4-error 5-critical 6-off
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
    
# Log configuration of the topology generation scripts (generate_cluster_topo.sh/generate_server_topo.sh/allocate_eid.sh)
# console_level: console log level of the scripts, consistent with the C++ log level definitions:
#   0-trace 1-debug 2-info 3-warn 4-error 5-critical 6-off
#   debug(1): outputs all logs (including details of each server/die/port)
#   info(2): outputs step and summary information (default)
#   warn(3): outputs only warnings and errors
#   error(4): outputs only errors
#   off(6): outputs no logs
script:
    console_level: 3    
```

## Third-Party Dependencies

Third-party dependencies specific to this module:

| Open Source Software | Version | License (SPDX) | Download URL |
| --- | --- | --- | --- |
| mariadb-connector-c | 3.3.8 | LGPL-2.1-or-later | [mariadb-connector-c-3.3.8-src.tar.gz](https://raw.gitcode.com/src-openeuler/mariadb-connector-c/blobs/9389ee06248685f68fe17b9d18a90c990e039020/mariadb-connector-c-3.3.8-src.tar.gz) |

The source code cache is stored in third_party/ at the repository root (archives and unpacked artifacts, which are ignored by git and verified by the build using SHA-256 and content fingerprints). The first build downloads the source code online. In offline scenarios, copy the original archive named mariadb-connector-c-3.3.8-src.tar.gz to third_party/ at the repository root. After the build verification passes, the archive is directly unpacked for use, without requiring network access.

The original license text is stored in [licenses/](licenses/).

[Back to the main guide](../../../README.md) · [Continue to usage examples in the main guide](../../../README.md#32-使用示例)

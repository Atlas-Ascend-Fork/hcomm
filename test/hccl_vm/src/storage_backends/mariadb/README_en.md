# HCCL-VM MariaDB Database Deployment

[Return to Main Guide](../../../README.md)

Deployment architecture (one simulator corresponds to one database instance):

```mermaid
flowchart LR
    SIM["HCCL-VM Simulator<br/>"]
    subgraph MARIADB ["MariaDB Server"]
        DATA[("hcclvm_* Databases<br/>")]
    end
    SIM -->|"hccl_vm Account Connection"| MARIADB
```

The source code paths in the following commands are based on `HCCL_VM_SOURCE_HOME` (configure according to the actual CheckerL2 source root directory):

```bash
export HCCL_VM_SOURCE_HOME=/home/workspace/CheckerL2
```

Deployment and running workflow:

- **Initial deployment**: Chapter 1 (deploy the database, create the account, and create the connection configuration file), complete once
- **Daily operation**: After initial deployment is complete, daily operation starts directly from Chapter 2

## 1. Initial Deployment Reference
> **📌 Note:** **One simulator corresponds to one database instance**

### 1.1 Deploy the MariaDB Environment

> **📌 Note:** This document configuration has been verified on Ubuntu 24.04 / MariaDB 10.11, other versions are not verified; WSL requires systemd enabled to use systemctl, when not enabled use `sudo service mariadb start` instead

```bash
sudo apt install -y mariadb-server mariadb-client
sudo systemctl start mariadb
sudo mariadb-admin ping
```

### 1.2 Create the Database Account

The following SQL creates the `hccl_vm` account, using the sample password `password`, replace with the actual password before execution; when the account already exists, it will be deleted and recreated first, and the password and permissions will be reset to the values set this time.
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

After execution, log in with this account to verify (replace the password with the actual password), output of 1 indicates success:

```bash
mariadb -h 127.0.0.1 -P 3306 -u hccl_vm -ppassword -e "SELECT 1;"
```

Copy and execute the following command on the database server (install the sample configuration from the repository as the server configuration):

```bash
sudo cp ${HCCL_VM_SOURCE_HOME}/src/storage_backends/mariadb/config/z-hccl-vm.cnf.example /etc/mysql/mariadb.conf.d/z-hccl-vm.cnf
sudo chmod 644 /etc/mysql/mariadb.conf.d/z-hccl-vm.cnf
```

***Parameter configuration adjustment for database servers with different specifications***  
Edit `/etc/mysql/mariadb.conf.d/z-hccl-vm.cnf` (sudo permission required), fill in `thread_pool_size` (sample value 224) according to the number of logical CPUs available to the database, and restart after modification. For instances with 16/32/64 available logical CPUs, fill in 16/32/64 respectively. When CPU affinity is set, use the limited quota.
[MariaDB Thread Pool Description](https://mariadb.com/docs/server/ha-and-performance/optimization-and-tuning/buffers-caches-and-threads/thread-pool/thread-pool-in-mariadb).

```bash
sudo systemctl restart mariadb
sudo mariadb-admin ping
```

### 1.3 Verify the Effective Values After Restart
> **📌 Note:** The account password needs to be replaced with the account password actually created in section 1.2  
> **📌 Note:** The expected value 224 for `thread_pool_size` in the verification SQL is a sample value, replace it with the number of local CPUs written to the cnf in section 1.2 (for a 64-core instance, use 64); for this item status=1 indicates the actual effective value matches the cnf setting, status=-1 indicates the two do not match

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

### 1.4 Simulator Configuration

For initial deployment, create the connection configuration file; for daily operation, only set the environment variable:

```bash
# Initial deployment: Create the connection configuration file. The storage directory can be selected as needed (the sample uses /home/workspace); the file contains the database password, pay attention to permissions
cp ${HCCL_VM_SOURCE_HOME}/src/storage_backends/mariadb/config/storage_mariadb.yaml.example /home/workspace/storage_mariadb.yaml
chmod 600 /home/workspace/storage_mariadb.yaml

# Daily operation: Set for each shell session (must be set before starting the simulator)
export HCCL_VM_STORAGE_CONFIG=/home/workspace/storage_mariadb.yaml
```

> **📌 Note:** **This variable must be set before executing `hccl-vm start` to start the simulator**.

Replace the URI address in /home/workspace/storage_mariadb.yaml with the actual database address, and replace the account and password with the account and password created in [Section 1.2](#12-create-the-database-account):

```yaml
uri: mariadb://hccl_vm:password@127.0.0.1:3306
```

Log adjustment:
```bash
cp ${HCCL_VM_SOURCE_HOME}/config/log_config.yaml /home/workspace/log_config.yaml 
```

After the simulator `hccl-vm start`, use this account to confirm that storage is using MariaDB (replace the password with the actual password), the output should include the `hcclvm_` prefix databases automatically created by the simulator:

```bash
mariadb -h 127.0.0.1 -P 3306 -u hccl_vm -ppassword -e "SHOW DATABASES"
```

## 2. Environment Configuration

Before daily operation, verify whether the database service is running normally; if the connection is abnormal or the service is not ready, refer to [Chapter 1](#1-initial-deployment-reference) for troubleshooting or deployment:

```bash
mariadb-admin ping
```

> **📌 Important:** `HCCL_VM_STORAGE_CONFIG` **must be set before executing `hccl-vm start`**.

```bash
# Set before starting the simulator (point to the configuration file created in section 1.4)
export HCCL_VM_STORAGE_CONFIG=/home/workspace/storage_mariadb.yaml
```

### 2.1 MPICH Configuration

```bash
# When using MPICH, set before starting the simulator
export MPICH_NO_LOCAL=1
```

### 2.2 PMI Socket Buffer Configuration

Complete the environment configuration on the host machine running MPI.

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

## 3. hccl_test 1024p Checkonly Mode Reference Configuration
> **📌 Note:** The 1024p scenario requires server available resources greater than 500G, ***when less than this specification, swap can be expanded to make available memory meet this specification requirement, but swap is affected by disk and other hardware factors, execution may become slower or timeout***  
> **📌 Note:** The 1024p scenario requires server available cores greater than 100, ***when less than this specification, execution may become slower or timeout due to CPU scheduling***

Configuration sample:
```bash
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_DFS_CONFIG=cluster_heartbeat:off
export HCCL_VM_COMMUNICATOR_WAIT_TIMEOUT_MS=7200000
export HCCL_VM_STORAGE_CONFIG=/home/workspace/storage_mariadb.yaml
export HCCL_VM_LOG_CONFIG_PATH=/home/workspace/log_config.yaml 
export ASCEND_GLOBAL_LOG_LEVEL=3
```

HCCL_VM_LOG_CONFIG_PATH log level tuning sample:
``` bash
# Low log level configuration: Only retain warn+error (info/debug/trace all closed)
# Level: 0-trace 1-debug 2-info 3-warn 4-error 5-critical 6-off
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
    
# Topology generation script (generate_cluster_topo.sh/generate_server_topo.sh/allocate_eid.sh) log configuration
# console_level: Script console log level, consistent with C++ log level definition:
#   0-trace 1-debug 2-info 3-warn 4-error 5-critical 6-off
#   debug(1): Output all logs (including per-server/per-die/per-port details)
#   info(2):  Output steps and summary information (default)
#   warn(3):  Output only warnings and errors
#   error(4): Output only errors
#   off(6):   Do not output any logs
script:
    console_level: 3    
```

## Third-Party Dependencies

Third-party dependencies exclusive to this module:

| Open Source Software | Version | License (SPDX) | Download Address |
| --- | --- | --- | --- |
| mariadb-connector-c | 3.3.5 | LGPL-2.1-or-later | [v3.3.5.tar.gz](https://codeload.github.com/mariadb-corporation/mariadb-connector-c/tar.gz/refs/tags/v3.3.5) |

The license text is located in [licenses/](licenses/).

[Return to Main Guide](../../../README.md) · [Continue to Main Guide Usage Samples](../../../README.md#32-使用示例)

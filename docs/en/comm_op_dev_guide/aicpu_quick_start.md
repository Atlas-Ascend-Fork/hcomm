# AI CPU Operator

<!-- md-trans-meta sourceCommit=1435fccbb795a1ecf7fa4675cb37d8bf3e83f779 translatedAt=2026-10-08T07:54:03.311Z pushedAt=2026-10-09T07:18:21.873Z -->

This section uses the AI CPU point-to-point communication operator as an example to describe the overall process of developing communication operators using HCCL communication programming APIs, helping you quickly understand the development steps of communication operators.

## Point-to-Point Communication Operator

This section uses the point-to-point communication operators Send and Receive as an example:

- Send: Sends data from the local rank to the peer rank.
- Receive: Receives data sent from the peer rank. It must be used together with the Send operator.

## Supported Products

This sample supports the following products, with single-server *N*-device networking (*N* is an even number greater than or equal to 2):

- Ascend 950PR&950DT products
- Atlas A3 products

## Sample Introduction

You can click [here](https://gitcode.com/cann/hccl/tree/9.2.0/examples/04_custom_ops_p2p) to obtain the complete sample code. This sample uses HCCL communication operator development APIs to implement the Send and Receive operators based on the AI CPU communication engine. The main implementation process is as follows:

- **Query topology information of the communicator**: Call the topology query APIs **HcclGetRankId\(\)** and **HcclGetRankSize\(\)** to obtain the **rank_id** operated by the current thread and the number of ranks in the communicator.
- **Create thread resources**: Call the resource management API **HcclThreadAcquire\(\)** to allocate communication thread resources.
- **Establish communication channels**: Call **HcclChannelAcquire\(\)** to create channel links between ranks.
- **Obtain remote communication memory information**: Call **HcclChannelGetHcclBuffer\(\)** to obtain the remote communication memory address.
- **Prepare input data**: The Send operator calls **HcommLocalCopyOnThread\(\)** to copy input data to the communication memory.
- **Perform pre-synchronization**:
  - The Send operator calls **HcommChannelNotifyRecordOnThread\(\)** to notify the remote side that data is ready.
  - The Receive operator calls **HcommChannelNotifyWaitOnThread\(\)** to wait for the remote data to be ready.

- **Read remote data**: The Receive operator calls the **HcommReadOnThread\(\)** API to read data from the remote communication memory.
- **Perform post-synchronization**:
  - The Receive operator calls the **HcommChannelNotifyRecordOnThread\(\)** API to notify the remote end that the read is complete.
  - The Send operator calls the **HcommChannelNotifyWaitOnThread\(\)** API to wait for the remote end to complete the read.

**Figure 1** Send/Receive operator test sample diagram  
![Send/Receive operator test sample diagram](./figures/send_receive_sample.png)

In addition, the sample includes a test program that creates one communicator, where communication members with even ranks send data and those with odd ranks receive data. The data sent is the even rank number. It covers the following functional points:

- Queries the number of available devices by calling the **aclrtGetDeviceCount\(\)** API.
- Uses rank 0 as the root node and generates the **rootinfo** identifier of the root node by calling the **HcclGetRootInfo\(\)** API.
- Initializes the communicator in each thread based on the **rootinfo** identifier by calling the **HcclCommInitRootInfo\(\)** API.
- Calls the operator APIs **HcclSendCustom\(\)** and **HcclRecvCustom\(\)** to verify basic send/receive functionality and prints the receiving results.

## Build and Installation

Run the following commands in the root directory of the CANN/HCCL code repository to build and install the custom operator package:

```bash
# Set the CANN package environment variable. This example uses the default installation path of the root user.
source /usr/local/Ascend/cann/set_env.sh

# Run the build.sh script for build, and specify the custom operator project path via custom_ops_path.
bash build.sh --vendor=cust --ops=p2p --custom_ops_path=./examples/04_custom_ops_p2p

# The custom operator installation package is located in the build_out directory of the repository.
./build_out/cann-hccl_custom_p2p_linux-<arch>.run --install
```

The custom operator package installation information is as follows:

- Header file: **\$\{ASCEND_HOME_PATH\}/opp/vendors/cust/include/hccl_custom_p2p.h**
- Dynamic library: **\$\{ASCEND_HOME_PATH\}/opp/vendors/cust/lib64/libhccl_custom_p2p.so**
- AI CPU operator description file: **\$\{ASCEND_HOME_PATH\}/opp/vendors/cust/aicpu/config/libp2p_aicpu_kernel.json**
- AI CPU operator package: **\$\{ASCEND_HOME_PATH\}/opp/vendors/cust/aicpu/kernel/aicpu_hccl_custom_p2p.tar.gz**

- Installation script: **\$\{ASCEND_HOME_PATH\}/opp/vendors/cust/scripts/install.sh**

## Sample Execution

1. Disable the AI CPU operator signature verification feature.

    The AI CPU operator package is loaded to the device when the service starts. During the loading process, the driver performs security signature verification by default to ensure the trustworthiness of the package. However, AI CPU operator packages compiled by users do not contain a signature header, so you must manually disable the driver signature verification mechanism before the package can be loaded properly.

    Run the following command as the **root** user on the physical machine. This example uses device 0:

    ```bash
    npu-smi set -t custom-op-secverify-enable -i 0 -d 1    # Enable signature verification.
    npu-smi set -t custom-op-secverify-mode -i 0 -d 0      # Disable custom signature verification.
    ```

    > [!NOTE] Note
    > - Disabling the driver security signature verification mechanism poses certain security risks. You must ensure that the custom communication operators are secure and reliable to prevent malicious attacks.
    > - For more commands, see the AI CPU operator signature verification section in the *[npu-smi Command Reference](https://support.huawei.com/enterprise/en/ascend-computing/ascend-hdk-pid-252764743?category=reference-guides&subcategory=command-reference)* of the matching Ascend HDK version.

2. Modify the AI CPU allowlist.

    If you add an AI CPU operator package, you need to configure the package in the AI CPU allowlist. Taking the default installation path of the root user as an example, edit the **ascend_package_load.ini** file:

    ```bash
    vim /usr/local/Ascend/cann/conf/ascend_package_load.ini
    ```

    Append the following content to **ascend_package_load.ini**:

    ```text
    name:aicpu_hccl_custom_p2p.tar.gz
    install_path:2
    optional:true
    package_path:opp/vendors/cust/aicpu/kernel
    load_as_per_soc:false 
    ```

3. Compile and run the test sample.

    ```bash
    # Go to the sample code directory.
    cd examples/04_custom_ops_p2p/testcase
    # Compile
    make
    # Run the test case.
    make test
    ```

## Result Analysis

Nodes with even rank IDs are responsible for sending data, with the content being their rank IDs. Nodes with odd rank IDs are responsible for receiving data. Therefore, in the printed results, each odd rank receives the ID of the previous rank.

```text
Found 8 NPU device(s) available
rankId: 1, output: [ 0 0 0 0 0 0 0 0 ]
rankId: 3, output: [ 2 2 2 2 2 2 2 2 ]
rankId: 5, output: [ 4 4 4 4 4 4 4 4 ]
rankId: 7, output: [ 6 6 6 6 6 6 6 6 ]
```

## Key Code Analysis

The following uses the custom Send and Receive operators as an example to explain their implementation details.

1. Parse the topology information of the communicator.

    ```c
    uint32_t rank, rankSize;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    ```

2. Create resources.

    ```c
    CommEngine engine = CommEngine::COMM_ENGINE_AICPU;
    ACLCHECK(aclrtCreateNotify(&(g_notifies[0]), ACL_NOTIFY_DEFAULT));
    ACLCHECK(aclrtCreateNotify(&(g_notifies[1]), ACL_NOTIFY_DEFAULT));
    AlgResourceCtx resCtxHost;
    for (uint32_t idx = 0; idx < AICPU_CONTROL_NOTIFY_NUM; idx++) {
        ACLCHECK(aclrtGetNotifyId(g_notifies[idx], &(resCtxHost.notifyIds[idx])));
    }
    CHK_RET(HcclThreadAcquire(comm, engine, 1, 0, &(resCtxHost.threadHandle)));
    ```

3. Establish communication links.

    ```c
    HcclChannelDesc channelDesc;
    HcclChannelDescInit(&channelDesc, 1);
    channelDesc.remoteRank = destRank;
    channelDesc.channelProtocol = CommProtocol::COMM_PROTOCOL_HCCS;
    channelDesc.notifyNum = 2;
    CHK_RET(HcclChannelAcquire(comm, engine, &channelDesc, 1, &(resCtxHost.channelHandle)));
    ```

4. Obtain the remote communication memory address.

    ```c
    CHK_RET(HcclChannelGetHcclBuffer(comm, resCtxHost.channelHandle, &(resCtxHost.remoteBuffer.addr), &(resCtxHost.remoteBuffer.size)));
    ```

5. Read the remote data.

    ```c
    // One-sided read
    CHK_RET(HcommReadOnThread(resCtx->threadHandle, resCtx->channelHandle, param.outputPtr, resCtx->remoteBuffer.addr, size));
    ```

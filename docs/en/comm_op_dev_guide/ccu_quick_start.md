# CCU Operator

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-24T07:11:58.222Z pushedAt=2026-10-08T08:31:03.165Z -->

This section uses the CCU collective communication operator as an example to introduce the overall process of developing communication operators using HCCL communication programming APIs, helping you quickly understand the development steps of communication operators.

## Collective Communication Operator

This section uses the AllGather collective communication operator as an example. The AllGather operation reorders the input of all nodes in the communicator by rank ID (in ascending order), concatenates them, and then sends the result to the output buffer of all nodes.

![](./figures/allgather.png)

## Supported Products

This sample supports the following Ascend products. The networking requires Mesh interconnection that supports the UB protocol, with N devices (N>=2):

- Ascend 950PR&950DT products

## Sample Introduction

You can click [here](https://gitcode.com/cann/hccl/tree/master/examples/05_custom_ops_allgather/ccu) to obtain the complete CCU sample code. This sample uses the HCCL communication operator development APIs to implement the AllGather operator based on the CCU communication engine. The main implementation process is as follows:

- **Query the topology information of the communicator**: Call the topology information query APIs **HcclGetRankId\(\)** and **HcclGetRankSize\(\)** to obtain the **rank\_id** of the current thread and the number of ranks in the communicator.
- **Create thread resources**: Call the resource management API **HcclThreadAcquire\(\)** to allocate communication thread resources.
- **Establish communication channels**: Call the **HcclChannelAcquire\(\)** API to create channel links between ranks.
- **Register a CCU kernel**: Call **HcommCcuKernelRegister\(\)** to create a CCU kernel and generate CCU instructions for execution on the CCU.
- **Obtain local communication memory**: Call **HcclGetHcclBuffer\(\)** to obtain local communication memory information.
- **Prepare input data**: Call **HcommLocalCopyOnThread\(\)** to copy input data to the local communication memory.
- **Generate a token secret key**: Call **HcommCcuGetMemToken\(\)** to generate a secret key that uniquely identifies an address block for remote interaction.
- **Launch the CCU kernel**: Call **HcommCcuKernelLaunch\(\)** to launch the CCU kernel.
- **Perform pre-synchronization**: Call the **ccu::WriteVariableWithNotify\(\)** API to notify the remote end that data is ready.
- **Write data**: Call the **ccu::Write\(\)** API to write local data to the remote communication memory.
- **Perform post-synchronization**: Call the **ccu::NotifyRecord\(\)** and **ccu::NotifyWait\(\)** APIs to notify the remote end that the write is complete.

In addition, this sample includes a test program that creates one communicator, where each thread operates one device to jointly complete the AllGather operation. It covers the following features:

- Query the number of available devices by calling the **aclrtGetDeviceCount\(\)** API.
- Use rank0 as the root node, and call **HcclGetRootInfo\(\)** to generate the **rootinfo** identifier of the root node.
- In each thread, call **HcclCommInitRootInfo\(\)** based on the **rootinfo** identifier to initialize the communicator.
- Call the operator API **HcclAllGatherCustom\(\)** and print the receiving result.

## Build and Installation

In the root directory of the CANN/HCCL code repository, run the following commands to build and install the custom operator package:

```bash
# Set the CANN package environment variable. This example uses the default installation path of the root user.
source /usr/local/Ascend/cann/set_env.sh

# Run the build.sh script for build. Use custom_ops_path to specify the custom operator project path.
bash build.sh --vendor=cust --ops=allgather --custom_ops_path=./examples/05_custom_ops_allgather/ccu

# The custom operator installation package is located in the build_out directory of the code repository.
./build_out/cann-hccl_custom_allgather_linux-<arch>.run --install
```

The custom operator package installation information is as follows:

- Header file: **$\{ASCEND\_HOME\_PATH\}/opp/vendors/cust/include/hccl\_custom\_allgather.h**
- Dynamic library: **$\{ASCEND\_HOME\_PATH\}/opp/vendors/cust/lib64/libhccl\_custom\_allgather.so**

- Installation script: **$\{ASCEND\_HOME\_PATH\}/opp/vendors/cust/scripts/install.sh**

## Sample Execution

Build and run the test sample.

```bash
# Go to the sample code directory.
cd examples/05_custom_ops_allgather/ccu/testcase
# Compile.
make
# Run the test case.
make test
```

## Result Analysis

The input data of each node is initialized to the **DeviceId** of that node. After successful execution, the device outputs log information similar to the following (using a 2-device run as an example):

```text
Found 2 NPU device(s) available
rankId: 0, output: [ 0 1 ]
rankId: 1, output: [ 0 1 ]
```

## Key Code Analysis

The following uses the custom AllGather operator as an example to explain its implementation details.

1. Parse the topology information of the communicator.

    ```c
    uint32_t rank, rankSize;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    ```

2. Create resources.

    ```c
    // Allocate thread resources. In host mode, encapsulate the main stream as a thread and create a Notify on the main stream.
    ThreadHandle thread;
    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream, 0, &thread)); 
    ```

3. Establish a communication link.

    ```c
    // Allocate channel resources.
    uint32_t netLayer = 0, listSize = 0;
    CommLink *linkList = nullptr;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank, &linkList, &listSize)); // Obtain link information between srcRank and dstRank.
    
    HcclChannelDesc desc;
    CHK_RET(HcclChannelDescInit(&desc, 1));
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UB_CTP;
    bool protocolExists = false;
    for (uint32_t idx = 0; idx < listSize; idx++) {
        CommLink link = linkList[idx];
        if (link.linkAttr.linkProtocol == protocol) {
            desc.remoteRank = remoteRank;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
            protocolExists = true;
            break;
        }
    }
    if (!protocolExists) {
        HCCL_ERROR("[GetChannelForCcu] Protocol %d not found between rank %u and rank %u",
            protocol, param.myRank, remoteRank);
        return HCCL_E_NOT_FOUND;
    }
    CHK_RET(HcclChannelAcquire(comm, param.engine, &desc, 1, &kernelChannels[channelIndex])); // Obtain the channel handle.
    ```

4. Register the CCU kernel.

    ```c
    CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
    
    // Register the kernel.
    CcuKernelHandle kernelHandle;
    constexpr uint32_t dieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    const void *kernelArgsArr[] = { kernelInfo.kernelArg }; // Construct the input paramter pointer array based on the HcommCcuKernelRegister signature.
    CcuResult regRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName, reinterpret_cast<void*>(kernelInfo.kernelFunc),
                                              kernelArgsArr, kernelArgNum, &kernelHandle);
    
    resCtxHost.ccuKernels[0] = kernelHandle;
    CcuResult regEndRet = HcommCcuKernelRegisterEnd(insHandle);
    ```

5. Obtain the local HCCL Buffer address.

    ```c
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    ```

6. Prepare the input data.

    ```c
    HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize);
    ```

7. Generate a token secret key to uniquely identify a block of address information for remote interaction.

    ```c
    uint64_t token = 0;
    uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    if (param.inputPtr != nullptr) {
        HcommCcuGetMemToken(baseInputAddr, static_cast<uint64_t>(dataSize), &token);
    } else if (param.outputPtr != nullptr) {
        HcommCcuGetMemToken(baseOutputAddr, static_cast<uint64_t>(dataSize), &token);
    }
    ```

8. Launch the CCU kernel.

    ```c
    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        currentRankSliceInputOffset,
        currentRankSliceOutputOffset,
        sliceSize,
        goSize[0],
        goSize[1],
        goSize[2],
        goSize[3],
    };
    
    CcuResult launchRet = HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], taskArgs.data(), taskArgs.size());
    ```

9. Perform pre-synchronization to notify the remote end that the data is ready.

    ```c
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.output[ctx.arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    ```

10. Write local data to the remote communication memory.

    ```c
    CCU_IF(ctx.sliceSize != 0)
    {
        uint32_t channelId = 0;
        for (uint64_t rankIdx = 0; rankIdx < ctx.arg->rankSize; rankIdx++) {
            const uint16_t mask = 1 << rankIdx;
            if (rankIdx != ctx.arg->rankId) {
                CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelId], dst[rankIdx], src, ctx.sliceSize, ctx.event, mask)); // Write local device data to the remote address.
                channelId++;
            }
        }
    }
    ```

11. Perform post-synchronization to notify the remote end that writing is complete.

    ```c
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID)); // Wait for the remote device data movement to complete.
    }
    ```

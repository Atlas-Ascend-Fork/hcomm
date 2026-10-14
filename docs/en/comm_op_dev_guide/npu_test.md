# Testing in the NPU Environment

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-11T06:58:29.991Z pushedAt=2026-08-20T11:39:14.556Z -->

Developers need to write their own test cases for verification based on the functionality of custom communication operators, which mainly involves constructing input data and checking semantic logic. The following briefly describes the test method for communication operators.

## Writing a Test Program

The test program consists of the following six steps:

1. Allocate the memory required for the collective communication operation.
2. Construct the input data.
3. Initialize the communicator.
4. Call the custom communication operator.
5. Verify the communication result.
6. Release resources.

The following is the sample test program code (developers must modify the relevant logic as needed; the sample code below cannot be directly executed):

```c
#include "acl/acl_rt.h"
#include "hccl/hccl_types.h"
#include "<custom_ops_header>"    // Communication operator API header file defined by the developer

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                        \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                          \
    do {                                                                                        \
        if (ret != HCCL_SUCCESS) {                                                              \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                         \
        }                                                                                       \
    } while (0)

struct ThreadContext {
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
};

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    uint32_t device = ctx->device;
    uint64_t count = ctx->devCount;
    size_t mallocSize = count * sizeof(float);

    // Set the device for the current thread.
    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));

    // Initialize the collective communicator.
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    // Create a task stream.
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    // Construct the input data.
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, mallocSize));
    // TODO: Write the input data to hostBuf.

    // Copy the input data to the device.
    ACLCHECK(aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtFreeHost(hostBuf));

    // TODO: Call the custom collective communication operator.
    HCCLCHECK(HcclOpsCustom(sendBuf, count, HCCL_DATA_TYPE_FP32, hcclComm, stream));
 
    // Block and wait until the collective communication task in the stream is complete.
    ACLCHECK(aclrtSynchronizeStream(stream));

    // Copy the collective communication task result from the device to the host.
    void *resultHostBuf;
    ACLCHECK(aclrtMallocHost(&resultHostBuf, mallocSize));
    ACLCHECK(aclrtMemcpy(resultHostBuf, mallocSize, recvBuf, mallocSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // TODO: Verify the result data in resultHostBuf.
    ACLCHECK(aclrtFreeHost(resultHostBuf));

    // Release resources.
    HCCLCHECK(HcclCommDestroy(hcclComm));  // Destroy the communicator.
    if (sendBuf) {
        ACLCHECK(aclrtFree(sendBuf));      // Release the memory on the device.
    }
    if (recvBuf) {
        ACLCHECK(aclrtFree(recvBuf));      // Release the memory on the device.
    }
    ACLCHECK(aclrtDestroyStream(stream));  // Destroy the task stream.
    ACLCHECK(aclrtResetDevice(device));    // Reset the device.
    return 0;
}

int main()
{
    // Initialize device resources.
    ACLCHECK(aclInit(NULL));
    // Query the number of devices.
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    // Generate root node information. All threads use the same RootInfo.
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    // Start threads to perform collective communication operations.
    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    for (uint32_t i = 0; i < devCount; i++) {
        args[i].rootInfo = rootInfo;
        args[i].device = i;
        args[i].devCount = devCount;
        threads[i] = std::thread(Sample, (void *)&args[i]);
    }
    for (uint32_t i = 0; i < devCount; i++) {
        threads[i].join();
    }

    // Release resources.
    ACLCHECK(aclrtFreeHost(rootInfoBuf));  // Release host memory.
    ACLCHECK(aclFinalize());               // Deinitialize the device.
    return 0;
}
```

## Writing a Makefile

The following is an example of a Makefile. Developers must modify it as needed.

```text
ifndef ASCEND_HOME_PATH
    $(error "ASCEND_HOME_PATH is not set, please ensure CANN is properly installed and \
             source environment variables by running `source /path/to/Ascend/cann/set_env.sh`")
endif

CXXFLAGS := -std=c++17 \
        -Werror \
        -fstack-protector-strong \
        -fPIE -pie \
        -O2 \
        -s \
        -Wl,-z,relro \
        -Wl,-z,now \
        -Wl,-z,noexecstack \
        -Wl,--copy-dt-needed-entries

SOURCES = main.cc
ASCEND_INC_DIR = ${ASCEND_HOME_PATH}/include
ASCEND_LIB_DIR = ${ASCEND_HOME_PATH}/lib64
CUSTOM_OPS_INC_DIR = ${ASCEND_HOME_PATH}/opp/vendors/<vendor>/include
CUSTOM_OPS_LIB_DIR = ${ASCEND_HOME_PATH}/opp/vendors/<vendor>/lib64
LIBS = -L$(ASCEND_LIB_DIR) -lacl_rt -L${CUSTOM_OPS_LIB_DIR} -l<custom_ops_so>
INCS = -I$(ASCEND_INC_DIR) -I${CUSTOM_OPS_INC_DIR}
TARGET = test_custom_ops

# Default target
all:
    g++ $(CXXFLAGS) $(SOURCES) $(INCS) $(LIBS) -o ${TARGET}
    @echo "${TARGET} compile completed"

# Test target
test:
    export LD_LIBRARY_PATH=${CUSTOM_OPS_LIB_DIR}:${LD_LIBRARY_PATH}; \
    ./$(TARGET)

# Clean build artifacts
clean:
    rm ${TARGET}

.PHONY: all test clean
```

## Executing the Test Program

1. Disable the AI CPU operator signature verification feature.

    The AI CPU operator package is loaded to the device when the service starts. During loading, the driver performs security signature verification by default to ensure the trustworthiness of the package. However, the AI CPU operator package built by users does not contain a signature header, so the driver's signature verification mechanism must be manually disabled for normal loading.

    Refer to the following command and run it as the root user on the physical machine. The following uses device 0 as an example:

    ```bash
    npu-smi set -t custom-op-secverify-enable -i 0 -d 1    # Enable signature verification configuration.
    npu-smi set -t custom-op-secverify-mode -i 0 -d 0      # Disable custom signature verification.
    ```

   > [!NOTE] Note
   > Disabling the driver signature verification mechanism poses certain security risks. You must ensure that the custom communication operator is secure and reliable to prevent malicious attacks.

2. Modify the AI CPU allowlist.

    If you add an AI CPU operator package, you must also configure the AI CPU operator package in the AI CPU allowlist. Taking the default installation path of the root user as an example, edit the **ascend_package_load.ini** file:

    ```bash
    vim /usr/local/Ascend/cann/conf/ascend_package_load.ini
    ```

    Append the following content to **ascend_package_load.ini**:

    ```text
    name:<aicpu_kernel_file_name>
    install_path:2
    optional:true
    package_path:<aicpu_kernel_file_path>
    ```

    Where:

    - **<aicpu_kernel_file_name\>**: indicates the AI CPU operator package file name, in tar.gz format, for example, **aicpu_hccl_custom_p2p.tar.gz**.
    - **<aicpu_kernel_file_path\>**: indicates the relative path of the AI CPU operator package under the CANN package, for example, **opp/vendors/cust/aicpu/kernel**.

3. Build and execute the test sample.

    ```bash
    # Build
    make
    # Execute the test case.
    make test
    ```

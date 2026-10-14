# HCOMM Header Files and Library Files

<!-- md-trans-meta sourceCommit=ea28a2a84bd47924bd51751e37d9926552e1ad78 translatedAt=2026-09-28T06:07:14.081Z pushedAt=2026-10-08T07:43:27.236Z -->

Huawei Communication (HCOMM) is the communication base library of HCCL. It provides communicator and communication resource management capabilities and serves as the underlying foundation of the CANN collective communication stack. HCCL dynamically loads HCOMM APIs through dlsym. HCCL and HCOMM are compiled and versioned independently.

This section describes the header files and library files of the external HCOMM APIs.

## API Categories

HCOMM external APIs are organized by API layers. The layering relationships and architectural constraints are authoritatively defined in the *External API Layering Relationships* and *Software Architecture Constraint Description* sections of [Architecture Introduction](../architecture/architecture-brief.md):

**Table 1** API categories

| API Category | Target Audience | Description |
| --- | --- | --- |
| Communicator management (L2-comm) | AI framework layer | APIs for communicator initialization and destruction. |
| Resources and topology (L2-res) | Operator developers | APIs for querying resources and topology such as the channel, memory, topology, and endpoint. |
| Basic primitives (L3-prim) | Operator/communication library developers | Data movement and synchronization primitive types. |
| Basic resources (L3-res) | Communication library developers | Resource management APIs such as endpoint/channel/memory registration. |
| CCU operator development | CCU operator developers | APIs for Collective Communication Unit (CCU) resource object encapsulation and kernel launch. |

## Header Files and Library Files Required for API Call

After the firmware, driver, and CANN software packages are installed, you can reference the HCOMM API header files and library files when compiling and running an application.

The external HCOMM header files are located in the `hccl/`, `hcomm/`, and `hcomm/ccu/` subdirectories under the `${INSTALL_DIR}/include/` directory, and the library files are located in the `${INSTALL_DIR}/lib64/` directory. Replace `${INSTALL_DIR}` with the path where the CANN software files are stored after installation. For example, if you install the software as the root user, the default storage path is `/usr/local/Ascend/cann`.

> [!CAUTION] Caution
> The header files in the `include/` directory are stable external header files. The `pkg_inc/` directory contains the inter-package APIs between HCOMM and HCCL, GE, and other components. These APIs are installed in a separate inter-package directory and their stability is not guaranteed. Do not directly reference them in external services. When compiling an API program, reference the library files corresponding to the included header files. Referencing unnecessary .so files may cause version function exceptions or compatibility issues during later version upgrades.

You need to include the required files based on the HCOMM APIs you actually use. The purpose of each header file is described in the following table.

**Table 2** Header files

| Header File | Purpose | Corresponding Library File |
| --- | --- | --- |
| hccl/hccl_comm.h | Used to define C APIs such as communicator initialization and destruction (weak symbols). | libhcomm.so |
| hccl/hccl_types.h | Used to define HCCL return code enums and basic types. | libhcomm.so |
| hccl/hccn_rping.h | Used to define HCCN RPing (Remote Ping) network connectivity detection C APIs, including types such as **HccnRpingCtx**, **HccnResult**, and **HccnRpingMode**, and detection APIs. | libhcomm.so |
| hccl/hccl_rank_graph.h | Used to define communication topology, endpoint attributes, and heterogeneous networking enums. | libhcomm.so |
| hccl/hccl_ccu_res.h | Used to define C APIs for querying CCU instance handles within a communicator. | libhcomm.so |
| hccl/hccl_res.h | Used to define HCCL communicator resource management (Thread/EngineCtx/memory registration) APIs and constants. | libhcomm.so |
| hccl/hccl_channel.h | Used to define HCCL channel description, channel creation/destruction, channel configuration, and other channel management APIs and constants. | libhcomm.so |
| hccl/hccl_sym_win.h | Used to define symmetric memory window (Symmetric Window) access APIs. | libhcomm.so |
| hccl/hccl_launch.h | Used to define P2P operator description and launch-related structures. | libhcomm.so |
| hcomm/hcomm_primitives.h | Used to define basic primitive types such as channel/thread handles and reduction operators, and provides data movement and synchronization primitives. | libhcomm.so |
| hcomm/hcomm_res.h | Used to define C APIs for basic resource management such as endpoint/memory registration/thread. | libhcomm.so |
| hcomm/hcomm_res_defs.h | Used to define HCOMM ABI version, handles, and resource description structures. | libhcomm.so |
| hcomm/hcomm_channel.h | Used to define HCOMM channel description, channel creation/destruction, channel configuration, and other channel management C APIs and types. | libhcomm.so |
| hcomm/ccu/ccu_primitives.hpp | CCU primitive aggregation header, containing type aliases and resource creation entry points. | libhcomm.so |
| hcomm/ccu/ccu_launch.h | Used to define C APIs for CCU kernel registration and launch (weak symbols). | libhcomm.so |
| hcomm/ccu/ccu_res.h | Used to define C APIs for CCU resource management and memory CCU access token (**HcommCcuGetMemToken**). | libhcomm.so |
| hcomm/ccu/ccu_types.h | Used to define basic types such as CCU return codes and condition types. | libhcomm.so |
| hcomm/ccu/ccu_control_flow_macro.h | Used to define control flow macros such as CCU while loops. | libhcomm.so |
| hcomm/ccu/ccu_address.hpp | CCU address object encapsulation (**Address** class). | libhcomm.so |
| hcomm/ccu/ccu_local_addr.hpp | CCU local address object encapsulation (**LocalAddr** class). | libhcomm.so |
| hcomm/ccu/ccu_remote_addr.hpp | CCU remote address object encapsulation (**RemoteAddr** class). | libhcomm.so |
| hcomm/ccu/ccu_buffer.hpp | CCU Buffer resource object encapsulation (**CcuBuffer** class). | libhcomm.so |
| hcomm/ccu/ccu_array.hpp | CCU resource contiguous container **Array** template, used for batch allocation. | libhcomm.so |
| hcomm/ccu/ccu_event.hpp | CCU Event resource object encapsulation (**Event** class). | libhcomm.so |
| hcomm/ccu/ccu_variable.hpp | CCU Variable resource object encapsulation (**Variable** class). | libhcomm.so |
| hcomm/ccu/ccu_loop.hpp | CCU loop structure object encapsulation. | libhcomm.so |
| hcomm/ccu/ccu_func.hpp | Utility template that wraps a lambda into a CCU Func. | libhcomm.so |
| hcomm/ccu/ccu_utils.hpp | Internal auxiliary definitions such as CCU exception classes and operator utilities. | libhcomm.so |

HCOMM is released as the `cann-hcomm_<version>_linux-<arch>.run` installation package, which contains `libhcomm.so`, external header files, and the `cann-hcomm-compat.tar.gz` compatible upgrade subpackage.

- For details about source code compilation and installation, see [Build Guide](../build/build.md).
- For details about the prototype definitions, parameter descriptions, and constraints of communicator management APIs, see [Communicator Creation and Management APIs (C Language)](./comm_mgr_c/README.md).
- For details about the prototype definitions, parameter descriptions, and constraints of communication operator development APIs, see [Communication Operator Development APIs](./comm_opdev/README.md).

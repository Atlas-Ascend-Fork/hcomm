# Build and Deployment

<!-- md-trans-meta sourceCommit=b91b5fc610070a3c9e821377fe6b4d1c01750eae translatedAt=2026-10-08T07:54:46.280Z pushedAt=2026-10-09T07:18:21.874Z -->

After developing a communication operator, you need to deploy it to the operating environment for functional verification.

## Understanding the Custom Operator Build and Packaging Project

The HCCL code repository provides a custom operator build and packaging project. You can refer to the [custom Send/Receive communication operator example](https://gitcode.com/cann/hccl/tree/9.2.0/examples/04_custom_ops_p2p) for development. The directory structure is as follows:

```text
├── build.sh                                # Entry point of the HCCL code repository build project
├── CMakeLists.txt                          # HCCL repository root build configuration file
├── cmake/                                  # CMake functions
│   ├── config.cmake
│   ├── func.cmake
│   ├── package.cmake
│   └── makeself_custom.cmake
├── scripts/
│   ├── custom/
│   │   └── install.sh                     # Custom operator package installation script
│   └── sign/
│       └── add_header_sign.py              # AI CPU operator package signing script
└── examples/04_custom_ops_p2p               # Custom operator project directory
    ├── CMakeLists.txt                       # Custom operator build configuration file
    ├── op_host/
    │   ├── send.cc                         # HcclSendCustom operator implementation source file
    │   ├── recv.cc                         # HcclRecvCustom operator implementation source file
    │   ├── load_kernel.cc                  # AI CPU kernel loading logic on the host side
    │   └── launch_kernel.cc                # AI CPU kernel launch logic on the host side
    ├── op_kernel_aicpu/
    │   ├── libp2p_aicpu_kernel.json        # AI CPU kernel operator description file
    │   ├── aicpu_kernel.cc                 # AI CPU kernel implementation logic
    │   └── exec_op.cc                      # AI CPU operator orchestration logic
    ├── inc/
    │   ├── hccl_custom_p2p.h               # Custom Send/Receive operator interface header file
    │   ├── common.h                        # Common type header file
    │   └── log.h                           # Log macro definitions
    ├── scripts/
    │   └── hccl_custom_p2p_check_cfg.xml   # Signing configuration file
    └── testcase/
        ├── main.cc                          # Test sample source file
        └── Makefile                         # Build configuration file
```

You are advised to organize code files based on the directory structure above. Pay attention to the following key points:

1. Place custom operator header files in the **inc** folder.
2. Place host-side implementation code in the **op_host** folder, and AI CPU-side implementation code in the **op_kernel_aicpu** folder.
3. Adjust the **CMakeLists.txt** file as needed.

## Building a Custom Operator Package

1. Set the CANN software environment variables.

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

    **/usr/local/Ascend** is the default installation path of CANN for the **root** user. If you install CANN as a non-root user or in a custom path, replace it with the actual path.

2. Download the HCCL code repository.

    ```bash
    git clone https://gitcode.com/cann/hccl.git
    ```

3. Build the custom operator package.

    ```bash
    bash build.sh --vendor=<vendor> --ops=<ops> --custom_ops_path=<ops_project_path>
    ```

    Where:

    - **<vendor\>**: Custom operator package identifier, which is user-defined and must be unique.
    - **<ops\>**: Custom operator name, which is user-defined and must be unique.
    - **<ops_project_path\>**: Root directory of the custom operator project, for example, the **./examples/04_custom_ops_p2p** directory in [Understanding the Custom Operator Build and Packaging Project](#understanding-the-custom-operator-build-and-packaging-project). It can be configured as an absolute or relative path.

    After the build is complete, a custom operator installation package **cann-hccl_custom_<ops\>_linux-<arch\>.run** is generated in the **build_out** directory of the current directory, where:

    - **<ops\>**: indicates the operator name specified by the **--ops** parameter when building the custom operator package.
    - **<arch\>**: indicates the system architecture of the current build environment.

## Deploying the Custom Operator Package

Run the following command to install the package:

```bash
./build_out/cann-hccl_custom_<ops>_linux-<arch>.run --install
```

The custom operator package installation information is as follows:

- Header file: ***$\{ASCEND_HOME_PATH\}*/opp/vendors/<vendor\>/include**
- Dynamic library: ***$\{ASCEND_HOME_PATH\}*/opp/vendors/<vendor\>/lib64**
- AI CPU operator information library file: ***$\{ASCEND_HOME_PATH\}*/opp/vendors/<vendor\>/aicpu/config**
- AI CPU operator package: ***$\{ASCEND_HOME_PATH\}*/opp/vendors/<vendor\>/aicpu/kernel**
- Installation script: ***$\{ASCEND_HOME_PATH\}*/opp/vendors/<vendor\>/scripts/install.sh**

Here, ***<vendor\>*** is the operator identifier specified by the **--vendor** parameter when compiling a custom operator package.

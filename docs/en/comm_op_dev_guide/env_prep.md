# Environment Setup

<!-- md-trans-meta sourceCommit=b91b5fc610070a3c9e821377fe6b4d1c01750eae translatedAt=2026-10-08T07:54:01.695Z pushedAt=2026-10-09T07:18:21.871Z -->

1. Install the driver, firmware, and CANN software package.

    The development and use of HCCL communication operators depend on the CANN software package. For detailed installation steps, see *[CANN Software Installation](https://hiascend.com/document/redirect/CannCommunityInstSoftware)*.

2. Set the environment variables.

    Before building and running the program, you need to set the CANN software environment variables.

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

    **/usr/local/Ascend** is the default installation path of CANN for the **root** user. If you install CANN as a non-root user or in a custom path, replace it with the actual path.

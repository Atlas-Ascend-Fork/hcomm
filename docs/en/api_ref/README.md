# HCOMM API Overview

<!-- md-trans-meta sourceCommit=8167da4e7ea711e122efa4d6174265e53a4c6bc8 translatedAt=2026-09-28T06:05:56.757Z pushedAt=2026-10-08T06:47:40.083Z -->

- [External Header Files and Library Files](./hcomm_header_and_lib.md): Lists the header files and library files officially exposed by HCOMM for reference in communication library/operator development and deployment.
- [Communicator Creation and Management APIs (C Language)](./comm_mgr_c/README.md): Used to implement framework adaptation in single-operator mode and enable distributed capabilities.
- [Communicator Creation and Management APIs (Python Language)](./comm_mgr_python/README.md): Used to implement framework adaptation in graph mode. Currently, it is only used for distributed optimization of TensorFlow networks on the NPU.
- [Communication Operator Development APIs](./comm_opdev/README.md): Provides control-plane and data-plane APIs to support developers in customizing communication operators.

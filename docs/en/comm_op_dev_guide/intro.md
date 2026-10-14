# Introduction

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-24T07:12:21.075Z pushedAt=2026-09-28T05:59:29.467Z -->

## Technical Background and Benefits

As large language models evolve toward deployment on clusters of tens of thousands of devices, traditional collective communication libraries face the following challenges:

- Built-in communication algorithms struggle to maintain optimal performance across diverse scenarios.
- With the trend toward convergence of communication and compute, users increasingly demand flexible communication operator programming semantics.

The closed black-box design of traditional collective communication libraries limits researchers from exploring new communication primitives. To address this, HCCL opens up its underlying communication capabilities and provides lightweight communication operator development APIs, enabling full-stack programmability of communication operators and facilitating innovation in communication schemes.

The HCCL communication operator development API provides the following key features:

- Supports multiple communication engines on Ascend devices, fully leveraging hardware capabilities.
- Supports multiple communication protocols, including PCIe, HCCS, RoCE, and UB.
- Decouples communication platform capabilities from communication operator development, enabling independent development of communication operators.

## Software Architecture

HCCL is a core component of CANN, providing high-performance and highly reliable communication solutions for NPU clusters. HCCL supports multiple AI frameworks at the upper layer and enables efficient interconnection between various Ascend AI processors at the lower layer. Its architecture is shown in the following figure.

**Figure 1**  Collective communication library software architecture  
![Collective communication library software architecture](figures/hccl_software_architecture.png)

HCCL consists of the HCCL collective communication library and the Huawei Communication (HCOMM) basics library:

- **HCCL collective communication library**: Contains built-in communication operators and extended communication operators, and provides external communication operator APIs.
  - Built-in communication operators: basic communication operators provided by HCCL, including collective communication operators and point-to-point communication operators.
  - Extended communication operators: Users can use the APIs provided by the HCOMM basics library to customize extended communication operators.

- **HCOMM basics library**: adopts a layered and decoupled design approach, dividing communication capabilities into a control plane and a data plane.

  - Control plane: provides topology information query and communication resource management functions.
  - Data plane: provides data movement and compute functions such as local operations, inter-operator synchronization, and communication operations.

    The control plane provides communication resources, and the data plane provides operation resources. The related APIs allow communication operator developers to focus on service innovation without concerning themselves with the complex implementation details at the chip level.

## Supported Product Models

The custom communication operator development feature currently supports the following products:

Ascend 950PR&950DT products

Atlas A3 products

Atlas A2 products (For the Atlas A2 products, only Atlas 800I A2 inference server, Atlas 300I A2 inference card, and A200I A2 Box heterogeneous component are supported.)

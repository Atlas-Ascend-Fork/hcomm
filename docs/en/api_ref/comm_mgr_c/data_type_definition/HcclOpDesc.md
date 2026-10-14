# HcclOpDesc

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T06:38:51.698Z pushedAt=2026-09-28T08:28:18.384Z -->

## Description

Describes the operator information when the AI CPU runs a kernel function, including the operator type, name, and communication-related parameters. This structure is used in custom communication operator scenarios and supports the description of Send and Receive communication tasks.

## Prototype

```c
typedef struct {
    CommAbiHeader header;
    uint32_t opDescType;
    char opName[HCCL_OP_DESC_OP_NAME_MAX_LEN];
    union {
        uint8_t raws[76];
        HcclOpP2pDesc p2p;
    };
} HcclOpDesc;
```

## Parameters

- **header**: ABI header, which contains information such as the version. For the type definition, see [CommAbiHeader](../../comm_opdev/datatype_definition/CommAbiHeader.md).
- **opDescType**: operator description type.
- **opName**: operator name, with a maximum length of 256 bytes (**HCCL_OP_DESC_OP_NAME_MAX_LEN**).
- **raws**: raw data used in general scenarios, with a length of 76 bytes.
- **p2p**: description parameters of Send and Receive communication tasks. See [HcclOpP2pDesc](./HcclOpP2pDesc.md) for the type definition. As a union member, it shares the 76-byte space with raws.

## Related Constants

```c
const uint32_t HCCL_OP_DESC_OP_NAME_MAX_LEN = 256;  // Maximum length of the operator name.
```

# HcclOpP2pDesc

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T06:39:58.752Z pushedAt=2026-09-28T08:34:30.677Z -->

## Description

Describes the detailed information of Send and Receive communication tasks, including parameters such as the data buffer address, communication command type, data type, number of data elements, remote rank ID, and unfold stream. This structure is used as a union member of [HcclOpDesc](./HcclOpDesc.md) and shares 76 bytes of space with `raws`.

## Prototype

```c
typedef struct {
    void *buffer;
    uint8_t reserved[8];
    HcclCMDType cmdType;
    HcclDataType dataType;
    uint64_t count;
    uint32_t remoteRank;
    void *unfoldStream;
} HcclOpP2pDesc;
```

## Parameters

- **buffer**: data buffer address, which is the memory address used to send or receive data. Ensure that the memory is correctly allocated and accessible.
- **reserved**: reserved field, 8 bytes in length, for future extension.
- **cmdType**: communication command type, which specifies the type of communication operation (such as **HCCL_CMD_SEND**, **HCCL_CMD_RECEIVE**). See [HcclCMDType](./HcclCMDType.md) for the definition of the **HcclCMDType** type.
- **dataType**: data type. For the type definition, see [HcclDataType](./HcclDataType.md).
- **count**: number of data elements to be transferred. It must match the actual data size and type of the buffer corresponding to **dataType**.
- **remoteRank**: remote rank ID, which specifies the ID of the peer node for communication. It must be within the valid rank ID range of the communicator.
- **unfoldStream**: unfold stream, used for stream control of AI CPU communication tasks.

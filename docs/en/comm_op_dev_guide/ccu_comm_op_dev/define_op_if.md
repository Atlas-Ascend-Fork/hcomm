# Defining Operator APIs

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-24T07:07:41.114Z pushedAt=2026-09-28T05:59:29.459Z -->

Developers need to first create an operator API definition header file based on the functionality of the communication operator. Subsequent calls to this communication operator depend on this header file.

Take the custom communication operator AllGather as an example. This type of operator API requires the source address, destination address, source data size, data type, as well as the communicator and stream information. Its API definition is as follows:

```c
HcclResult HcclAllGatherCustom(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream);
```

# Defining Operator APIs

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-11T07:09:48.966Z pushedAt=2026-08-20T11:39:14.564Z -->

Developers must first create an operator API definition header file based on the functionality of the communication operator. Subsequent calls to this communication operator depend on this header file.

Take the custom point-to-point communication operators Send and Receive as an example. The Send operator sends data at a specified location on the local rank to the remote end, while the Receive operator receives data from the remote end to a specified location on the local rank. The two must be used in pairs. Therefore, in addition to communicator information and stream information, this type of operator API also requires the data address, data size, data type, and peer rank number. The API is defined as follows:

```c
HcclResult HcclSendCustom(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream);
HcclResult HcclRecvCustom(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream);
```

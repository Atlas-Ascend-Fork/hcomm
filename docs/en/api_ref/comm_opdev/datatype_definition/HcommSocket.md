# HcommSocket

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-28T09:01:05.313Z pushedAt=2026-10-08T06:22:15.647Z -->

## Description

Socket handle type in HCOMM basic communication, used to identify pre-created socket resources. It is used as a field in [HcommChannelDesc](./HcommChannelDesc.md) to pass in the pre-created socket when creating a channel.

## Prototype

```c
typedef void *HcommSocket;
```

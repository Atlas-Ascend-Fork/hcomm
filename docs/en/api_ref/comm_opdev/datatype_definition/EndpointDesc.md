# EndpointDesc

<!-- md-trans-meta sourceCommit=7ff807bedd6173de4c7cb9ba16dadd5138b23868 translatedAt=2026-09-28T08:54:56.779Z pushedAt=2026-10-08T04:00:09.118Z -->

## Description

Structure defining the endpoint description type.

## Prototype

```c
typedef struct {
    CommProtocol protocol;  /* Communication protocol. */
    CommAddr commAddr;      /* Communication address. */
    EndpointLoc loc;        /* Location information of the endpoint. */
    union {
        uint8_t raws[52];   /* General data. */
    };
} EndpointDesc;
```

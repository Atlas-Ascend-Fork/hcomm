# EndpointLocType

<!-- md-trans-meta sourceCommit=7ff807bedd6173de4c7cb9ba16dadd5138b23868 translatedAt=2026-09-28T08:55:25.460Z pushedAt=2026-10-08T04:01:38.180Z -->

## Description

Defines the location of the communication device endpoint.

## Prototype

```c
typedef enum {
    ENDPOINT_LOC_TYPE_RESERVED = -1,  /* Reserved endpoint location. */
    ENDPOINT_LOC_TYPE_DEVICE = 0,     /* Endpoint on the device. */
    ENDPOINT_LOC_TYPE_HOST = 1,       /* Endpoint on the host. */
} EndpointLocType;
```

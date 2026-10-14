# HcommSocketRole

<!-- md-trans-meta sourceCommit=7494246995d6d252dbb49e1c6304dffdf77e6d53 translatedAt=2026-09-28T09:01:32.698Z pushedAt=2026-10-08T06:22:35.798Z -->

## Description

Defines the socket role type.

## Prototype

```c
typedef enum {
    HCOMM_SOCKET_ROLE_RESERVED = -1, /* Reserved socket role. */
    HCOMM_SOCKET_ROLE_CLIENT = 0,    /* Client role, used to initiate connections. */
    HCOMM_SOCKET_ROLE_SERVER = 1,    /* Server role, used to listen for connections. */
} HcommSocketRole;
```

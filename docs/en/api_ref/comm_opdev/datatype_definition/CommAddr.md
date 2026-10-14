# CommAddr

<!-- md-trans-meta sourceCommit=7494246995d6d252dbb49e1c6304dffdf77e6d53 translatedAt=2026-09-28T08:52:12.312Z pushedAt=2026-10-08T03:55:28.757Z -->

## Description

Communication device address description structure.

## Prototype

```c
static const uint32_t COMM_ADDR_EID_LEN = 16U;
typedef struct {
    CommAddrType type;         /* Communication address type. */
    union {
        uint8_t raws[36];      /* Generic data. */
        struct in_addr addr;   /* IPv4 address structure. */
        struct in6_addr addr6; /* IPv6 address structure. */
        uint32_t id;           /* Identifier. */
        uint8_t eid[COMM_ADDR_EID_LEN];  /* EID address type. */
    };
} CommAddr;
```

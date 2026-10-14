# CommAddrType

<!-- md-trans-meta sourceCommit=ea7699d628e74a1d296238ca1b93f1da5a4e757c translatedAt=2026-09-28T08:52:14.629Z pushedAt=2026-10-08T03:55:41.014Z -->

## Description

Address types of communication devices.

## Prototype

```c
typedef enum {
    COMM_ADDR_TYPE_RESERVED = -1, /* Reserved address type. */
    COMM_ADDR_TYPE_IP_V4 = 0,     /* IPv4 address type. */
    COMM_ADDR_TYPE_IP_V6 = 1,     /* IPv6 address type. */
    COMM_ADDR_TYPE_ID = 2,        /* ID address type. */
    COMM_ADDR_TYPE_EID = 3,       /* EID address type. */
} CommAddrType;
```

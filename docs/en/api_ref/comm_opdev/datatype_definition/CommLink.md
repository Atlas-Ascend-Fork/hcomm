# CommLink

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-09-28T08:52:39.767Z pushedAt=2026-10-08T03:56:39.454Z -->

## Description

Communication connection information, including the protocol, address, and other information required for creating a communication channel.

## Prototype

```c
typedef struct {
    CommAbiHeader header; /* Abi-compatible field. */
    EndpointDesc srcEndpointDesc; /* Source endpoint description type. */
    EndpointDesc dstEndpointDesc; /* Destination endpoint description type. */
    union {
        uint8_t raws[128];
        struct {
            CommProtocol linkProtocol; /* Communication protocol type. */
            uint8_t hop; /* Link hop count. */
        };
    } linkAttr;
} CommLink;
```

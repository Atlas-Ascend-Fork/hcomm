# EndpointAttr

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-09-28T08:53:59.205Z pushedAt=2026-10-08T03:59:26.926Z -->

## Description

Defines the attributes of an endpoint.

## Prototype

```c
typedef  enum {
    ENDPOINT_ATTR_INVALID = -1, /* Invalid attribute. */
    ENDPOINT_ATTR_BW_COEFF =  0, /* Bandwidth attribute. */
    ENDPOINT_ATTR_DIE_ID = 1,   /* DIE ID */
    ENDPOINT_ATTR_LOCATION = 2, /* Location of the endpoint. */
} EndpointAttr;
```

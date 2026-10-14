# CommMemType

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-28T08:53:03.071Z pushedAt=2026-10-08T03:57:52.125Z -->

## Description

Memory physical location type.

## Prototype

```c
typedef enum {
    COMM_MEM_TYPE_INVALID = -1,   /* Invalid memory type. */
    COMM_MEM_TYPE_DEVICE = 0,     /* Device-side memory (such as NPU). */
    COMM_MEM_TYPE_HOST = 1,       /* Host-side memory. */
    COMM_MEM_TYPE_CCU = 2,        /* CCU resource space. */
} CommMemType;
```

## Constraints

- `COMM_MEM_TYPE_CCU` indicates CCU resource space memory. This type is valid only on products that support CCU (Ascend 950PR&950DT products).
- The registration process for CCU-type memory is the same as that for the device, both completed through the standard memory registration path.

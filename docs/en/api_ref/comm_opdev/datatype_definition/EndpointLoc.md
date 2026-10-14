# EndpointLoc

<!-- md-trans-meta sourceCommit=7ff807bedd6173de4c7cb9ba16dadd5138b23868 translatedAt=2026-09-28T08:55:20.712Z pushedAt=2026-10-08T04:01:18.324Z -->

## Description

Structure defining the endpoint location type.

## Prototype

```c
typedef struct {
    EndpointLocType locType;        /* Location type of the endpoint. */
    union {
        uint8_t raws[60];           /* General data. */
        struct {
            uint32_t devPhyId;      /* Physical ID of the device. */
            uint32_t superDevId;    /* Device ID of the SuperPoD. */
            uint32_t serverIdx;     /* Index of the server. */
            uint32_t superPodIdx;   /* Position index of the SuperPoD. */
        } device;                   /* Used when locType is DEVICE. */
        struct {
            uint32_t id;            /* Common ID, used when locType is HOST or the like. */
        } host;
    };
} EndpointLoc;
```

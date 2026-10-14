# ThreadConfig

<!-- md-trans-meta sourceCommit=4af10b8ce51fdf33b4182261887dbef8ef8acfcc translatedAt=2026-09-28T09:02:30.639Z pushedAt=2026-10-08T06:29:10.136Z -->

## Description

Thread configuration structure used to configure the number of synchronization resources per thread.

## Prototype

```c
typedef struct {
    CommAbiHeader header;              /* ABI header containing version and other information. */
    uint16_t notifyNumPerThread;       /* Number of synchronization resources (Notify) in each communication thread. */
    uint8_t reserved[14];              /* Reserved field. */
} ThreadConfig;
```

## Member Description

| Member | Description |
| --- | --- |
| header | ABI header containing version and other information. For the definition of the CommAbiHeader type, see [CommAbiHeader](CommAbiHeader.md). |
| notifyNumPerThread | Number of synchronization resources (Notify) in each communication thread. The value ranges from 0 to 65535. Limited by the underlying Notify resource pool, the total number of synchronization resources of all threads in a communicator cannot exceed 65536. |
| reserved | Reserved field. |

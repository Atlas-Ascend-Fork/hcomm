# Querying Topology Information

<!-- md-trans-meta sourceCommit=2b912e7d2b105735579276a63a51364890c56788 translatedAt=2026-09-24T07:06:13.141Z pushedAt=2026-09-28T05:59:29.456Z -->

## Background

To cope with complex network topologies, communication operators need to select the most suitable algorithm based on the topology of the communicator. To this end, the HCCL control plane provides the topology information query function for developers.

## Topology Information

The following table lists the topology information that can be queried through HCCL control plane APIs.

| Topology Information | Query API |
| --- | --- |
| Obtains the rank ID of a device in a specified communicator. | [HcclGetRankId](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclGetRankId.md) |
| Queries the number of ranks in a specified communicator. | [HcclGetRankSize](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclGetRankSize.md) |
| Queries the list of topology hierarchy numbers that include the current rank and the number of topology hierarchies. | [HcclRankGraphGetLayers](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetLayers.md) |
| Given a communicator and a topology hierarchy number, returns the list of all rank IDs and the number of ranks in the topology instance where the current rank resides at this hierarchy. | [HcclRankGraphGetRanksByLayer](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetRanksByLayer.md) |
| Given a communicator and a topology hierarchy number, returns the number of ranks in the topology instance where the current rank resides at this hierarchy. | [HcclRankGraphGetRankSizeByLayer](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetRankSizeByLayer.md) |
| Given a communicator and a topology hierarchy number, returns the topology type of the hierarchy where the current rank resides. | [HcclRankGraphGetTopoTypeByLayer](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetTopoTypeByLayer.md) |
| Given a communicator and a topology hierarchy number, queries the number of topology instances at this hierarchy and the number of ranks in each instance. | [HcclRankGraphGetInstSizeListByLayer](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetInstSizeListByLayer.md) |
| Given a communicator and a topology hierarchy number, queries the communication link information between the source rank and the destination rank. | [HcclRankGraphGetLinks](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetLinks.md) |

## Sample Code

- Query the ID of the current rank in the communicator.

    ```c
    u32 userRank = INVALID_VALUE_RANKID;
    HcclResult ret = HcclGetRankId(comm, &userRank);
    if (userRank == root && sendBuf == nullptr) {     // The sendBuf of the root node cannot be null.
        return HCCL_E_PTR;
    }
    ```

- Query the number of ranks in the communicator.

    ```c
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    HcclResult ret = HcclGetRankSize(comm, &rankSize);
    if (userRank >= rankSize) {     // rank_id is out of range.
        return HCCL_E_PARA;
    }
    ```

- Query the link information between the local device and device 0 in the server.

    ```c
    u32 dstRank = 0;
    u32 srcRank = rank;
    CommLink *linkList = nullptr;
    u32 listSize = 0;
    HcclResult ret = HcclRankGraphGetLinks(comm, 0, srcRank, dstRank, &linkList, &listSize);
    for (u32 i = 0; i < listSize; ++i) {
        CommLink& currentLink = linkList[i]; // Enumerate all link objects.
        if (currentLink.linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_HCCS) {
            // Determine and process HCCS links.
        }
    }
    ```

- Query the number of SuperPoDs contained in the communicator where the local device resides.

    ```c
    u32 *netlayers = nullptr;
    u32 netLayersNum = 0;
    HcclResult ret = HcclRankGraphGetLayers(comm, &netlayers, &netLayersNum); // Obtain the communication network layers contained in the communicator.
    u32 superPodNum;
    u32 level1RankListNum = 0;
    u32 *level1SizeList = nullptr;
    if (netLayersNum == 3) { // SuperPoD scenario
        ret = HcclRankGraphGetInstSizeListByLayer(comm, 1, &level1SizeList, &level1RankListNum);
        superPodNum = level1RankListNum;
    }
    ```

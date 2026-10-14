# Querying Topology Information

<!-- md-trans-meta sourceCommit=b3aeada8ad178e55bc5a83b8ba51324990d4c8f3 translatedAt=2026-09-24T07:04:53.165Z pushedAt=2026-09-28T05:59:29.453Z -->

## Background

To handle complex network topology structures, communication operators need to select the most suitable algorithm based on the topology structure of the communicator. To this end, the HCCL control plane provides the topology information query function for developers.

## Topology Information

The following table lists the topology information that can be queried through HCCL control plane APIs.

| Topology Information | Query API |
| --- | --- |
| Obtains the rank ID corresponding to a device in a specified communicator. | [HcclGetRankId](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclGetRankId.md) |
| Queries the number of ranks in a specified communicator. | [HcclGetRankSize](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclGetRankSize.md) |
| Queries the list of topology hierarchy numbers that contain the current rank and the number of topology hierarchies. | [HcclRankGraphGetLayers](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetLayers.md) |
| Given a communicator and a topology hierarchy number, returns the list of all rank IDs and the number of ranks in the topology instance where the current rank resides at this hierarchy. | [HcclRankGraphGetRanksByLayer](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetRanksByLayer.md) |
| Given a communicator and a topology hierarchy number, returns the number of ranks in the topology instance where the current rank resides at this hierarchy. | [HcclRankGraphGetRankSizeByLayer](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetRankSizeByLayer.md) |
| Given a communicator and a topology hierarchy number, returns the topology type of the hierarchy where the current rank resides. | [HcclRankGraphGetTopoTypeByLayer](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetTopoTypeByLayer.md) |
| Given a communicator and a topology hierarchy number, queries the number of topology instances at this hierarchy and the number of ranks contained in each instance. | [HcclRankGraphGetInstSizeListByLayer](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetInstSizeListByLayer.md) |
| Given a communicator and a topology hierarchy number, queries the communication link information between the source rank and the destination rank. | [HcclRankGraphGetLinks](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetLinks.md) |

- Queries the ID of the current rank in the communicator.

    ```c
    u32 userRank = INVALID_VALUE_RANKID;
    HcclResult ret = HcclGetRankId(comm, &userRank);
    if (userRank == root && sendBuf == nullptr) {     // sendBuf of the root node cannot be null.
        return HCCL_E_PTR;
    }
    ```

- Query the number of ranks in the communicator.

    ```c
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    HcclResult ret = HcclGetRankSize(comm, &rankSize);
    if (userRank >= rankSize) {     // rank_id out of range
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
            // Determine and process the HCCS link.
        }
    }
    ```

- Query the number of SuperPoDs in the communicator where the local device resides.

    ```c
    u32 *netlayers = nullptr;
    u32 netLayersNum = 0;
    HcclResult ret = HcclRankGraphGetLayers(comm, &netlayers, &netLayersNum); // Obtain the communication network layers in the communicator.
    u32 superPodNum;
    u32 level1RankListNum = 0;
    u32 *level1SizeList = nullptr;
    if (netLayersNum == 3) { // SuperPoD scenario
        ret = HcclRankGraphGetInstSizeListByLayer(comm, 1, &level1SizeList, &level1RankListNum);
        superPodNum = level1RankListNum;
    }
    ```

## Sample Code

```c
// Take a Mesh connection of a single server with 4 devices as an example.
struct TopoInfo {
    uint32_t rankId; // Used to uniquely identify a rank.
    uint32_t rankSize; // Number of ranks participating in this collective communication.
    std::vector<u32> rankList; // Set of rank IDs participating in this collective communication.
    CommTopo topoType; // Link connection type: COMM_TOPO_1DMESH/COMM_TOPO_CLOS, etc.
    std::vector<HcclChannelDesc> channels; // Link information between the local rank and other devices.
};
HcclResult FillSimpleTopoInfo(HcclComm comm, TopoInfo &topoInfo){
    HcclResult ret = HcclGetRankId(comm, &topoInfo.rankId);
    // Verify the ret result.
    ret = HcclGetRankSize(comm, &topoInfo.rankSize);
    // Verify the ret result.
    uint32_t *ranks = nullptr;
    uint32_t rankNum = 0;
    // Since this is a single server with four devices, netLayer=0, and the ranks obtained under netLayer 0 are the set of rank IDs for this collective communication.
    ret = HcclRankGraphGetRanksByLayer(comm, 0, &ranks, &rankNum);
    // Verify the ret result.
    for (size_t index = 0; index < rankNum; index++) {
        topoInfo.rankList.push_back(ranks[index]);
    }
    ret = HcclRankGraphGetTopoTypeByLayer(comm, 0, &topoInfo.topoType);
    // Verify the ret result.
    // Calculate the required channels.
    for (auto remoteRankId : topoInfo.rankList) {
       if (remoteRankId == topoInfo.rankId) {continue;}
       CommLink *linkList = nullptr;
       u32 listSize = 0;
       uint32_t netLayer = 0;
       ret = HcclRankGraphGetLinks(comm, netLayer, topoInfo.rankId, remoteRankId, &linkList, &listSize);
       // Verify the ret result.
       for (uint32_t idx = 0; idx < listSize; idx++) {
           HcclChannelDesc channelDesc;
           HcclChannelDescInit(&channelDesc, 1);
           channelDesc.remoteRank = remoteRankId;
           CommLink link = linkList[idx];
           channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
           channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
           channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
           channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
           channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
           channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
           channelDesc.channelProtocol = link.linkAttr.linkProtocol;
           channelDesc.notifyNum = 3;
           topoInfo.channels.push_back(channelDesc);
       }
    }
    return ret;    
}
```

# hccl.manage.api

<!-- md-trans-meta sourceCommit=7494246995d6d252dbb49e1c6304dffdf77e6d53 translatedAt=2026-09-28T06:46:14.172Z pushedAt=2026-09-29T02:58:41.798Z -->

The **hccl.manage.api** module provides collective communication group management APIs, including group creation and destruction and rank information query. All APIs in this module must be called after collective communication initialization is complete.

## API List

- [create_group](create_group.md): Creates a collective communication group with the group name.
- [destroy_group](destroy_group.md): Destroys a group.
- [get_rank_size](get_rank_size.md): Gets the number of ranks in a group.
- [get_rank_id](get_rank_id.md): Gets the rank ID corresponding to the device in a group.
- [get_local_rank_size](get_local_rank_size.md): Gets the number of local ranks on the server where the device in the group resides.
- [get_local_rank_id](get_local_rank_id.md): Gets the local rank ID on the server where the device in the group resides.
- [get_world_rank_from_group_rank](get_world_rank_from_group_rank.md): Gets the world rank ID based on the rank ID of the process in the group.
- [get_group_rank_from_world_rank](get_group_rank_from_world_rank.md): Gets the group rank ID of the process in the group based on the world rank ID.

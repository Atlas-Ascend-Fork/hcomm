# hccl.split.api

<!-- md-trans-meta sourceCommit=7494246995d6d252dbb49e1c6304dffdf77e6d53 translatedAt=2026-09-28T06:47:13.552Z pushedAt=2026-09-29T02:59:52.673Z -->

The **hccl.split.api** module provides APIs for setting the reverse gradient split strategy, which are used for performance tuning of allreduce fusion. All APIs in this module must be called after collective communication initialization is complete.

## API List

- [set_split_strategy_by_idx](set_split_strategy_by_idx.md): Sets the reverse gradient split strategy in a collective communication group based on the gradient index ID.
- [set_split_strategy_by_size](set_split_strategy_by_size.md): Sets the reverse gradient split strategy in a collective communication group based on the gradient data size percentage.

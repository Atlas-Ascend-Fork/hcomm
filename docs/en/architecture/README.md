# HCCL Architecture Documentation

This document describes the software architecture of HCCL (Huawei Collective Communication Library).

## Document Contents

- [System Overview](./overview.md) - Overall system architecture and module division
- [base_comm](./base_comm/) - Basic communication module
- [coll_comm](./coll_comm/) - Collective communication domain management module

## Architecture Principles

1. **Modular Design** - Each module has a single responsibility, and modules are decoupled
2. **Platform Abstraction** - Shield the underlying hardware differences and provide unified interfaces
3. **Extensibility** - Support flexible extension of new algorithms and new communication primitives

## Related Links

- [Contribution Guide](../../../CONTRIBUTING.md)
- [RFC Documents](../rfcs/)
- [Developer Guide](../comm_op_dev_guide/)

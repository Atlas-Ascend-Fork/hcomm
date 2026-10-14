# CCU Programming Model and Concepts

<!-- md-trans-meta sourceCommit=cca54d1cd891f10b71bb835d788da440620115a5 translatedAt=2026-09-24T07:08:46.753Z pushedAt=2026-09-28T05:59:29.468Z -->

## CCU Architecture

### System Architecture

The Collective Communication Unit (CCU) is a dedicated collective communication coprocessor in the Ascend NPU, located on the IO Die. The following figure shows the position of the CCU in the Ascend 950PR&950DT products:

![](figures/ccu_in_950.png)

### Basic Concepts

The CCU contains multiple key components that work together to accomplish collective communication tasks:

![](figures/ccu_arch.png)

- **On-chip cache**: An on-chip data cache that uses 4 KB slices as the basic operation unit and supports on-chip reduction operations across multiple slices.
- **Registers**: classified into general-purpose registers, synchronization registers, and address registers.
  - **General-purpose registers**: used to store parameters or data during CCU execution, supporting simple value assignment and addition operations.
  - **Synchronization registers**: used to implement semaphore-like synchronization operations, supporting the Wait or Set operation.
  - **Address registers**: Used to store memory addresses for communication operations. They will be deprecated in the future and replaced by general-purpose registers.

- **Concurrent execution engine**: Provides the CCU with the capability to execute instructions concurrently and in loops.
- **Instruction space**: Stores the instruction sequences executed by the CCU.
- **Channel table**: Stores multiple channel table entries, each containing context information for UB communication.

### Typical CCU Usage and Advantages

In large-scale distributed training and inference services, collective communication performance is one of the key bottlenecks affecting overall system performance. In traditional communication approaches, collective communication relies on compute units such as AI CPUs and AI Cores to construct communication task descriptors through the software protocol stack, which then drives the hardware to execute communication tasks. This execution method consumes compute cores or compute power, and the scheduling overhead of the software protocol stack is significant, resulting in high overall communication latency.

To address the above issue, the NPU introduces the **Collective Communication Unit (CCU)** as a dedicated processor to accelerate collective communication tasks. The CCU can receive collective communication tasks dispatched by the NPU scheduler, execute communication algorithm instructions, and complete cross-NPU synchronization, address exchange, data transfer, and reduction operations, thereby delivering full collective communication capabilities.

A typical CCU usage pattern is to use on-chip cache for data relay, thereby reducing memory access, as shown in the following figure.

![](figures/ccu_typical_usage.png)

- **Reduction operation**: Reads data from the remote end to the on-chip cache, performs reduction with local memory data, and finally writes data back to the local memory.

    Without the on-chip cache, data reduction on each peer end requires one read and one write of local data. Combined with the reads of local data by each peer end, this results in a total of 2\(n-1\) reads and n-1 writes. With the on-chip cache, only one read of local data and one write-back after reduction are needed. Combined with the reads of local data by each peer end, this results in a total of n reads and 1 write.

- **Broadcast operation**: Similarly, uses the on-chip cache to reduce the number of memory accesses.

    Similarly, the broadcast operation can reduce n-1 reads and n-1 writes to 1 read and n-1 writes.

**CCU advantages**:

1. Saving of memory access bandwidth

    In distributed training and inference services, memory access bandwidth is a performance bottleneck. By leveraging the data movement characteristics of collective communication operators and built-in independent on-chip cache, the CCU reduces the memory access requirements of communication operators such as AllReduce by approximately an order of magnitude, freeing up more memory access bandwidth for compute operators and thereby improving the concurrency performance of communication and computation.

2. Promoted-precision reduction and determinism

    Reduction operations in collective communication suffer from precision loss and order indeterminacy. The CCU uses independent on-chip MS and a promoted-precision reduction unit to ensure that the addition order and floating-point truncation errors are controllable, effectively improving the precision and determinism of reduction operations.

3. Low-latency communication without occupying compute cores

    Traditional communication methods rely on AI CPU/AI Vector cores, occupying compute resources. By using hardware acceleration for the construction and dispatch of communication task descriptors, the CCU reduces communication latency without consuming compute power, providing independent hardware support for low-latency services.

## Resource Abstraction

| Resource Type | Description | Corresponding Resource |
| --- | --- | --- |
| `ccu::Variable` | Variable | General-purpose register |
| `ccu::Address` | Address | Address register |
| `ccu::Event` | Event | Synchronization register |
| `ccu::CcuBuffer` | On-chip cache slice, 4 KB in size | CCU on-chip cache |
| `ccu::LocalAddr` | Local address, used for communication operations | Contains an address and a token, stored by Address and Variable respectively. |
| `ccu::RemoteAddr` | Remote address, used for communication operations | Contains an address and a token, stored by Address and Variable respectively. |

### Creating Resources

- Single resource creation: Use the default constructor to create a resource directly.
- Batch contiguous resource creation (Loop and LoopGroup operations require resource contiguity):

  Use **ccu::Array** to create resources. Note that using native arrays to create resources (for example, **Variable vars\[10\]**) does not guarantee resource contiguity.

> [!NOTE] Note
>
> - CCU concurrent operations require resource contiguity.
> - Using native arrays to create resources (for example, **Variable vars\[10\]**) does not guarantee resource contiguity.

### Usage Example

```c
// Single resource
ccu::Variable var; 
ccu::CcuBuffer buf;  
// Allocate resources in batches.
ccu::Array<ccu::Variable> vars(10); 
ccu::Array<ccu::Event> events(5); 
```

## Data Movement Capabilities

As shown in the figure, the CCU provides the following data movement capabilities:

![](figures/ccu_datacopy.png)

- Local memory \<-\> Variable/Address registers
  - Load data from local memory to registers.
  - Save register content to local memory.

- Local memory \<-\> CcuBuf
  - Read data from local memory to CcuBuf.
  - Write CcuBuf data to local memory.

- CcuBuf \<-\> Remote memory
  - Read data from remote memory to CcuBuf.
  - Write CcuBuf data to remote memory.

- Local memory \<-\> Remote memory
  - Read remote memory.
  - Write remote memory.

- Write remote registers.
  - Write remote synchronization registers.
  - Write local variable data to the remote variable.

## Compute Capabilities

### General-Purpose Registers/Address Registers

- Immediate value assignment

    ```c
    // Assign a value to the general-purpose register.
    Variable x; x = 5;  
    // Assign a value to the address register.
    uint64_t ptr; 
    // ... 
    Address addr; 
    addr = ptr; 
    ```

- Variable assignment

    ```c
    Variable x, y;
    // ... 
    Variable v; 
    v = x;  
    Address addr; 
    addr = y; 
    ```

- Addition

    ```c
    Variable x=5, y=3, z;  
    z = x+y;  
    Address addr1, addr2;
    // ... 
    addr1 += x; 
    addr2 = addr1 + x; 
    ```

## CcuBuffer

Sequential reduction computation is supported for up to eight CcuBuffers, with the input type being the same as the output type. Supported reduction operations: ADD, MAX, and MIN. Supported data types: FP32, FP16, BF16, INT32, INT16, INT8, and UINT8.

For the ADD operation on FP16/BF16 data, to avoid precision loss, the CCU promotes the precision to FP32 for reduction computation, as shown in the following figure.

![](figures/ccubuffer.png)

## Concurrency Model

- Loop: basic concurrency unit in a CCU task.
  - Operations within a loop are executed serially, while operations in different loops are executed in parallel.
  - A loop can cyclically execute internal operations, working with CcuBuffer to cyclically move data.
  - Loop parameters: number of loop repeats and the incremental offset of the memory address for each loop.
  - When using CcuBuffer for data movement, the amount of data moved in a single operation is limited due to the buffer size. Using multiple concurrent loops can fully utilize the link bandwidth.

    ![](figures/ccu_loop.png)

- LoopGroup: Replicates a group of loops to multiple copies to increase the number of concurrent loops.
  - A loop cannot be executed independently and must be placed within a LoopGroup.
  - For loops within a LoopGroup, the LoopGroup can specify the starting loop from which replication begins and the number of replications.
  - LoopGroup needs to specify the resource ID offset used by each group of loops after replication.

    ![](figures/ccu_loopgroup.png)

- Usage Example

  ```c
  ccu::Func func([](){
      Write(chann, remoteMem, localCcuBuf, 4096);
  }); 
  ccu::Loop loop({4096, 10}, func);  // Execute the loop 10 times, with an address increment offset of 4096B each time. 
  
  // For each loop in the LoopGroup, from the 0th loop to the last one, replicate it 5 times.
  ccu::LoopGroup loopGroup({5, 0, 4096*10, 10, 10}, {loop});
  ```

## Synchronization Mechanism

The CCU uses synchronization registers to implement synchronization operations:

- Wait operation: Waits for the register bit corresponding to the mask to be set before returning.
- Record operation: Sets the register bit corresponding to the mask.

Synchronization registers are used in two scenarios:

- Local operations on a single device, abstracted as events.
- Communication operations between different devices, abstracted as Notify resources.

### Local Events

Local events are used in two scenarios:

- An event object can be passed to an asynchronous operation, and **EventWait** can be used to wait for the asynchronous operation to complete.
- Different CCU tasks can use **EventRecord** and **EventWait** to achieve synchronization through event objects.

Example:

```c
ccu::Event evt;  

ccu::EventRecord(evt, 0x01); 

ccu::EventWait(evt, 0x01); 
```

### Remote Notification

After two communication entities establish a channel: (1) The local end can write data to the peer synchronization register based on the Notify sequence number in the channel; (2) The local end can wait for a notification signal from the peer end based on the Notify sequence number in the channel.

| API | Description |
| --- | --- |
| NotifyRecord(ch, notifyIdx, mask) | Sends a notification to the remote end. |
| NotifyWait(ch, notifyIdx, mask) | Waits for a notification from the remote end. |

Example:

```c
// Send a notification to the remote end.
ccu::NotifyRecord(channelHandle, 0, 0x12);  
// Wait for the notification from the remote end.
ccu::NotifyWait(channelHandle, 0, 0x12);
```

## Flow Control

- Conditional branching

  ```c
  CCU_IF(counter != limit) {
       // then-block} 
  CCU_ELSE {
       // else-block
  } 
  ```

- Loops

  ```c
  CCU_WHILE(counter != limit) {
       accumulator = accumulator + step;
       counter = counter + one; 
  } 
  ```

- Function blocks

  | Type/API | Description |
  | --- | --- |
  | `ccu::Func` | Defines a function block. |
  | `ccu::CallFunc<func>(...)` | Calls a function block. |

- Usage example

  ```c
  // Define the function block.
  ccu::Func myFunc([](Variable x, Variable y){
     ...
  });  
  // Call the function block.
  Variable arg1, arg2;
  ...
  ccu::CallFunc<myFunc>(arg1, arg2); 
  ```

## Constraints

### Parameter Loading

- A CCU task descriptor supports a maximum of 13 64-bit parameters.
- The parameter loading instruction **LoadArg** must be called at the very beginning of the program.

### Resource Creation

Once created, resources do not support dynamic reclamation. For example, for a resource defined in a local scope, such as a **Variable**, the **Variable** object itself is destroyed when the local scope ends, but the corresponding register resources are not reclaimed.

### Data Movement

- The length of a data movement operation cannot be 0. Otherwise, the hardware behavior is unpredictable.
- The length of data moved from local memory to remote memory must be less than 256 MB.
- The length of a data movement operation involving CcuBuffer must be less than or equal to 4096 bytes.
- Reading data from or writing data to the CcuBuffer of another device is not supported.

### Concurrent Operations

- Only data movement instructions and **Wait** instructions can be placed in a loop. Other instructions are not supported.
- The number of loop repeats must be greater than 0.
- When creating a loop and LoopGroup, if an immediate value is used as a parameter, the created loop and LoopGroup consume **Variable** resources to store the configuration parameters.
- Before creating a loop and LoopGroup, compute the required resources and allocate them in advance to avoid out-of-bound access.

A LoopGroup can replicate a group multiple times for concurrent execution, and the resources used by each loop must not conflict. Therefore, in addition to specifying the number of replications, a LoopGroup also needs to specify the incremental offset of the register ID used by the replicated loops. As shown in the following figure, take loop S as an example. Loop S is manually written by the developer, and the LoopGroup replicates it *N* times. The CcuBuffer used by the replicated loop is incrementally offset from the base CcuBuffer. Loop S uses CcuBuffer x, and the LoopGroup specifies the CcuBuffer offset parameter as offset. Then the *N*th loop replicated from loop S uses CcuBuffer x + *N* × offset. During code development, only CcuBuffer x can be explicitly perceived. To prevent the resources used by the replicated loops from going out of bounds, you need to plan resource usage and allocate resources in advance.

### Instruction Deployment

The CCU is located on the IO die of a device. A device may contain multiple IO dies, and different IO dies contain different network devices. In actual networking, different network devices may be interconnected with different other devices. A CCU on an IO die cannot use the network device on another IO die for communication.

Therefore, when developing a program that runs on a CCU, you must ensure that all network devices it uses are on the same die. The CCU translator deduces the target CCU device on which the translated instruction sequence is deployed based on the network devices it uses.

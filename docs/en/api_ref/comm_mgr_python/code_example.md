# Sample Code

<!-- md-trans-meta sourceCommit=7494246995d6d252dbb49e1c6304dffdf77e6d53 translatedAt=2026-09-28T06:42:33.881Z pushedAt=2026-09-29T02:33:47.803Z -->

This code example targets a TensorFlow 1.15 network and uses the default global communicator for communication.

Assume that the code file is named **hccl_test.py**.

```python
import tensorflow as tf
import sys
import os
import numpy as np
import time
import argparse
from npu_bridge.npu_init import *
from hccl.manage.api import get_rank_size, get_rank_id

def tensor_type(list1, type):
    tensor1=[]
    tensor1 = tf.Variable(list1, dtype=tf.int64)
    return tensor1

def numpy_type(type):
    input_type = np.int64
    return input_type

def hccl_operator(rank_id, root_rank, rank_size,  group, dtype, data):
    tensors={}

    # allreduce
    list_1=['sum','max','min','prod']
    for i in range(len(list_1)):  
        exec('list_1=["sum","max","min","prod"]')
        exec('element_list'+str(i)+'=[1 for i in range(data)]')
        exec('tensor_'+str(i)+'= tensor_type(element_list'+str(i)+', dtype)')
        exec('tensor_tmp'+str(i)+'= tf.add(tensor_'+str(i)+', rank_id + 1)')
        exec('new_tensor'+str(i)+'= tf.reshape(tensor_tmp'+str(i)+', [rank_size, -1])')
        exec('tensors[\'allreduce_'+list_1[i]+'\'] = hccl_ops.allreduce(new_tensor'+str(i)+','+'\"'+list_1[i]+'\"'+', group=group)')

    # broadcast
    exec('list_test = np.ones((1,data))')
    exec('tensor_test = tensor_type(list_test, dtype)')
    exec('tensor_z = tf.add(tensor_test, rank_id + 1)')
    exec('new_tensor10 = tf.reshape(tensor_z, [rank_size, -1])')
    exec('test_list1=[new_tensor10]')
    exec('tensors[\'broadcast\'] = hccl_ops.broadcast(test_list1, root_rank, group=group)')

    # allgather
    exec('tensors[\'gather_tensor\'] = hccl_ops.allgather(new_tensor'+str(1)+', rank_size, group=group)')    

    # reducescatter
    for i in range(len(list_1)):  
        exec('list_1=["sum","max","min","prod"]')
        exec('element_list'+str(i+5)+'=[1 for i in range(data)]')
        exec('tensor_'+str(i+5)+'= tensor_type(element_list'+str(i+5)+', dtype)')
        exec('tensor_tmp'+str(i+5)+'= tf.add(tensor_'+str(i+5)+', rank_id + 1)')
        exec('new_tensor'+str(i+5)+'= tf.reshape(tensor_tmp'+str(i+5)+', [rank_size, -1])')
        exec('tensors[\'reducescatter_'+list_1[i]+'\'] = hccl_ops.reduce_scatter(new_tensor'+str(i+5)+','+'\"'+list_1[i]+'\"'+', '+'rank_size, group=group)')

    # reduce
    for i in range(len(list_1)):  
        exec('list_1=["sum","max","min","prod"]')
        exec('element_list'+str(i+10)+'=[1 for i in range(data)]')
        exec('tensor_'+str(i+10)+'= tensor_type(element_list'+str(i+10)+', dtype)')
        exec('tensor_tmp'+str(i+10)+'= tf.add(tensor_'+str(i+10)+', rank_id + 1)')
        exec('new_tensor'+str(i+10)+'= tf.reshape(tensor_tmp'+str(i+10)+', [rank_size, -1])')
        exec('tensors[\'reduce_'+list_1[i]+'\'] = hccl_ops.reduce(new_tensor'+str(i+10)+','+'\"'+list_1[i]+'\"'+', '+'root_rank, group=group)')

    input_type = numpy_type(dtype)
    data1_shape = data*rank_size + (rank_size-1)*rank_size
    data1_ = np.arange(1,data1_shape+1).astype(input_type)

    check_data_shape = (data + rank_id) * rank_size
    check_data_ = np.arange(1,check_data_shape+1).astype(input_type)

    send_data = tf.Variable(data1_)
    check_data = tf.Variable(check_data_)
    send_counts_list = [data+i for i in range(rank_size)]
    send_counts = tf.constant(send_counts_list,dtype=tf.int64)
    send_displacements = tf.constant([rank_id*(data+i) for i in range(rank_size)],dtype=tf.int64)

    # Static-shape recv_counts and recv_displacements must use tf.constant.
    recv_counts = tf.constant([rank_id+data for _ in range(rank_size)],dtype=tf.int64)     
    recv_displacements = tf.constant([(rank_id+data)*i for i in range(rank_size)],dtype=tf.int64)    

    all_to_all_v = hccl_ops.all_to_all_v(send_data,send_counts,send_displacements,recv_counts,recv_displacements,group=group)
    tensors['alltoallv_tensor'] = all_to_all_v
    tensors['check_tensors'] = check_data    
    return tensors

def main():
    config = {}
    hccl_session_config = tf.ConfigProto() 
    custom_op =  hccl_session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    npu_init = npu_ops.initialize_system()
    npu_shutdown = npu_ops.shutdown_system()
    with tf.Session(config=hccl_session_config) as sess:
        # Initialize collective communication.
        sess.run(npu_init)
        # Obtain the number of ranks in the group.
        config['rank_size'] = get_rank_size()
        # Obtain the rank ID of the device in the group.
        config['rank_id'] = get_rank_id()
        try:
            # Dispatch the collective communication operator.
            tensors = hccl_operator(config['rank_id'], 0, config['rank_size'], "hccl_world_group",  "float32", 1024)
            # Initialize the global variables of the tf framework.
            init_var = tf.global_variables_initializer()
            sess.run(init_var)
            # Perform training. This is only an example.
            v = sess.run(tensors)
            tf.logging.info(v)

        except Exception as e:
            print('ERROR : %s'  % e)
            print('train fail')
        else:
            print('train success')
        # Close the session.
        sess.run(npu_shutdown)

if __name__ == '__main__':
    # Enable logging.
    tf.logging.set_verbosity(tf.logging.INFO)
    # Execute the main function.
    main()
```

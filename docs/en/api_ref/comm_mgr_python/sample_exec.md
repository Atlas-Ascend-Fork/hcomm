# Sample Execution

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-09-28T06:48:29.945Z pushedAt=2026-09-29T03:10:48.606Z -->

This section uses a single-server 8-device network with resource information configured through a rank table file as an example to describe how to run the sample code in [Code Example](code_example.md).

1. Prepare the rank table file.

    For details about how to configure the rank table, see [Cluster Information Configuration](https://gitcode.com/cann/hccl/blob/9.3.0-beta.1/docs/en/user_guide/cluster_info_config/README.md). Here, **rank_table.json** is used as an example name.

2. Construct the startup script.

    Assume the script is named **hccl_start_8p.sh**. An example is as follows:

    ```bash
    # Configure the CANN software environment variables (using the root user as an example):
    source /usr/local/Ascend/cann/set_env.sh
    # TF Adapter Python library, where ${TFPLUGIN_INSTALL_PATH} is the installation path of the TF Adapter software package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    export RANK_SIZE=8
    export RANK_TABLE_FILE=/home/test/rank_table.json    # Path of the rank table resource configuration file. Replace it with the actual path.
    export JOB_ID=10087      # User-defined job ID, which can contain uppercase and lowercase letters, digits, hyphens, or underscores.

    for((RANK_ID=0;RANK_ID&lt;$((RANK_SIZE));RANK_ID++));
    do
        export RANK_ID=$RANK_ID
        export ASCEND_DEVICE_ID=$RANK_ID
        # Execute the script. Replace the script path and name with the actual ones.
        nohup python3 /home/test/hccl_test.py &
    done
    ```

3. Execute the startup script.

    ```bash
    bash hccl_start_8p.sh
    ```

    The result is as follows:

    ```text
    ... ...
    'reduce_sum': array([[ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ...,  0,  0,  0],
           ...,
           [ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ..., 44, 44, 44]]), 'reduce_max': array([[4097, 4098, 4099, ..., 4222, 4223, 4224],
           [4225, 4226, 4227, ..., 4350, 4351, 4352],
           [4353, 4354, 4355, ..., 4478, 4479, 4480],
           ...,
           [4737, 4738, 4739, ..., 4862, 4863, 4864],
           [4865, 4866, 4867, ..., 4990, 4991, 4992],
           [4993, 4994, 4995, ...,    9,    9,    9]]), 'reduce_min': array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 2, 2, 2]]), 'reduce_prod': array([[     0,      0,      0, ...,      0,      0,      0],
           [     0,      0,      0, ...,      0,      0,      0],
           [     0,      0,      0, ...,      0,      0,      0],
           ...,
           [     0,      0,      0, ...,      0,      0,      0],
           [     0,      0,      0, ...,      0,      0,      0],
           [     0,      0,      0, ..., 362880, 362880, 362880]]), 'alltoallv_tensor': array([   1,    2,    3, ..., 8246, 8247, 8248]), 'check_tensors': array([   1,    2,    3, ..., 8246, 8247, 8248])
    train success
    ```

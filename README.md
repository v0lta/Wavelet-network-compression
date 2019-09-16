# waveletnet

# image net training:
```CUDA_VISIBLE_DEVICES=0,1 ipython github_imnet_pytorch.py -- -a alexnet --lr 0.01 /home/wolter/waveletnet/data_sets/image_net ```
```CUDA_VISIBLE_DEVICES=0,1 ipython github_imnet_pytorch.py -- -a alexnet --lr 0.01  --dist-url 'tcp://127.0.0.1:6116' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /home/wolter/waveletnet/data_sets/image_net ```
```CUDA_VISIBLE_DEVICES=0,1 ipython github_imnet_pytorch.py -- -a alexnet --lr 0.01  --workers 8  -b 512 --dist-url 'tcp://127.0.0.1:6116' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /home/wolter/waveletnet/data_sets/image_net```
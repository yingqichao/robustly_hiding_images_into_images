# train
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --master_port 3040 --nproc_per_node=1 train.py \
                                 -opt options/train/train_IRN+_x4.yml -val 0 -images_in 3 --launcher pytorch

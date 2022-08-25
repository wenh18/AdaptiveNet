#python classification_train.py --dataset imagenet --model mobilenetv2_100 -b 160 --sched step --epochs 450 --decay-epochs 1 --decay-rate .89 --opt adam --opt-eps .001 -j 16 --warmup-lr 8e-3 --weight-decay 1e-5 --drop 0.3 \
#--drop-connect 0.2 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr 7e-3 --num-classes 1000 --data_dir ../project1/MultiBranchNet/mobilenet-yolov4-pytorch/pytorch-image-models-master/data/imagenet/ \
#--original_model pretrainedweight/mobilenetv2_100_ra-b33bc2c4.pth --stage 0 --savemodelfreq 1 --start-epoch 0 --prune --device_id 1 --outputdir output/supernet/

python classification_train.py --dataset imagenet --model resnet50 --amp --batch 64 --sched step --lr 3.9e-4 --warmup-lr 4.3e-4 --epochs 600 --weight-decay 0.01 \
 --train-interpolation bicubic --crop-pct 0.95 --smoothing 0.1 --warmup-epochs 2 --aa rand-m7-n3-mstd1.0-inc1 --seed 0 --opt adamp --drop-path 0.05 --drop 0.1 \
  --reprob 0.35 --mixup .2 --cutmix 1.0 --bce-loss --data_dir ../project1/MultiBranchNet/mobilenet-yolov4-pytorch/pytorch-image-models-master/data/imagenet/ \
  -j 8 --original_model pretrainedweight/resnet50_a1_0-14fe96d1.pth --stage 0 --pretrain_epochs 60 --decay-epochs 1.5 --decay-rate .875 # GPU0

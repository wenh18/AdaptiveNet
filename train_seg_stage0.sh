python train_resdet.py --dataset imagenetimagenet --model resnet50 --amp --batch 192 --sched step \
 --lr 3.5e-4 --warmup-lr 4.3e-4 --epochs 600 --weight-decay 0.01 --train-interpolation bicubic \
  --crop-pct 0.95 --smoothing 0.1 --warmup-epochs 2 --aa rand-m7-n3-mstd1.0-inc1 --seed 0 --opt adamp \
   --drop-path 0.05 --drop 0.1 --reprob 0.35 --mixup .2 --cutmix 1.0 --bce-loss  \
   --data_dir ../../project1/MultiBranchNet/mobilenet-yolov4-pytorch/pytorch-image-models-master/data/imagenet/ \
   -j 8 --original_model pretrainedweight/multiclass.pth --stage 0 \
   --pretrain_epochs 60 --decay-epochs 1 --decay-rate .93 # GPU0

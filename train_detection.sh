#bash distributed_train.sh 2 --dataset mscoco --model tf_efficientdet_d0 -b 16 --amp --lr .09 --warmup-epochs 5 --sync-bn --opt fusedmomentum --model-ema
python train_detection.py --dataset coco2017 --model resdet50 -b 24 --amp --lr 1.1e-5 --warmup-lr 1.3e-5 --warmup-epochs 1 --sync-bn --opt adam --sched step \
--root /home/data/datasets/coco --backbonepath /home/wenh/Desktop/efficientdet-pytorch-master/pretrainedweights/resnet_0epoch16.pth \
 --headpath /home/wenh/Desktop/efficientdet-pytorch-master/pretrainedweights/resdet50_416-08676892.pth --decay-epochs 1 --decay-rate 0.985 \
--stage 1 --distilledmodel /home/wenh/Desktop/efficientdet-pytorch-master/multisubnet/distilled_model/epoch21_max_0367.pth --device_id 1
# --distillhead --stage 0 -b 22 --amp --lr 9e-5 --warmup-lr 1e-4
#./distributed_train.sh 2 --dataset coco2017 --model resdet50 -b 16 --amp --lr 1e-4 --warmup-lr 5e-4 --warmup-epochs 1 --sync-bn --opt adam --root ../datasets/coco
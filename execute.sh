python3 ResNeStGPU.py --dataset imagenet --model resnest50 --lr-scheduler cos --epochs 100 --lr 0.001 --batch-size 32 --label-smoothing 0.1 --last-gamma --no-bn-wd --rectify --FiveFold
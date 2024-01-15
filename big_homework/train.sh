transformer="transformer"
#length=96
length=336
mode_='train'
seed=0
exp="exp/transformer/model.${length}.${seed}.pt"
pic_path="train.${length}.${seed}/"

nohup python -u run.py \
    --mode $mode_ \
    --epochs 500 \
    --model-name=$transformer \
    --save-path=$exp \
    --length $length \
    --save-pic-path $pic_path \
    --device 4 \
    --batch-size 32 \
    --seed $seed \
    > train.$length.last.$seed.log 2>&1 &


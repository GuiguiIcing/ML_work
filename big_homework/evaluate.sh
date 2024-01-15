transformer="transformer"
#length=336
length=96
seed=24
pic_path="eval.${length}.${seed}/"
mode_='evaluate'
path="exp/transformer/model.${length}.${seed}.pt"

nohup python -u run.py \
    --mode=$mode_ \
    --model-name=$transformer \
    --length $length \
    --save-path $path \
    --device 4 \
    --save-pic-path $pic_path \
    --batch-size 32 \
    > eval.$length.$seed.log 2>&1 &

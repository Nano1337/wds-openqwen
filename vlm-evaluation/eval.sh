DATAPATH=$1
CKPT_PATH=$2
CKPTID=$3

# supported tasks: gqa vqa-v2 vizwiz okvqa ai2d text-vqa pope mmmu mmbench seedbench-image mathvista mmstar mantis mmlu

for task in ai2d text-vqa pope mmmu mmbench seedbench mmstar mathvista ; do
    subset="full"
    # rm ${DATAPATH}/vlm-evaluation/datasets/${task}/*
    # python scripts/datasets/prepare.py --dataset_family ${task} --root_dir ${DATAPATH}/vlm-evaluation/ --shots 0
    
    # rm -r results/${task}/${task}-${subset}/

    accelerate launch --main_process_port 29511 --num_processes=8 vlm-evaluation/scripts/evaluate.py --model_dir $CKPT_PATH --dataset.type ${task}-${subset} --dataset.root_dir ${DATAPATH}/vlm-evaluation/ --results_dir ./results --model_id $CKPTID

    python vlm-evaluation/scripts/score.py --model_id $CKPTID --dataset.type ${task}-${subset} --dataset.root_dir ${DATAPATH}/vlm-evaluation/ --results_dir ./results
done

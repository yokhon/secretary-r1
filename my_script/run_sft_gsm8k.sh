set -x

CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 torchrun --nnodes=1 --nproc_per_node=4 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/mnt/sdb/hanxu/projects/secretary-r1/data/sft_gsm8k_v2/train.parquet \
    data.val_files=/mnt/sdb/hanxu/projects/secretary-r1/data/sft_gsm8k_v2/test.parquet \
    data.prompt_key=question \
    data.response_key=formatted_answer \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=/data2/share/Qwen3/Qwen3-4B-Base \
    trainer.default_local_dir=/data2/share/hanxu/verl/checkpoints/agent-omni/gsm8k/sft-Qwen3-4B-Base \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-Qwen3-4B-Base \
    trainer.total_epochs=8 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null
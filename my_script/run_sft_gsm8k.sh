set -x

CUDA_VISIBLE_DEVICES=6,7 PYTHONUNBUFFERED=1 torchrun --nnodes=1 --nproc_per_node=2 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/mnt/sdb/hanxu/projects/secretary-r1/data/sft_gsm8k_v2/train.parquet \
    data.val_files=/mnt/sdb/hanxu/projects/secretary-r1/data/sft_gsm8k_v2/test.parquet \
    data.prompt_key=question \
    data.response_key=formatted_answer \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=/data2/share/hanxu/model/Llama-3.2-3B-Instruct \
    trainer.default_local_dir=/data2/share/hanxu/verl/checkpoints/agent-omni/gsm8k/sft-Llama-3.2-3B-Instruct-v2 \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-Llama-3.2-3B-Instruct-v2 \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null

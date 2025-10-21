#!/bin/bash
#SBATCH --nodelist=chacha
#SBATCH --job-name=dit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --gres=shard:48
#SBATCH --mail-type=begin
#SBATCH --mail-type=end 

apptainer exec --nv \
    --bind $PHOTOCAST_DATA_DIR/images/roundshot/f61943996d2ca2f3362c4bcae233c11e:$PWD/data/f61943996d2ca2f3362c4bcae233c11e:ro \
    --bind $PHOTOCAST_OUTDIR/runs:$PWD/results \
    $HOME/.apptainer/photocast-dit.sif \
    torchrun train.py \
        --data-path ./data \
        --results-dir ./results \
        --model DiT-S/8 \
        --image-size 256 \
        --global-batch-size 256 \
        --num-classes 2 \
        --ckpt-every 10000 \
        --epochs 100  

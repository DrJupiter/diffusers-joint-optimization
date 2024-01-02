#!bin/sh
#BSUB -q gpuv100
#BSUB -J specialcourse 
#BSUB -n 4
#BSUB -gpu "num=4:mode=shared"
#BSUB -R "select[gpu32gb]"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -u s204123@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o %J.out
#BSUB -e %J.out


module load python3/3.11.4
module swap cudnn/v8.9.1.23-prod-cuda-12.X
module unload nccl/2.18.1-1-cuda-11.8
module swap cuda/12.2.2
module swap tensorrt/8.6.1.6-cuda-11.X tensorrt/8.6.1.6-cuda-12.X

source /zhome/33/4/155714/dtu/bin/activate

python3 -m accelerate.commands.launch --mixed_precision="fp16" /zhome/33/4/155714/diffusers-joint-optimization/expansion/train-torch-base.py 
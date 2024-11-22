# Compute Canada User Manual

# Environment Setup

## Log In

Set up multifactor authentication on the website at `My Account -> Multifactor Authentication Management`.

```shell
ssh $USER@cedar.alliancecan.ca
ssh $USER@graham.alliancecan.ca
ssh $USER@narval.alliancecan.ca								# possible issue with wandb
```

## Create Virtual Environment

Load `python/3.10` and `cuda/12.2` before creating the virtual environment.

```shell
module load python/3.10										# load python first
module load cuda/12.2										# load cuda next
virtualenv --no-download $ENV_NAME							# create virtual environment
```

## Activate Virtual Environment

Load `python/3.10` and `cuda/12.2` before activating the virtual environment.

```shell
module load python/3.10										# load python first
module load cuda/12.2										# load cuda next
source $ENV_NAME/bin/activate								# activate virtual environment
```

# Running Jobs

## Interactive Jobs

Apply for a CPU or GPU session for interactive usage.

```shell
# check node availability
sinfo -eO "CPUs:8,Memory:9,Gres:80,NodeAI:14,NodeList:50"

# apply for a cpu or gpu session
salloc --account=def-bboulet --gres=gpu:a100_3g.20gb:1 --cpus-per-task=2 --mem=40gb --time=1:0:0
salloc --time=1:0:0 --cpus-per-task=1 --mem=64000M --account=def-cpsmcgil
salloc --time=1:0:0 --gpus-per-node=1 --mem=64000M --account=def-cpsmcgil
salloc --time=1:0:0 --gpus-per-node=p100:1 --mem=64000M --account=def-cpsmcgil
exit
```

## Submit to Server

```shell
>> sbatch nas_exp.sh
```

Where `nas_exp.sh` is the script to run the experiments, which should be something like the following:

The lines starting with #SBATCH are used to set up the hardware resources.

```sh
#!/bin/bash
#SBATCH --account=def-cpsmcgil
#SBATCH --output=log/nas_exp.out
#SBATCH --gres=gpu:v100l:1  # --gres=gpu:a100:1
#SBATCH --time=20:20:00
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham. #40000 Narval

source /home/yeyuan66/nips2023c/bin/activate
#module load python/3.7
#module load cuda/10.2
module load cuda
#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate

nvidia-smi

#conda activate local
#conda install -n EXP40 pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
#conda install -c dglteam dgl-cuda10.2
#python test_dgl.py
echo "prt"
#python -u grad.py --device 0
#python -u grad.py --task DKittyMorphology-Exact-v0 --method triteach --N 20 --device 0
#python -u test.py 

tasks=('AntMorphology-Exact-v0' 'DKittyMorphology-Exact-v0' 'TFBind8-Exact-v0' 'TFBind10-Exact-v0')
for seed in {1..20};
do
	for task in ${tasks[*]};
	do
		python -u main.py
	done
done

deactivate
```

## Check the Status of Submitted Jobs

```shell
>> sq
 JOBID     USER      ACCOUNT           NAME  ST  TIME_LEFT NODES CPUS TRES_PER_N MIN_MEM NODELIST (REASON)
 65514402 yeyuan66 def-cpsmcgil  nas_exp.sh  PD   20:20:00     1    4 gres:gpu:1  32000M  (Priority)

```

`watch -n 5 'sq'`  Check `sq` status every five seconds

# Tune all hyperparameters

```bash
#!/bin/bash

# Arrays of hyperparameters to grid search over
lrs=("1e-5" "5e-5" "1e-4" "5e-4" "1e-3")
weight_decays=("1e-1" "1e-2" "1e-3" "1e-4" "0")

# Loop through all combinations of hyperparameters
for lr in "${lrs[@]}"; do
  for weight_decay in "${weight_decays[@]}"; do

    # Generate a unique output file for each job to avoid overwriting
    output_file="/home/haolun/scratch/SNAKE/exp_out/M2.1_T5S_D3_toy_${lr}_${weight_decay}.out"

    # Launch a separate job for each hyperparameter combination
    sbatch <<EOL
#!/bin/bash
#SBATCH --account=def-cpsmcgil
#SBATCH --output=${output_file}
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M

source /home/haolun/projects/def-cpsmcgil/SNAKE/venv_SNAKE/bin/activate
module load cuda
nvidia-smi

# Run your script with the current hyperparameter combination
WANDB__SERVICE_WAIT=300 python3 /home/haolun/scratch/SNAKE/experiment_cc.py --model_choice="Single-mask-Multi-Entity-Step1" --use_data=3 --pretrained_model_name='T5-small' --batch_size=3 --epochs=500 --log_wandb=True --use_lora=True --lr=${lr} --weight_decay=${weight_decay}

deactivate
EOL

  done
done

```

# Management

## Check Disk Usage

```shell
diskusage_report --per_user --all_users
```

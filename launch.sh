#!/bin/sh

batches=$1
budget_multiplier=$2
number_of_kernels=$3
popsize_factor=$4
model=$5
prefix=$6
# module load py3-scipy/1.1.0  # need numpy etc

reportDir="./reporting/${prefix}_k${number_of_kernels}_${popsize_factor}_b${budget_multiplier}_m${model}_hv${hv_all}"

echo $(mkdir $reportDir)

date  # for the record only
for ((i=1; i<=$batches; i++))
do
    # Comment if SLRUM is NOT available
    srun --ntasks=1 -l -o $reportDir/slurm.out$i -e $reportDir/slurm.err$i python3 run.py budget_multiplier=$budget_multiplier batch=$i/$batches number_of_kernels=$number_of_kernels popsize_factor=$popsize_factor model=$model prefix=$prefix &
    # Uncomment if SLRUM is NOT available and you can use nohup instead
    # nohup python3 run.py budget_multiplier=$budget_multiplier batch=$i/$batches number_of_kernels=$number_of_kernels popsize_factor=$popsize_factor model=$model prefix=$prefix > $reportDir/log.out$i 2> $reportDir/log.err$i &
done

wait  # the above background processes will be killed by the end of the script, so we better wait for them
date  # for the record only


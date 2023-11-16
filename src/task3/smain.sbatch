#!/bin/bash
#SBATCH -J lab3 
#SBATCH -p gpuserv 
#SBATCH -e errjob.%j.err
#SBATCH -o outjob.%j.out
#SBATCH -N 1  
#SBATCH -n 3  
#SBATCH -c 4  	
#SBATCH -t 00:01:00

module load nvidia/cuda 
                        
echo "Current path= `pwd`"
echo "node=`hostname`"
echo "nproc=`nproc`"
echo $SLURM_JOBID   
echo $SLURM_SUBMIT_DIR 
echo $SLURM_JOB_NODELIST
echo $SLURM_CPUS_PER_TASK
echo $SLURM_NTASKS	

nvcc main.cu -o lab3 -lpng -lpng16
./lab3

sleep 10


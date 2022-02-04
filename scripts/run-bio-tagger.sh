#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -p gputest
#SBATCH -t 00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2001426
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err


#function on_exit {
#    rm -f out-$SLURM_JOBID.tsv
#    rm -f jobs/$SLURM_JOBID
#}
#trap on_exit EXIT


batch_size=32
sentences_on_batch=50000
#input_file="/home/joffe/projektit/temp/bio-ner-tagger/texts-2022/pubmed22n0670.tsv"
#ner_model="/home/joffe/projektit/temp/bio-ner-tagger/ner-model"
input_file="./texts-2022/pubmed22n0471.tsv"
ner_model="./ner-model"

#rm -f logs/latest.out logs/latest.err
#ln -s "$SLURM_JOBID.out" "logs/latest.out"
#ln -s "$SLURM_JOBID.err" "logs/latest.err"


#module purge
#module load tensorflow/2.4

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# echo "START $SLURM_JOBID: $(date)"


python3 ner-cmv-tagger.py \
    --batch_size $batch_size \
    --input_data "$input_file" \
    --output_spans "./pubmed-output/out.spans" \
    --output_tsv "./pubmed-output/out.tsv" \
    --sentences_on_batch "$sentences_on_batch" \
    --ner_model_dir "$ner_model" \

#seff $SLURM_JOBID
#gpuseff $SLURM_JOBID
#echo "END $SLURM_JOBID: $(date)"

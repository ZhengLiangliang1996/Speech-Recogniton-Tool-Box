#!/usr/bin/env bash

for loop in {2..300} 
do
    echo "loop index: $loop"
    if [ $loop -eq 1 ]
    then
        python libri_train.py --mode=train --checkpoint_dir '/scratch/brussel/102/vsc10260/CHECKPOINT' || break
    else
        python libri_train.py --mode=test --restore=True --checkpoint_dir '/scratch/brussel/102/vsc10260/CHECKPOINT' || break
        python libri_train.py --batch_size=64 --mode=train --restore=True --checkpoint_dir '/scratch/brussel/102/vsc10260/CHECKPOINT' || break
    fi
done

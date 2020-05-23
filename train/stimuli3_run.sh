#!/usr/bin/env bash

<<<<<<< HEAD
python stimuli3_main.py --mode=train --checkpoint_dir '/scratch/brussel/102/vsc10260/CHECKPOINT'

for loop in {2..300} 
do
    echo "loop index: $loop"

    python stimuli3_main.py --mode=test --restore=True --checkpoint_dir '/scratch/brussel/102/vsc10260/CHECKPOINT' || break
    python stimuli3_main.py --batch_size=64 --mode=train --restore=True --checkpoint_dir '/scratch/brussel/102/vsc10260/CHECKPOINT' || break

=======
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
>>>>>>> master
done

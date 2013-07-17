#!/usr/bin/env bash

# Run parallel EM locally
python big_data_em/preprocess.py # this creates a file called encoded.txt

num_iterations=20

rm -f parameters.txt
for ((i=0; i<${num_iterations}; i++ ))
do
    # Note: add "-r local" when the lxmls module is installed.
    python big_data_em/emstep.py -r local --file=word_tag_dict.pkl < encoded.txt > parameters_new.txt
    mv parameters_new.txt hmm.txt
    echo "Iteration $i"
done


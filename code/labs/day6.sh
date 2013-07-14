#!/usr/bin/env bash

# Run parallel EM locally
python big_data_em/preprocess.py # this creates a file called encoded.txt

num_iterations=20

rm -f parameters.txt
for ((i=0; i<${num_iterations}; i++ ))
do
    # Note: add "-r local" when the lxmls module is installed.
    python big_data_em/parallel_em.py < encoded.txt > parameters_new.txt
    mv parameters_new.txt parameters.txt
    echo "Iteration $i"
    grep "Log-likelihood" parameters.txt
done

# Run distributed EM on Amazon
# IMPORTANT: you must export your AWS_ACCESS_KEY and AWS_SECRET_KEY variables before running this!
# python big_data_em/parallel_em.py -r emr --num-ec2-instances 2 --aws-region eu-west-1

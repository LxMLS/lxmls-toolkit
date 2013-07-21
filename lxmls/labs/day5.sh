#!/usr/bin/env bash

# Run wordcount locally
python big_data/wordcount.py ../data/wikipedia/en_perline001.txt > results_local_wordcount.txt

# Run wordcount on Amazon
# IMPORTANT: you must export your AWS_ACCESS_KEY and AWS_SECRET_KEY variables before running this!
# python big_data/wordcount.py -r emr ../data/wikipedia/en_perline001.txt --num-ec2-instances 2 --aws-region eu-west-1 > results_AWS_wordcount.txt

# Run language detection locally
python big_data/trimercount.py ../data/wikipedia/en_perline001.txt > output.en.txt
python big_data/trimercount.py ../data/wikipedia/pt_perline01.txt > output.pt.txt
python big_data/postprocess.py

# Run language detection on Amazon
# python big_data/kmers.py -r emr s3://lxmls-labs/en_perline01.txt --num-ec2-instances 10 --aws-region eu-west-1 > output.en.txt
# python big_data/kmers.py -r emr s3://lxmls-labs/pt_perline10.txt --num-ec2-instances 10 --aws-region eu-west-1 > output.pt.txt
# python big_data/postprocess.py

#!/usr/bin/env bash

echo "------------"
echo "Exercise 5.1"
echo "------------"

# Run wordcount locally 
python lxmls/big_data/wordcount.py data/wikipedia/en_perline001.txt > results.txt
grep '\<will\>' results.txt

echo "------------"
echo "Exercise 5.2"
echo "------------"

echo "# Run the amazon part (see in $0, Line $LINENO)"
# Run wordcount on Amazon ...
echo "# Remote login with"
echo ""
echo "ssh <username>@ec2-46-137-27-37.eu-west-1.compute.amazonaws.com"
echo ""
echo "# Use <username> and <password> given in the handout, then run"
echo ""
echo "cd wordcount"
echo "python wordcount.py -r emr en_perline001.txt > results2.txt"
echo "grep '\<will\>' results2.txt"
echo ""
echo "python wordcount.py -r emr s3://lxmls-labs/pt_perline10.txt"
echo ""

echo "------------"
echo "Exercise 5.3"
echo "------------"

# Run language detection locally
python big_data/trimercount.py ../data/wikipedia/en_perline001.txt > output.en.txt
python big_data/trimercount.py ../data/wikipedia/pt_perline01.txt > output.pt.txt
python big_data/postprocess.py

# Note: I copied by scp the scripts there to do the test, the students do not know this (probably). How are they going to build the post processor from day1 code, al by hand? 
# scp lxmls/big_data/kmers.py <username>@ec2-46-137-27-37.eu-west-1.compute.amazonaws.com:./wordcount/
# scp lxmls/big_data/postprocess.py <username>@ec2-46-137-27-37.eu-west-1.compute.amazonaws.com:./wordcount/

echo "Run language detection on Amazon (see in $0, Line $LINENO)"
echo ""
echo "python kmers.py en_perline001.txt > output.en.txt"
echo "python kmers.py pt_perline01.txt > output.pt.txt"
echo "python postprocess.py"

#!/bin/bash
# This script converts all notebooks into python files and stores them in the
# tests/ folder. This is useful for debugging. Note the some ipython calls need
# to be manually remove still
set -o errexit
set -o nounset
set -o pipefail

if [ ! -d "labs/" ];then
    echo "Should be called from the lxmls-toolkit root as 
    
    bash labs/$0
"
    exit
fi

for notebook in $(find labs/notebooks/ -iname '*.ipynb' | grep -v .ipynb_checkpoints/);do
    # Keep one level depth folder structure    
    output_dir=labs/scripts/$(basename $(dirname $notebook))
    if [ ! -d $output_dir ];then
        mkdir -p $output_dir
    fi
    jupyter nbconvert --to script $notebook --output-dir $output_dir

done

# Remove inline jupyter lines. These cause errors
for file in $(find labs/scripts/ -iname '*.py');do
    python labs/remove_ipython_inline.py $file
done

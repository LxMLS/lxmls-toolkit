"""
Removes get_ipython() inline calls from scripts
"""
import re

import sys

EXCLUDE_LINES = re.compile('[^#]*get_ipython().*')
PYTHON_FILE = re.compile('.*\.py$')

if __name__ == '__main__':

    # Argument hanlding
    if len(sys.argv[1:]) != 1:
        print("\n%s <file>.py\n" % sys.argv[0])
        exit(1)
    script_file = sys.argv[1]
    if not PYTHON_FILE.match(script_file):
        print("\nExpected a python file\n")
        exit(1)

    # Line filtering
    with open(script_file, 'r', 'utf-8') as fid:
        new_lines = []
        for line in fid.readlines():
            if EXCLUDE_LINES.match(line.strip()):
                print("Skipped: %s" % line.strip())
            else:
                new_lines.append(line)

    # Rewrite file
    with open(script_file, 'w', 'utf-8') as fid:
        for line in new_lines:
            fid.write(line)

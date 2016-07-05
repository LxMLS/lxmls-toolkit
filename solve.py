'''
This script solves the exercises of days that have been completed. Jut in case
the students did not made it by their own.
'''
import sys
import urllib2

def download_and_replace(url, target_file):
    '''
    Downloads file through http with progress report. Version by PabloG 
    obtained in stack overflow
    
    http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http
    -using-python
    '''
    # Try to connect to the internet
    try:
        u = urllib2.urlopen(url)
    except Exception, err:
        if getattr(err, 'code', None): 
            print "\nError: %s Could not get file %s\n" % (err.code, url)     
        else:
            # A generic error is most possibly no available internet
            print "\nCould not connect to the internet\n"    
        exit(1)
 
    with open(target_file, 'wb') as f:
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl, 
                                                  file_size_dl*100./file_size)
            status = status + chr(8)*(len(status)+1)


# CONFIGURATION
master_URL = 'https://github.com/LxMLS/lxmls-toolkit/raw/master/'
labs_URL = 'https://github.com/LxMLS/lxmls-toolkit/raw/student/'

# FILES TO BE REPLACED FOR THAT DAY
code_day = {
    'day1': ['lxmls/classifiers/multinomial_naive_bayes.py', 
             'lxmls/classifiers/perceptron.py'], 
    'day2': ['lxmls/sequences/hmm.py',
             'lxmls/sequences/sequence_classification_decoder.py'],
    'day3': ['lxmls/sequences/structured_perceptron.py'],
    'day4': ['lxmls/parsing/dependency_decoder.py'],
    'day5': ['lxmls/deep_learning/mlp.py'],
    'day6': ['lxmls/deep_learning/rnn.py']
           }

# ARGUMENT PROCESSING
if ((len(sys.argv) == 2) and 
    (sys.argv[1] in ['day0', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6'])): 
    undo_flag = 0
    day = sys.argv[1]

elif ((len(sys.argv) == 3) and 
      (sys.argv[1] == '--undo') and
      (sys.argv[2] in ['day0', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6'])): 
    undo_flag = 1
    day = sys.argv[2]

else:
    print ("\nUsage:\n"
           "\n"
           "python solve.py day<day number>         # To solve exercise \n"
           "\n"
           "python solve.py --undo day<day number>  # To undo solve\n"
           "" )
    exit(1)

# CHECK THERE ARE FILES TO SAVE
if day in code_day:
    print "\nsolving %s" % day 
else:
    print "\nTheres actually no code to solve on %s!\n" % day
    exit()

# OVERWRITE THE FILES TO SOLVE THEM
for pyfile in code_day[day]:
    if undo_flag:
        download_and_replace(labs_URL + pyfile, pyfile)
        print "Unsolving: %s" % pyfile 
    else:
        download_and_replace(master_URL + pyfile, pyfile)
        print "Solving: %s" % pyfile 

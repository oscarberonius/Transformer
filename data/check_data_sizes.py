import sys

def symbols(fname):
    max_size = 0
    with open(fname, 'r') as f:
        for line in f.readlines():
            size = len(line)
            if(size > max_size):
                max_size = size

    print(f'Max # of symbols:{max_size}')

def words(fname):
    max_size = 0
    with open(fname, 'r') as f:
        for line in f.readlines():
            size = len(line.split(' '))
            if(size > max_size):
                max_size = size

    print(f'Max # of words:{max_size}')
    
if __name__=='__main__':
    args = sys.argv
    fname = args[1]
    symbols(fname)
    words(fname)

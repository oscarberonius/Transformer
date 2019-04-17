import sys

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
        return i+1

if __name__=='__main__':
    args = sys.argv
    fname = args[1]
    file_len = file_len(fname)
    print(f'{fname} contains {file_len} lines')

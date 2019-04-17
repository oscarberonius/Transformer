import sys
import readlines

def remove_lines(fname, fnew, numlines):
    total_lines=readlines.file_len(fname)
    with open(fname, 'r') as f:
        data_list = f.readlines()
    #    for i, l in enumerate(f):
    #        pass
    #    total_lines = i+1
    new_lines = str(total_lines-int(numlines))
    #new_name = fname[0:-4]+'_'+new_lines+'_dps'+'.txt'
    
    del data_list[0:int(numlines)]
    with open(fnew, 'w') as f:
        f.writelines(data_list)
    
    return fnew 
        

if __name__== '__main__':
    args = sys.argv
    fname = args[1]
    fnew = args[2]
    numlines = args[3]
    new_file_name = remove_lines(fname, fnew, numlines)
    print(f'removed {numlines} lines in {new_file_name}')


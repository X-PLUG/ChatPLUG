# -*- coding: utf-8 -*- 
"""
@Time : 2022-07-04 21:18 
@Author : zhimiao.chh 
@Desc : 
"""
import sys
import os
from xdpx.utils import io


def cli_main(argv):
    """for interactive testing of model behaviour"""
    assert len(argv) >= 4
    file = argv[1]
    cut_num = int(argv[2])
    output_dir = argv[3]
    start_file_index = 0
    if len(argv) == 5:
        start_file_index = int(argv[4])
    # get line num
    row_num = 0
    with io.open(file) as f:
        for _ in f.readlines():
            row_num+=1
    print("file has {} rows...".format(row_num))

    if not io.exists(output_dir):
        io.makedirs(output_dir)
    file_row_num = row_num // cut_num + 1
    _, file_name = os.path.split(file)
    file_name_prefix,file_name_suffix = os.path.splitext(file_name)
    with io.open(file) as f:
        file_index = start_file_index
        out_file = os.path.join(output_dir,file_name_prefix+"_"+str(file_index)+file_name_suffix)
        outf = io.open(out_file, mode='w')
        row = 1
        print("start writing " + out_file + "......")
        while True:
            line = f.readline()
            if not line: break
            outf.write(line)
            if row < file_row_num:
                row += 1
            else:
                outf.close()
                file_index += 1
                out_file = os.path.join(output_dir, file_name_prefix + "_" + str(file_index) + file_name_suffix)
                outf = io.open(out_file, mode='w')
                row = 1
                print("start writing " + out_file + "......")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: x-script cut_file $file $cut_num $output_dir $start_file_index')
        exit()
    cli_main(sys.argv)

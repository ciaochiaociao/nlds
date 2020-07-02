from nlu.parser_utils import bioes2iob1_file
from argparse import *

argparser = ArgumentParser()

argparser.add_argument(
    '-i', '--input',
)
argparser.add_argument(
    '-o', '--output',
)

args = argparser.parse_args()
in_fpath = args.input
out_fpath = args.output

bioes2iob1_file(in_fpath, out_fpath)
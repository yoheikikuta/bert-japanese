# coding: utf-8


"""
Modify sentencepiece trained files

Such vocabularies in vocab:
1. Those start with '_(\u2581)'
2. And vocabularies made via getting rid of '_' from them are not contained in vocab
are replaced by surfaces without '_'.
In other words, we consider some '_xxx' when pretraining as 'xxx' when finetuning.
(See the reason of trying this modification in sp.ipynb)
"""


import argparse
import shutil


def modify_sp_vocab(src_path, dest_path):
    lines = []
    with open(src_path, 'r') as f:
        lines = [_.strip().split('\t') for _ in f.readlines()]
    
    tokens = []
    for t, _ in lines:
        tokens.append(t)
    
    is_u2581_heads = lambda t: len(t) > 1 and t.startswith('\u2581')
    
    output_lines = lines[:]
    for p in output_lines:
        if is_u2581_heads(p[0]):
            pp = p[0].strip('\u2581')
            if not pp in tokens:
                p[0] = pp
   
    with open(dest_path, 'w') as f:
        f.write('\n'.join(['%s\t%s'%(tuple(_)) for _ in output_lines]))
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='replace sentencepiece dict')
    parser.add_argument('--src_path', type=str, required=True, help='path to source vocab')
    parser.add_argument('--dest_path', type=str, required=True, help='path to destination vocab')
    args = parser.parse_args()
    
    modify_sp_vocab(args.src_path, args.dest_path)
    
    # copy model file
    src_path = args.src_path.strip('.vocab') + '.model'
    dest_path = args.dest_path.strip('.vocab') + '.model'
    shutil.copy(src_path, dest_path)


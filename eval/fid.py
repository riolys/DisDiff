import os
from pytorch_fid.fid_score import calculate_fid_given_paths
import argparse
batch_size = 1
device = 'cuda'
dims = 2048
num_workers = 1
fid_results = []

def parse_args():
    parser = argparse.ArgumentParser(description='FDFR and ISM evaluation')
    parser.add_argument('--data_dir', type=str, default='', required=True, help='path to datadir')
    parser.add_argument('--emb_dir', type=str, default='', required=True)
    parser.add_argument('--prompt', type=str, default='a_photo_of_sks_person', required=True)
    parser.add_argument('--celeb', type=bool, default=False)
    args = parser.parse_args()
    return args

arg = parse_args()
base_dir = arg.data_dir
prompt = 'a_photo_of_sks_person'
prompt = arg.prompt
if base_dir.endswith('EF'):
    prompt = prompt.replace('sks', 't@t')
print('prompt:', prompt)
for dic in os.listdir(base_dir):
    print(dic)
    if os.path.isdir(os.path.join(base_dir, dic)):
        data_dir = os.path.join(base_dir, dic, 'checkpoint-1000/dreambooth', prompt)
        if arg.celeb:
            import re
            emb_dir = os.path.join(arg.emb_dir, re.findall(r'\d+', dic)[0], 'set_A')
        else:
            emb_dir = os.path.join(arg.emb_dir, dic[:7], 'set_A')
        fid_value = calculate_fid_given_paths([emb_dir, data_dir],
                                            batch_size,
                                            device,
                                            dims,
                                            num_workers)
        fid_results.append(fid_value)
        print('single fid:', fid_value)

average_fid = sum(fid_results) / len(fid_results)
print(f"Average FID: {average_fid}")
output_file = os.path.join(base_dir, 'eval.txt')
with open (output_file, 'a') as f:
    f.write(f"Average FID is {average_fid}\n")
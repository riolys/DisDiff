import argparse
import os
from PIL import Image
from brisque import BRISQUE
import ipdb
def parse_args():
    parser = argparse.ArgumentParser(description='FDFR and ISM evaluation')
    parser.add_argument('--data_dir', type=str, default='', required=True, help='path to datadir')
    parser.add_argument('--prompt', type=str, default='a_photo_of_sks_person', required=True)
    args = parser.parse_args()
    return args


def main(args):
    Base_path = args.data_dir
    obj = BRISQUE(url=False)
    prompt_score = 0
    count = 0
    prompt = args.prompt
    if Base_path.endswith('EF'):
        prompt = prompt.replace('sks', 't@t')
    print('prompt', prompt)
    for path in os.listdir(Base_path):
        if os.path.isdir(os.path.join(Base_path, path)):
            if 'dreambooth' in path:
                pth = os.path.join(Base_path, path, 'checkpoint-1000/dreambooth', prompt )
                for img_name in os.listdir(pth):
                    if "png" in img_name or "jpg" in img_name:
                        img_path = os.path.join(pth, img_name)
                        img = Image.open(img_path)
                        brisque_score = obj.score(img)
                        prompt_score += brisque_score
                        count += 1
            else:
                raise ValueError("The brisques score calculate Dir is not dreambooth")
    print('total count:', count)
    brisque = prompt_score/count
    output_file = os.path.join(Base_path, 'eval.txt')
    with open (output_file, 'a') as f:
        f.write("The brisque score is {}\n".format(brisque))
    return brisque


if __name__ == '__main__':
    args = parse_args()
    brisque = main(args)
    print("The brisque score is {}".format(brisque))


                

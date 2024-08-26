from deepface import DeepFace
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse
from compute_idx_emb import compute_idx_embedding
import ipdb

def compute_face_embedding(img_path):
    """Extract face embedding vector of given image
    Args:
        img_path (str): path to image
    Returns:
        None: no face found
        vector: return the embedding of biggest face among the all found faces
    """
    try:
        resps = DeepFace.represent(img_path = os.path.join(img_path), 
                                   model_name="ArcFace", 
                                   enforce_detection=True, 
                                   detector_backend="retinaface", 
                                   align=True)
        if resps == 1:
            # detect only 1 face
            return np.array(resps[0]["embedding"])
        else:
            # detect more than 1 faces, choose the biggest one
            resps = list(resps)
            resps.sort(key=lambda resp: resp["facial_area"]["h"]*resp["facial_area"]["w"], reverse=True)
            return np.array(resps[0]["embedding"])
    except Exception:
        # no face found
        return None

def get_precomputed_embedding(path):
    """Get face embedding by loading the path to numpy file
    Args:
        path (str): path to numpy file 
    Returns:
        vector: face embedding
    """
    return np.load(path)



def matching_score_id(image_path, avg_embedding):
    """getting the matching score between face image and precomputed embedding

    Args:
        img (2D images): images
        emb (vector): face embedding

    Returns:
        None: cannot detect face from img
        int: identity score matching
    """
    image_emb = compute_face_embedding(image_path)
    id_emb = avg_embedding
    if image_emb is None:
        return None
    image_emb, id_emb = torch.Tensor(image_emb), torch.Tensor(id_emb)
    ism = F.cosine_similarity(image_emb, id_emb, dim=0)
    return ism

def matching_score_genimage_id(images_path, list_id_path):
    image_list = os.listdir(images_path)
    fail_detection_count = 0
    ave_ism = 0
    avg_embedding = compute_idx_embedding(list_id_path)

    for image_name in image_list:
        image_path = os.path.join(images_path, image_name)
        ism = matching_score_id(image_path, avg_embedding)
        if ism is None:
            fail_detection_count += 1
        else:
            ave_ism += ism
    if fail_detection_count != len(image_list):
        return ave_ism/(len(image_list)-fail_detection_count), fail_detection_count/len(image_list)
    return 0, 1

def parse_args():
    parser = argparse.ArgumentParser(description='FDFR and ISM evaluation')
    parser.add_argument('--data_dir', type=str, default='', required=True, help='path to datadir')
    parser.add_argument('--prompt', type=str, default='a_photo_of_sks_person', required=True)
    parser.add_argument('--emb_dir', type=str, default='', required=True)
    parser.add_argument('--celeb', type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    arg = parse_args()
    base_dir = arg.data_dir
    prompt = arg.prompt
    print("prompt : ", prompt)
    if base_dir.endswith('EF'):
        prompt = prompt.replace('sks', 't@t')
    count, ism, fdr = 0, 0, 0
    for dic in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, dic)):
            data_dir = os.path.join(base_dir, dic, 'checkpoint-1000/dreambooth', prompt)
            # emb_dir = os.path.join(arg.emb_dir, dic[:7], 'set_A')
            if arg.celeb:
                import re
                emb_dir = os.path.join(arg.emb_dir, re.findall(r'\d+', dic)[0], 'set_A')
            else:
                emb_dir = os.path.join(arg.emb_dir, dic[:7], 'set_A')
            ism_per, fdr_per = matching_score_genimage_id(data_dir, [emb_dir])
            ism += ism_per
            fdr += fdr_per
            print(dic, 'ism:', ism_per, 'fdr:', fdr_per)
            count += 1
    ism = ism / count
    fdr = fdr / count
    print("ISM and FDFR are {} and {}".format(ism, fdr))
    output_file = os.path.join(base_dir, 'eval.txt')
    with open (output_file, 'a') as f:
        f.write("ISM and FDFR are {} and {}\n".format(ism, fdr))
if __name__ == '__main__':
    main()


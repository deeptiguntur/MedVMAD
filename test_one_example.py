import CLIP
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import MedVMAD_PromptLearner
from PIL import Image

import os
import random
import numpy as np
from utils import get_transform, normalize
from CLIP.adapter import CLIP_Inplanted

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# from visualization import visualizer
import cv2


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def visualizer(path, anomaly_map, img_size):
    filename = os.path.basename(path)
    dirname = os.path.dirname(path)
    vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
    mask = normalize(anomaly_map[0])
    vis = apply_ad_scoremap(vis, mask)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
    save_vis = os.path.join(dirname, f'anomaly_map_{filename}')
    cv2.imwrite(save_vis, vis)

from scipy.ndimage import gaussian_filter
def test(args):
    img_size = args.image_size
    features_list = args.features_list
    image_path = args.image_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    MedVMAD_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = CLIP.load("ViT-L/14@336px", device=device, design_details = MedVMAD_parameters)
    model.eval()

    model_img = CLIP_Inplanted(clip_model=model, features=args.features_list).to(device)
    model_img.eval()

    preprocess, target_transform = get_transform(args)


    prompt_learner = MedVMAD_PromptLearner(model.to("cpu"), MedVMAD_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)

    model_img.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model_img.det_adapters.load_state_dict(checkpoint["det_adapters"])
    model.to(device)
    # model.visual.DAPM_replace(DPAM_layer = 20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)

    img = Image.open(image_path)
    img = preprocess(img)
    
    print("img", img.shape)
    image = img.reshape(1, 3, img_size, img_size).to(device)
   
    with torch.no_grad():
        # image_features, patch_features = model.(image, features_list, DPAM_layer = 20)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        _, seg_patch_tokens, det_patch_tokens = model_img(image)
        seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
        det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

        # text_probs = det_patch_tokens @ text_features.permute(0, 2, 1)
        # text_probs = (text_probs/0.07).softmax(-1)
        # text_probs = text_probs[:, 0, 1]

        anomaly_map_list = []
        # for idx, patch_feature in enumerate(patch_features):
        #     if idx >= args.feature_map_layer[0]:
        #         patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
        #         similarity, _ = CLIP.compute_similarity(patch_feature, text_features[0])
        #         similarity_map = CLIP.get_similarity_map(similarity[:, 1:, :], args.image_size)
        #         anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
        #         anomaly_map_list.append(anomaly_map)

        # for layer in range(len(seg_patch_tokens)):
        #         seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
        #         anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features[0].t()).unsqueeze(0)
        #         B, L, C = anomaly_map.shape
        #         H = int(np.sqrt(L))
        #         anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
        #                                     size=args.image_size, mode='bilinear', align_corners=True)
        #         anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
        #         anomaly_map_list.append(anomaly_map.cpu().numpy())
        for layer in range(len(seg_patch_tokens)):
            seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
            anomaly_map = ( seg_patch_tokens[layer] @ text_features[0].t()).unsqueeze(0)
            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                        size=args.image_size, mode='bilinear', align_corners=True)
            anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
            anomaly_map_list.append(torch.from_numpy(anomaly_map.cpu().numpy()))  # Convert to tensor

        anomaly_map = torch.stack(anomaly_map_list)
        
        anomaly_map = anomaly_map.sum(dim = 0)
      
        anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )

        visualizer(image_path, anomaly_map.detach().cpu().numpy(), args.image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MedVMAD", add_help=True)
    # paths
    parser.add_argument("--image_path", type=str, default="./data/Brain_AD", help="path to test dataset")
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=336, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)

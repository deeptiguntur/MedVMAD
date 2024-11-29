import CLIP
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import MedVMAD_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
from CLIP.adapter import CLIP_Inplanted

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    MedVMAD_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = CLIP.load("ViT-L/14@336px", device=device, design_details = MedVMAD_parameters)
    model.eval()
  
    # Image learnable
    model_img = CLIP_Inplanted(clip_model=model, features=args.features_list).to(device)
    model_img.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

  ##########################################################################################
    prompt_learner = MedVMAD_PromptLearner(model.to("cpu"), MedVMAD_parameters)
    prompt_learner.to(device)
    model.to(device)
    # model.visual.DAPM_replace(DPAM_layer = 20)
    ##########################################################################################
    # Text optimizer
    optimizer_text = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    # Img optimizer
    seg_optimizer = torch.optim.Adam(list(model_img.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model_img.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    
    
    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label =  items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            # with torch.no_grad():
            #     # Apply DPAM to the layer from 6 to 24
            #     # DPAM_layer represents the number of layer refined by DPAM from top to bottom
            #     # DPAM_layer = 1, no DPAM is used
            #     # DPAM_layer = 20 as default
            #     image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)
            #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            ##############################################################################################
            # Image tokens
            # image_sq = image.squeeze(0).to(device)
            _, seg_patch_tokens, det_patch_tokens = model_img(image)
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]
                    
           ##################################################################################################
            # Prompt tokens
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            # Apply DPAM surgery
            # text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            # text_probs = text_probs[:, 0, ...]/0.07
            # image_loss = F.cross_entropy(text_probs.squeeze(), label.long().cuda())
            ################################################################################################
            image_loss = 0
            image_label = label.squeeze(0).to(device).float()
            for layer in range(len(det_patch_tokens)):
                det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                # Matrix mul
                # Multiply by 100
                anomaly_map = (det_patch_tokens[layer] @ torch.transpose(text_features[0], 0, 1)).unsqueeze(0)    
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                anomaly_score = torch.mean(anomaly_map)
                # image_loss += F.cross_entropy(anomaly_score.squeeze(), image_label.long().cuda())
                image_loss += loss_bce(anomaly_score, image_label)

            # Existing
            image_loss_list.append(image_loss.item())
            #########################################################################
            similarity_map_list = []
            # similarity_map_list.append(similarity_map)
            # for idx, patch_feature in enumerate(patch_features):
            #     if idx >= args.feature_map_layer[0]:
            #         patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
            #         similarity, _ = CLIP.compute_similarity(patch_feature, text_features[0])
            #         similarity_map = CLIP.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
            #         similarity_map_list.append(similarity_map)

            # for idx, patch_feature in enumerate(seg_patch_tokens):
            #     if idx >= args.feature_map_layer[0]:
            #         patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
            #         similarity, _ = CLIP.compute_similarity(patch_feature, text_features[0])
            #         similarity_map = CLIP.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
            #         similarity_map_list.append(similarity_map)

            seg_loss=0
            for layer in range(len(seg_patch_tokens)):
                seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                # print(seg_patch_tokens[layer].shape, text_feature_list[seg_idx].shape) # torch.Size([289, 768]) torch.Size([768, 2])
                anomaly_map = (seg_patch_tokens[layer] @ text_features[0].t()).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.image_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                seg_loss += loss_focal(anomaly_map, gt)
                seg_loss += loss_dice(anomaly_map[:, 1, :, :], gt)
                seg_loss += loss_dice(anomaly_map[:, 0, :, :], 1-gt)

            loss = 0
            # for i in range(len(similarity_map_list)):
            #     loss += loss_focal(similarity_map_list[i], gt)
            #     loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
            #     loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            loss += seg_loss

            optimizer_text.zero_grad()
            seg_optimizer.zero_grad()
            det_optimizer.zero_grad()
            (loss+image_loss).backward()
            optimizer_text.step()
            seg_optimizer.step()
            det_optimizer.step()
            loss_list.append(loss.item())
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            # torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)
            # Image learnable save
            torch.save({"prompt_learner": prompt_learner.state_dict(), 'seg_adapters': model_img.seg_adapters.state_dict(), 'det_adapters': model_img.det_adapters.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MedVMAD", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/Brain_AD", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')


    parser.add_argument("--dataset", type=str, default='brain', help="train dataset name")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=2, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=336, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)

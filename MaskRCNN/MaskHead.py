from sklearn import metrics
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
from inspect import modulesbyfile
from tqdm import tqdm
import h5py
import albumentations as A
from PIL import Image
from skimage.transform import resize
import warnings
from torchvision.models.detection.image_list import ImageList

warnings.filterwarnings('ignore')
torch.set_printoptions(linewidth=100)

np.set_printoptions(linewidth=100)
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
seed = 17
torch.manual_seed(seed)

def IOU_all(tar1, tar2):
    ##################################
    #TODO compute the IOU between the boxA, boxB boxes
    ##################################
    xl_int = torch.maximum(tar1[:,0], tar2[0])
    yl_int = torch.maximum(tar1[:,1], tar2[1])
    xr_int = torch.minimum(tar1[:,2], tar2[2])
    yr_int = torch.minimum(tar1[:,3], tar2[3])
    Area_of_int = torch.maximum(torch.Tensor([0]).to(device), xr_int - xl_int) * torch.maximum(torch.Tensor([0]).to(device), yr_int - yl_int)
    Area_of_union = (torch.abs(tar1[:,2] - tar1[:,0]) * torch.abs(tar1[:,3] - tar1[:,1])) + (torch.abs(tar2[2] - tar2[0]) * torch.abs(tar2[3] - tar2[1])) - Area_of_int
    return Area_of_int/(Area_of_union+1e-6)


def xyxy_xywh(coords, to_wh=False):
  if to_wh:
    if coords.shape[:]==torch.Size([4]):
      coords =  coords.reshape(1,4)
    newcoords = torch.zeros(coords.shape).to(device)
    xl  = coords[:,0]
    yl  = coords[:,1]
    xr  = coords[:,2]
    yr  = coords[:,3]

    newcoords[:,2] = xr - xl
    newcoords[:,3] = yr - yl
    newcoords[:,0] = xl + newcoords[:,2]/2
    newcoords[:,1] = yl + newcoords[:,3]/2
    return newcoords

  else:
    newcoords = torch.zeros(coords.shape).to(device)
    cx  = coords[:,0]
    cy  = coords[:,1]
    w  = coords[:,2]
    h  = coords[:,3]

    newcoords[:,0] = cx - w/2
    newcoords[:,1] = cy - h/2
    newcoords[:,2] = cx + w/2 
    newcoords[:,3] = cy + h/2
    return newcoords

def pretrained_models_680(checkpoint_file,eval=True):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    if(eval):
        model.eval()
    model = model.to(device)
    backbone = model.backbone
    rpn = model.rpn

    if(eval):
        backbone.eval()
        rpn.eval()

    rpn.nms_thresh=0.6
    checkpoint = torch.load(checkpoint_file)

    backbone.load_state_dict(checkpoint['backbone'])
    rpn.load_state_dict(checkpoint['rpn'])

    return backbone, rpn

def MatrixNMS(sorted_masks, sorted_scores, method='gauss', gauss_sigma=0.5):
    n = len(sorted_scores)
    sorted_masks = sorted_masks.reshape(n, -1)
    intersection = torch.mm(sorted_masks, sorted_masks.T)
    areas = sorted_masks.sum(dim=1).expand(n, n)
    union = areas + areas.T - intersection
    ious = (intersection / union).triu(diagonal=1)

    ious_cmax = ious.max(0)[0].expand(n, n).T
    decay = (1 - ious) / (1 - ious_cmax)
    decay = decay.min(dim=0)[0]
    return sorted_scores * decay

def output_decodingd(regressed_boxes_t,flatten_proposals, labels, device='cpu'):
  #######################################
  # TODO decode the output
  #######################################
  flatten_proposals = xyxy_xywh(flatten_proposals, to_wh=True)
  xa = flatten_proposals[:,0]
  ya = flatten_proposals[:,1]
  wa = flatten_proposals[:,2]
  ha = flatten_proposals[:,3]

  newcoords = torch.zeros(regressed_boxes_t.shape)
  tx  = regressed_boxes_t[:,0]
  ty  = regressed_boxes_t[:,1]
  tw  = regressed_boxes_t[:,2]
  th  = regressed_boxes_t[:,3]

  # Decoding
  cx = wa*tx + xa
  cy = ha*ty + ya
  w = wa* torch.exp(tw)
  h = ha* torch.exp(th)

  newcoords[:,0] = cx - w/2
  newcoords[:,1] = cy - h/2
  newcoords[:,2] = cx + w/2 
  newcoords[:,3] = cy + h/2

  newcoords[(labels==0)[:,0],:] = 0
  return newcoords

class BoxHead(pl.LightningModule):
    def __init__(self,Classes=3,P=7):
        super(BoxHead, self).__init__()
        self.loss_sum=0
        self.closs_sum=0
        self.rloss_sum=0
        self.epoch_loss=[]
        self.val_epoch_loss=[]
        self.epoch_closs=[]
        self.val_epoch_closs=[]
        self.epoch_rloss=[]
        self.val_epoch_rloss=[]
        self.C=Classes
        self.P=P
        self.nreg = 0
        self.celoss=nn.CrossEntropyLoss()
        self.smoothl1 = nn.SmoothL1Loss(reduction ='sum')
        self.feature_sizes = np.array([ [200, 272], [100, 136], [50, 68], [25, 34], [13, 17]])
        # TODO initialize BoxHead
        self.intermediate_layer =  nn.Sequential(
            nn.Linear(256 * P * P, 1024), 
            nn.ReLU(),

            nn.Linear(1024,1024), 
            nn.ReLU()
            )
            
        self.classhead= nn.ModuleList()    #maybe wont need
        self.classhead.append(nn.Linear(1024, Classes + 1))

        self.softmax = nn.Softmax(dim=1)

        self.reghead= nn.ModuleList()
        self.reghead.append(nn.Linear(1024, 4 * Classes))

    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth(self,proposals,gt_labels,bbox):
        labels = torch.empty(0, 1).to(device)
        regressor_target = torch.empty(0, 4).to(device)
        for i in range(len(proposals)):
          prop_im_xy = proposals[i]
          gtlab_im = gt_labels[i].to(device)
          gtbbox_im = bbox[i].to(device)
          gtbbox_im_wh = xyxy_xywh(gtbbox_im, to_wh=True)
          prop_im = xyxy_xywh(prop_im_xy, to_wh=True)
          
          ious =  torch.zeros((len(gtlab_im), len(prop_im))).to(device)
          for gts in range(len(gtlab_im)):
            ious[gts] = IOU_all(prop_im_xy, gtbbox_im[gts])
          ious, bbox_idx = torch.max(ious, dim=0)    
          ious[ious<=0.5]=0
          ious[ious>0.5] = gtlab_im[bbox_idx[ious>0.5]].float()
          # Encoding needs to be checked

          boxBall = torch.zeros(len(prop_im),4).to(device)
          boxBall[ious>0.5] = gtbbox_im_wh[bbox_idx[ious>0.5]].float()
          tx = (boxBall[:,0] - prop_im[:,0]) /  prop_im[:,2]
          ty = (boxBall[:,1] - prop_im[:,1]) /  prop_im[:,3]
          tw = torch.log(boxBall[:,2]/prop_im[:,2])
          th = torch.log(boxBall[:,3]/prop_im[:,3])

          boxen = torch.stack((tx, ty, tw, th)).T  
          boxen[ious<=0.5, :] = 0
   
          labels = torch.cat((labels, torch.unsqueeze(ious.int(),1)), 0)
          regressor_target = torch.cat((regressor_target, boxen), 0)
        return labels,regressor_target


    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################
        feature_vectors=[]
        for bz in range(len(proposals)):
          prop_im = proposals[bz].clone().to(device)
          xl, yl, xr, yr = prop_im[:,0], prop_im[:,1], prop_im[:,2], prop_im[:,3]
          w = xr- xl
          h  = yr - yl
          k = torch.clip(torch.floor(torch.Tensor([4]).to(device) + torch.log2(torch.sqrt(w * h).to(device) / 224)),2,5).int()  # k = torch.clip(torch.floor(4+torch.log2(torch.sqrt(w*h)/224)),2,5).int()

          s_x = 800/self.feature_sizes[k.cpu().numpy()-2,0]
          s_y = 1088/self.feature_sizes[k.cpu().numpy()-2,1]

          s_x = torch.from_numpy(s_x).to(device)
          s_y = torch.from_numpy(s_y).to(device)

          
          prop_im[:,0] = prop_im[:,0] / s_x 
          prop_im[:,1] = prop_im[:,1] / s_y
          prop_im[:,2] = prop_im[:,2] / s_x
          prop_im[:,3] = prop_im[:,3] / s_y

          for i in range(len(k)):
            input = torch.unsqueeze(fpn_feat_list[k[i]-2][bz], 0) # (1, 256, h, w)   
            roi_alig  = torchvision.ops.roi_align(input, [prop_im[i].reshape(1,-1)], output_size=P, sampling_ratio=-1)  # 1, 256, P, P

            feature_vectors.append(torch.flatten(roi_alig))
              
        feature_vectors = torch.vstack(feature_vectors) # total_proposals, 256*P*P

        return feature_vectors


    def training_step(self, batch, batch_idx):
      images, labels, boxes, masks, indexes = batch
      images = images.to(device)
      backout = backbone(images.float())
      im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
      rpnout = rpn(im_lis, backout)

      keep_topK = 200
      proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]

      fpn_feat_list= list(backout.values())
      gt_lab, gt_box = self.create_ground_truth(proposals,labels,boxes)
      
      feature_vectors = self.MultiScaleRoiAlign(fpn_feat_list,proposals)

      class_logits, box_pred= self(feature_vectors.detach().to(device))

      l1, l2 = self.compute_loss(class_logits,box_pred,gt_lab,gt_box)
      loss = l1+l2

      self.closs_sum = self.closs_sum + l1.item()
      self.rloss_sum = self.rloss_sum + l2.item()
      self.loss_sum = self.loss_sum + loss.item()

      self.log("train_loss", loss, prog_bar = True)
      if batch_idx==(batch_tr-1):
        self.epoch_loss.append(self.loss_sum/batch_tr)
        self.epoch_closs.append(self.closs_sum/batch_tr)
        self.epoch_rloss.append(self.rloss_sum/batch_tr)
        print("Training")
        print("total loss: ",self.epoch_loss)
        print("closs: ", self.epoch_closs)
        print("rloss: ", self.epoch_rloss)
        self.loss_sum=0
        self.closs_sum=0
        self.rloss_sum=0
      return loss

    def validation_step(self, batch, batch_idx):
      images, labels, boxes, masks, indexes = batch
      images = images.to(device)
      backout = backbone(images.float())
      im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
      rpnout = rpn(im_lis, backout)

      keep_topK = 200
      proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]

      fpn_feat_list= list(backout.values())
      gt_lab, gt_box = self.create_ground_truth(proposals,labels,boxes)
      
      feature_vectors = self.MultiScaleRoiAlign(fpn_feat_list,proposals)
      class_logits, box_pred= self(feature_vectors.detach().to(device))

      l1, l2 = self.compute_loss(class_logits,box_pred,gt_lab,gt_box)
      loss = l1+l2

      self.closs_sum = self.closs_sum + l1.item()
      self.rloss_sum = self.rloss_sum + l2.item()
      self.loss_sum = self.loss_sum + loss.item()
      
      self.log("val_loss", loss, prog_bar = True)

      if batch_idx==(batch_val-1):
        self.val_epoch_loss.append(self.loss_sum/batch_val)
        self.val_epoch_closs.append(self.closs_sum/batch_val)
        self.val_epoch_rloss.append(self.rloss_sum/batch_val)
        print("Validation")
        print("total loss: ", self.val_epoch_loss)
        print("closs: ", self.val_epoch_closs)
        print("rloss: ", self.val_epoch_rloss)
        self.loss_sum=0
        self.closs_sum=0
        self.rloss_sum=0
  
      return loss

    def configure_optimizers(self):
      optim = torch.optim.Adam(self.parameters(), lr=0.0007)
      lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [7, 12], gamma=0.1)
      return {"optimizer": optim, "lr_scheduler": lr_scheduler}

# This function does the post processing for the results of the Box Head for a batch of images
    # Use the proposals to distinguish the outputs from each image
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
    def postprocess_detections(self, class_logits, box_pred, proposals, conf_thresh=0.5, keep_num_preNMS=20, keep_num_postNMS=2):
      sortedboxes, sortedconf, sorted_labels = self.topkplotter(class_logits, box_pred, proposals, conf_thresh=0.5, keep_num_preNMS=keep_num_preNMS)
      if type(sortedboxes) == int:
        return 0,0,0

      scores = MatrixNMS(sortedboxes.to(device), sortedconf.to(device))
      top_scores, top5kidx = torch.sort(scores, descending=True)
      top_scores =top_scores[0 : keep_num_postNMS]
      topboxes = sortedboxes[top5kidx][0 : keep_num_postNMS]
      toplab = sorted_labels[top5kidx][0 : keep_num_postNMS]
      return top_scores, topboxes, toplab

    def non_max_suppression(self, bbox, score):
      prs= score
      all_lab= bbox
      for i in range(len(all_lab)):
        tar1= all_lab[i]
        for j in range(i+1,len(all_lab)):
          tar2=all_lab[j]
          iou=self.IoU_NMS(tar2,tar1)
          if iou>=0.3:
            if prs[i]>prs[j]:
              all_lab[j,:]=0
              prs[j] = 0
            else:
              all_lab[i,:]=0 
              prs[i] = 0
      return prs, all_lab

    def IoU_NMS(self, tar1, tar2): 
      xl_int = torch.maximum(tar1[0], tar2[0])
      yl_int = torch.maximum(tar1[1], tar2[1])
      xr_int = torch.minimum(tar1[2], tar2[2])
      yr_int = torch.minimum(tar1[3], tar2[3])
      Area_of_int = torch.maximum(torch.Tensor([0]).to(device), xr_int - xl_int) * torch.maximum(torch.Tensor([0]).to(device), yr_int - yl_int)
      Area_of_union = (torch.abs(tar1[2] - tar1[0]) * torch.abs(tar1[3] - tar1[1])) + (torch.abs(tar2[2] - tar2[0]) * torch.abs(tar2[3] - tar2[1])) - Area_of_int
      return Area_of_int/(Area_of_union+1e-13)

    def topkplotter(self, class_logits, box_pred, proposals, conf_thresh=0.5, keep_num_preNMS=500, map_eval=False):
        cmax, labelsno = torch.max(class_logits, dim=1)
        idxs = labelsno.nonzero()
        i = labelsno[idxs] - 1
        if len(i)==0:
          return 0,0,0

        encoded_boxes =  []
        for num in range(len(i)):
          lev = i[num]
          encoded_boxes.append(box_pred[num, lev*4:(lev*4)+4])

        encoded_boxes =  torch.stack((encoded_boxes))
        propcoder = torch.squeeze(torch.vstack((proposals))[idxs,:])
        dec_boxes = output_decodingd(encoded_boxes,propcoder,i+1).to(device)
        conf = cmax[idxs]

        sortedconf, sort_idxs = torch.sort(conf.T, descending=True)
        sorted_labels= i[sort_idxs] +1
        sortedboxes=dec_boxes[sort_idxs, :]

        fin_boxes = []
        confe =[]
        labs = []

        for ou in range(len(sortedboxes[0])):
          if (sortedboxes[0][ou]>0).sum() ==4:
            fin_boxes.append(sortedboxes[0][ou])
            confe.append(sortedconf[0][ou])
            labs.append(sorted_labels[0][ou])

        sortedboxes = torch.stack((fin_boxes))  
        sortedconf = torch.stack((confe))
        sorted_labels = torch.stack((labs))

        return sortedboxes[:keep_num_preNMS, :], sortedconf[:keep_num_preNMS], sorted_labels[:keep_num_preNMS, :]

      
    def loss_clas(self,gt_clas,out_clas):
      loss_c = self.celoss(out_clas, gt_clas)
      return loss_c 

    def loss_reg(self,gt_clas,gt_coord, out_clas, out_coord):
      loss_r = 0
      for i in range(3):
        idxs = (gt_clas==i+1).nonzero()[:,0]
        pos_out_r = out_coord[idxs, i*4:(i*4)+4]
        pos_target_coord = gt_coord[idxs, :]
        loss_r=  loss_r + self.smoothl1(pos_out_r, pos_target_coord)
      return loss_r

    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=10,effective_batch=32):
      nobag_idx = (labels[:,0]).nonzero()
      bag_idx = (labels[:,0] == 0).nonzero()

      if len(nobag_idx)>(3*effective_batch/4):
        rand_idx = torch.randperm(len(nobag_idx))[:int(3*effective_batch/4)]
        sampled_nobag_idx = nobag_idx[rand_idx]
        rand_idx = torch.randperm(len(bag_idx))[:int(effective_batch/4)]
        sampled_bag_idx = bag_idx[rand_idx]
      else:
        sampled_nobag_idx = nobag_idx
        rand_idx = torch.randperm(len(bag_idx))[:effective_batch - len(nobag_idx)]
        sampled_bag_idx = bag_idx[rand_idx]
      
      out_clas = torch.squeeze(torch.cat((class_logits[sampled_nobag_idx,:], class_logits[sampled_bag_idx,:]),0))
      gt_clas = torch.squeeze(torch.cat((labels[sampled_nobag_idx,0], labels[sampled_bag_idx,0]),0))

      loss_class =  self.loss_clas(gt_clas.long() ,out_clas)

      out_coord = torch.squeeze(box_preds[sampled_nobag_idx,:])
      gt_coord = torch.squeeze(regression_targets[sampled_nobag_idx,:])
      out_clas = torch.squeeze(class_logits[sampled_nobag_idx,:])
      gt_clas = torch.squeeze(labels[sampled_nobag_idx,0])
      
      loss_regr = l * self.loss_reg(gt_clas,gt_coord, out_clas, out_coord)
      return loss_class, loss_regr

    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors, eval=False):
        x = self.intermediate_layer(feature_vectors)
        class_logits = self.classhead[0](x)

        if eval:
          class_logits = self.softmax(class_logits)

        box_pred = self.reghead[0](x)
        return class_logits, box_pred

def load_datset(viz):
  labels_og = np.load("hw3_mycocodata_labels_comp_zlib.npy", allow_pickle=True, encoding='latin1')
  bbox_og = np.load("hw3_mycocodata_bboxes_comp_zlib.npy", allow_pickle=True, encoding='latin1')
  imgs = h5_to_npy("hw3_mycocodata_img_comp_zlib.h5") 
  masks_og = h5_to_npy("hw3_mycocodata_mask_comp_zlib.h5")

  # Testing if Files have been sucessfully loaded
  if viz:
    c=['none', 'green', 'blue','yellow']
    counter = []
    for l in labels_og:
      counter.append(len(l))
    for i in range(5):
      img_o=np.transpose(imgs[i], axes=(1,2,0))
      plt.imshow(img_o)
      maskstart= int(np.sum(counter[:i]))
      for num in range(maskstart, maskstart+len(labels_og[i]),1):
        plt.imshow(masks_og[num], cmap=ListedColormap([c[0], c[labels_og[i][num-maskstart]]]), alpha=.3)
        xl, yl, xr, yr = bbox_og[i][num-maskstart]
        plt.gca().add_patch(Rectangle((xl,yl),(xr-xl),(yr-yl),linewidth=1,edgecolor='r',facecolor='none'))
      plt.show()
  return labels_og, bbox_og, imgs, masks_og

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, bbox, masks):
        self.x_scale = ( 800 / 300)
        self.y_scale = ( 1066 / 400)
        self.imgs_data = data
        self.labels_data = labels
        self.bbox_data = bbox
        self.mask_data = []

        counter = []
        for l in labels:
          counter.append(len(l))
        for i in range(len(data)):
          maskstart= int(np.sum(counter[:i]))
          temp=[]
          for num in range(maskstart, maskstart+len(labels[i]),1):
            temp.append(masks[num])
          self.mask_data.append(temp)

        self.transform=transforms.Compose([transforms.Resize((800,1066)), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), transforms.Pad((11,0))])
        self.transform_mask=transforms.Compose([transforms.Resize((800,1066)), transforms.Pad((11,0))])

    def __getitem__(self, index):
        image = self.transform(torch.from_numpy(self.imgs_data[index,:,:,:].astype(float)/255))
        old_mask=self.mask_data[index]

        masks=[]
        for i in range(len(old_mask)):
          masks.append(self.transform_mask(torch.from_numpy((old_mask[i].reshape(1,300,400)).astype(float))))

        label=self.labels_data[index]
        bbox=np.zeros(self.bbox_data[index].shape)
        bbox[:,0]= self.bbox_data[index][:,0]*self.x_scale + 11
        bbox[:,2]= self.bbox_data[index][:,2]*self.x_scale + 11
        bbox[:,1]= self.bbox_data[index][:,1]*self.y_scale
        bbox[:,3]= self.bbox_data[index][:,3]*self.y_scale
        bbox= torch.from_numpy(bbox)
        label= torch.from_numpy(label)
        return image, label, bbox, masks, index
        
    def __len__(self):
        return len(self.imgs_data)
    
    def collate_fn(self,batch):
        images, labels, bounding_boxes, masks, indexs = list(zip(*batch))
        return torch.stack(images), labels, bounding_boxes, masks, indexs

def h5_to_npy(loc):
  file= h5py.File(loc,'r') 
  key = list(file.keys())[0]
  # data = list(file[key])
  data= file[key][()]
  return data


class MaskHead(pl.LightningModule):
    def __init__(self,Classes=3,P=14):
        super(MaskHead, self).__init__()
        self.C=Classes
        self.P=P
        self.loss_sum=0
        self.epoch_loss=[]
        self.val_epoch_loss=[]
        self.bceloss = nn.BCELoss(reduction ='sum')
        self.mask_conv_layers =  nn.Sequential(
        # Convolution 1
        nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding='same'),
        nn.ReLU(inplace=True),   

        # Convolution 2
        nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding='same'),
        nn.ReLU(inplace=True),   

        # Convolution 3
        nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding='same'),
        nn.ReLU(inplace=True),   

        # Convolution 4
        nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding='same'),
        nn.ReLU(inplace=True),
        
        nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, self.C, kernel_size=(1,1), stride=1),
        nn.Sigmoid()
        )

        boxmod = BoxHead.load_from_checkpoint(checkpoint_path='/content/drive/My Drive/680/hw4B/model_seedless/epoch=18_Model.ckpt')
        self.boxmod =  boxmod.to(device)

    # This function does the pre-prossesing of the proposals created by the Box Head (during the training of the Mask Head)
    # and create the ground truth for the Mask Head
    #
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)  ([t_x,t_y,t_w,t_h])
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       masks: list:len(bz){(n_obj,800,1088)}
    #       IOU_thresh: scalar (threshold to filter regressed with low IOU with a bounding box)
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
    #       gt_masks: list:len(bz){(post_NMS_boxes_per_image,2*P,2*P)}
    def preprocess_ground_truth_creation(self, class_logits, box_pred, proposals, gt_labels,bbox ,masks , IOU_thresh=0.5, keep_num_preNMS=2000, keep_num_postNMS=100):
        cmax, labelsno = torch.max(class_logits, dim=1)
        idxs = labelsno.nonzero()
        i = labelsno[idxs] - 1
        if len(i)==0:
          return torch.Tensor([0]),torch.Tensor([[0,0,0,0]]),torch.Tensor([[0]]),torch.zeros((1,28,28))
        encoded_boxes =  []
        for num in range(len(i)):
          lev = i[num]
          encoded_boxes.append(box_pred[num, lev*4:(lev*4)+4])

        encoded_boxes =  torch.stack((encoded_boxes))
        propcoder = torch.squeeze(torch.vstack((proposals))[idxs,:])
        dec_boxes = output_decodingd(encoded_boxes,propcoder,i+1).to(device)
        conf = cmax[idxs]
        
        ious_all =  torchvision.ops.box_iou(dec_boxes, bbox)
        ioumax, labgt = torch.max(ious_all, dim=1)
        idx_box = ioumax>0.5
        gt_masks =  masks[labgt]
        gt_bbox =  bbox[labgt]
        gtmasks = []
        for all in range(len(gt_bbox)):
          mask = gt_masks[all]
          box = gt_bbox[all].int()
          crop = torch.unsqueeze(mask[box[1]: box[3], box[0]: box[2]],0)
          interp = nn.functional.interpolate(torch.unsqueeze(crop,0), (28, 28), mode='bilinear')
          gtmasks.append(torch.squeeze(interp))
        gt_masks = torch.stack(gtmasks)

        dec_boxes = dec_boxes[idx_box,:][:keep_num_preNMS]
        conf = conf[idx_box][:keep_num_preNMS]
        labels = i[idx_box][:keep_num_preNMS]

        sortedconf, sort_idxs = torch.sort(conf.T, descending=True)
        sorted_labels= (labels[sort_idxs] +1)[0]

        sortedboxes= dec_boxes[sort_idxs, :][0]
        scores = MatrixNMS(sortedboxes.to(device), sortedconf.to(device))
        top_scores, top5kidx = torch.sort(scores, descending=True)
        top_scores = top_scores[0 : keep_num_postNMS]
        topboxes = sortedboxes[top5kidx][0 : keep_num_postNMS]
        toplab = sorted_labels[top5kidx][0 : keep_num_postNMS]
        gt_masks = gt_masks[top5kidx][0 : keep_num_postNMS]

        if len(torch.squeeze(topboxes))==0:
          return torch.Tensor([0]).to(device),torch.Tensor([[0,0,0,0]]).to(device),torch.Tensor([[0]]).to(device),torch.zeros((1,28,28)).to(device)
        return top_scores[0], topboxes[0], toplab[0], gt_masks[0]

    def trainprocess(self, images, gt_labels, bbox, masks):
      backout = backbone(images.float())
      im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
      rpnout = rpn(im_lis, backout)
      proposals=[proposal[0:1000,:] for proposal in rpnout[0]]
      fpn_feat_list= list(backout.values())
      feature_vectors = self.boxmod.MultiScaleRoiAlign(fpn_feat_list,proposals)
      class_logits, box_pred= self.boxmod(feature_vectors.detach().to(device))
      
      confs, boxes, labs, gt_masks = self.preprocess_ground_truth_creation(class_logits, box_pred, proposals, gt_labels, bbox[0].to(device)  , torch.stack(masks[0]).to(device)[:,0])
      new_feature_vectors = self.boxmod.MultiScaleRoiAlign(fpn_feat_list,[boxes], P=14)
  
      maskpred = self(new_feature_vectors.detach().to(device))
      return confs, boxes, labs, maskpred, gt_masks

    def training_step(self, batch, batch_idx):
      images, labels, boxes, masks, indexes = batch
      images = images.to(device)
      finconfs, finboxes, finlabs, maskpred, gt_masks = self.trainprocess(images, labels, boxes, masks)
      loss = self.compute_loss(maskpred.to(device),finlabs.to(device),gt_masks.to(device))
      self.loss_sum = self.loss_sum + loss.item()

      self.log("train_loss", loss, prog_bar = True)
      if batch_idx==(batch_tr-1):
        self.epoch_loss.append(self.loss_sum/batch_tr)
        print("Training")
        print("Mask loss: ",self.epoch_loss)
        self.loss_sum=0

      return loss

    def configure_optimizers(self):
      optim = torch.optim.Adam(self.parameters(), lr=0.0007)
      lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [7, 12, 20, 27], gamma=0.1)
      return {"optimizer": optim, "lr_scheduler": lr_scheduler}

    def validation_step(self, batch, batch_idx):
      images, labels, boxes, masks, indexes = batch
      images = images.to(device)
      finconfs, finboxes, finlabs, maskpred, gt_masks = self.trainprocess(images, labels, boxes, masks)
      loss = self.compute_loss(maskpred.to(device),finlabs.to(device),gt_masks.to(device))
      self.loss_sum = self.loss_sum + loss.item()

      self.log("test_loss", loss, prog_bar = True)
      if batch_idx==(batch_val-1):
        self.val_epoch_loss.append(self.loss_sum/batch_val)
        print("Validation")
        print("Mask loss: ",self.val_epoch_loss)
        self.loss_sum=0
        
      return loss

    # Compute the total loss of the Mask Head
    # Input:
    #      mask_output: (total_boxes,C,2*P,2*P)
    #      labels: (total_boxes)
    #      gt_masks: (total_boxes,2*P,2*P)
    # Output:
    #      mask_loss
    def compute_loss(self,mask_output,labels,gt_masks):
        mask_loss = 0 
        for c in range(3):
          c_mask = mask_output[:,c]
          c_gt = gt_masks.clone()
          c_gt[torch.squeeze(labels!=c+1)]=0
          mask_loss = mask_loss + self.bceloss(c_mask, c_gt.float())/len(c_gt)
        return mask_loss/3

    # Forward the pooled feature map Mask Head
    # Input:
    #        features: (total_boxes, 256,P,P)
    # Outputs:
    #        mask_outputs: (total_boxes,C,2*P,2*P)
    def forward(self, features):
        mask_outputs= self.mask_conv_layers(features.reshape(features.shape[0],256,self.P,self.P))
        return mask_outputs
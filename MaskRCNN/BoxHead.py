import numpy as np
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torchvision.models.detection.image_list import ImageList
import warnings
warnings.filterwarnings('ignore')
torch.set_printoptions(linewidth=100)

np.set_printoptions(linewidth=100)
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

from utils import *

def Resnet50Backbone(checkpoint_file=None, device="cuda", eval=True):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)

    if eval == True:
        model.eval()

    model.to(device)

    resnet50_fpn = model.backbone

    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)

        resnet50_fpn.load_state_dict(checkpoint['backbone'])

    return resnet50_fpn

def pretrained_models_680(checkpoint_file,eval=True, useours=False):
  if useours:
    backbone = Resnet50Backbone().to(device)
    rpn = RPNHead.load_from_checkpoint(checkpoint_path=checkpoint_file).to(device)
    if(eval):
      backbone.eval()
      rpn.eval()

  else:
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

class RPNHead(pl.LightningModule):
    def __init__(self, num_anchors=3, in_channels=256,
                 anchors_param=dict(ratio=[[1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2]],
                                    scale=[32, 64, 128, 256, 512],
                                    grid_size=[(200, 272), (100, 136), (50, 68), (25, 34), (13, 17)],
                                    stride=[4, 8, 16, 32, 64])):
        super(RPNHead,self).__init__()
        self.loss_sum=0
        self.closs_sum=0
        self.rloss_sum=0
        self.epoch_loss=[]
        self.val_epoch_loss=[]
        self.epoch_closs=[]
        self.val_epoch_closs=[]
        self.epoch_rloss=[]
        self.val_epoch_rloss=[]
        self.lr =  0.01

        self.bceloss = nn.BCELoss(reduction ='sum')
        self.smoothl1 = nn.SmoothL1Loss(reduction ='sum')
          
        # TODO  Define Intermediate Layer
        self.intermediate_layer =  nn.Sequential(
            # Convolution 1
            nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),   
        )

        self.classhead= nn.Sequential(nn.Conv2d(in_channels, num_anchors, kernel_size=(1,1), stride=1, padding='same'),  nn.Sigmoid())

        self.reghead= nn.Sequential(nn.Conv2d(in_channels, num_anchors*4, kernel_size=(1,1), stride=1, padding='same'),  nn.Sigmoid())
    
        #  find anchors
        self.anchors_param=anchors_param
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict= {}
        self.backbone = Resnet50Backbone()
        
    # Forward each level of the FPN output through the intermediate layer and the RPN heads
    # Input:
    #       X: list:len(FPN){(bz,256,grid_size[0],grid_size[1])}
    # Ouput:
    #       logits: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       bbox_regs: list:len(FPN){(bz,4*num_anchors, grid_size[0],grid_size[1])}
    def forward(self, images):
        feature_pyramid = [v.detach() for v in self.backbone(images).values()] # this has strides [4,8,16,32,64]
        logits=[]
        bbox_regs=[]
        for i in range(len(feature_pyramid)):
          pyram= feature_pyramid[i]
          intermediate= self.intermediate_layer(pyram)

          class_out= self.classhead(intermediate)
          reg_out= self.reghead(intermediate)

          logits.append(class_out)
          bbox_regs.append(reg_out)
        return logits, bbox_regs

    # This function creates the anchor boxes for all FPN level
    # Input:
    #       aspect_ratio: list:len(FPN){list:len(number_of_aspect_ratios)}
    #       scale:        list:len(FPN)
    #       grid_size:    list:len(FPN){tuple:len(2)}
    #       stride:        list:len(FPN)
    # Output:
    #       anchors_list: list:len(FPN){(grid_size[0]*grid_size[1]*num_anchors,4)}
    def create_anchors(self, aspect_ratio, scale, grid_size, stride):
        anchors_list = []
        for lev in range(len(scale)):
          anchors_list.append(self.create_anchors_single(aspect_ratio[lev], scale[lev], grid_size[lev], stride[lev]))
        return anchors_list

    # This function creates the anchor boxes for one FPN level
    # Input:
    #      aspect_ratio: list:len(number_of_aspect_ratios)
    #      scale: scalar
    #      grid_size: tuple:len(2)
    #      stride: scalar
    # Output:
    #       anchors: (grid_size[0]*grid_size[1]*num_acnhors,4)
    def create_anchors_single(self, aspect_ratio, scale, grid_sizes, stride):
        anchors=[]
        for aspec in aspect_ratio:
          h= scale/(aspec**0.5)
          w= aspec * h
          wst = torch.ones((grid_sizes[0], grid_sizes[1])) * w
          hst = torch.ones((grid_sizes[0], grid_sizes[1])) * h

          xs = torch.linspace(0, grid_sizes[0]-1, steps=grid_sizes[0])
          ys = torch.linspace(0, grid_sizes[1]-1, steps=grid_sizes[1])
          x_, y_ = torch.meshgrid(ys, xs, indexing='xy') 
          x = x_ * stride
          y = y_ * stride
          cx =  x+8
          cy =  y+8

          single_anchors = torch.stack((cx, cy, wst, hst))
          single_anchors = torch.permute(single_anchors, (1, 2, 0))
          assert single_anchors.shape == (grid_sizes[0] , grid_sizes[1],4)

          anchors.append(single_anchors)

        anchors = torch.stack((anchors))   
        assert anchors.shape == (len(aspect_ratio),grid_sizes[0], grid_sizes[1],4)

        return anchors.to(device)  

    def get_anchors(self):
        return self.anchors

    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)
    def postprocessImg(self,mat_clas,mat_coord, IOU_thresh,keep_num_preNMS, keep_num_postNMS):
      keep_num_preNMS = 2000
      # nms_prebox = 1
      flatten_bbox,flatten_clas,flatten_anchors=output_flattening(mat_coord,mat_clas,self.get_anchors()) 
      decoded_coord=output_decoding(flatten_bbox,flatten_anchors).to(device)

      # top k boxes - sort thres_clas,  keep top k
      sorted_score, sorted_idx = torch.sort(flatten_clas, descending = True)
      sorted_idx = sorted_idx[:keep_num_preNMS].to(device)
      topK_score = sorted_score[sorted_idx]
      topK_bbox = decoded_coord[sorted_idx]
      
      # nms_clas, nms_prebox = self.non_max_suppression(topK_bbox, topK_score)
      scores = MatrixNMS(topK_bbox.to(device), topK_score.to(device))
      top_scores, top5kidx = torch.sort(scores, descending=True)
      top_scores =top_scores[0 : keep_num_postNMS]
      topboxes = topK_bbox[0 : keep_num_postNMS]
      return top_scores, topboxes

class BoxHead(pl.LightningModule):
    def __init__(self,batch_tr, batch_val, Classes=3,P=7):
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
        self.batch_tr = batch_tr
        self.batch_val = batch_tr

        checkpoint_ours= '/content/drive/My Drive/680/Final Project/model_change/epoch=21_Model.ckpt'
        # pretrained_path='checkpoint680.pth'
        # self.backbone, self.rpn = pretrained_models_680(pretrained_path, useours=False)
        self.useours=True

        self.backbone, self.rpn = pretrained_models_680(checkpoint_ours, useours=True)

        self.nreg = 0
        self.celoss=nn.CrossEntropyLoss()
        self.smoothl1 = nn.SmoothL1Loss(reduction ='sum')
        self.feature_sizes = np.array([ [200, 272], [100, 136], [50, 68], [25, 34], [13, 17]])
        self.intermediate_layer =  nn.Sequential(
            nn.Linear(256 * P * P, 1024), 
            nn.ReLU(),

            nn.Linear(1024,1024), 
            nn.ReLU()
            )
            
        self.classhead= nn.ModuleList()    
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
        feature_vectors=[]
        for bz in range(len(proposals)):
          prop_im = proposals[bz].clone()    
          xl, yl, xr, yr = prop_im[:,0], prop_im[:,1], prop_im[:,2], prop_im[:,3]
          w = xr- xl
          h  = yr - yl
          k = torch.clip(torch.floor(torch.Tensor([4]).to(device) + torch.log2(torch.sqrt(w * h) / 224)),2,5).int()  # k = torch.clip(torch.floor(4+torch.log2(torch.sqrt(w*h)/224)),2,5).int()

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
      backout = self.backbone(images.float())

      if self.useours:
        logits, reg_bbox = self.rpn.forward(images.float())
        _ , proposals= self.rpn.postprocessImg(logits,reg_bbox, 0.5, 2000, 1000)
        proposals = [proposals]
      
      else:
        im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
        rpnout = self.rpn(im_lis, backout)
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
      if batch_idx==(self.batch_tr-1):
        self.epoch_loss.append(self.loss_sum/self.batch_tr)
        self.epoch_closs.append(self.closs_sum/self.batch_tr)
        self.epoch_rloss.append(self.rloss_sum/self.batch_tr)
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
      backout = self.backbone(images.float())
      
      if self.useours:
        logits, reg_bbox = self.rpn.forward(images.float())
        _ , proposals= self.rpn.postprocessImg(logits,reg_bbox, 0.5, 2000, 1000)
        proposals = [proposals]
      
      else:
        im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
        rpnout = self.rpn(im_lis, backout)
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

      if batch_idx==(self.batch_val-1):
        self.val_epoch_loss.append(self.loss_sum/self.batch_val)
        self.val_epoch_closs.append(self.closs_sum/self.batch_val)
        self.val_epoch_rloss.append(self.rloss_sum/self.batch_val)
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
        sortedboxes=dec_boxes[sort_idxs.to(device), :]

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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from utils import *
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
seed = 17
torch.manual_seed(seed)

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



##Main Function
class RPNHead(pl.LightningModule):
    def __init__(self, batch_tr, batch_val, num_anchors=3, in_channels=256,
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
        self.batch_tr = batch_tr
        self.batch_val =  batch_val

        self.bceloss = nn.BCELoss(reduction ='sum')
        self.smoothl1 = nn.SmoothL1Loss(reduction ='sum')
          
        self.intermediate_layer =  nn.Sequential(
            # Convolution 1
            nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),   
        )

        self.classhead= nn.Sequential(nn.Conv2d(in_channels, num_anchors, kernel_size=(1,1), stride=1, padding='same'),  nn.Sigmoid())

        self.reghead= nn.Sequential(nn.Conv2d(in_channels, num_anchors*4, kernel_size=(1,1), stride=1, padding='same'),  nn.Sigmoid())
    
        #  Find anchors
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

    # This function creates the ground truth for a batch of images
    # Input:
    #      bboxes_list: list:len(bz){(number_of_boxes,4)}
    #      indexes: list:len(bz)
    #      image_shape: list:len(bz){tuple:len(2)}
    # Ouput:
    #      ground_clas: list:len(FPN){(bz,num_anchors,grid_size[0],grid_size[1])}
    #      ground_coord: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    def create_batch_truth(self, bboxes_list, indexes, image_shape):
        ground_clas = [[],[],[],[],[]]
        ground_coord= [[],[],[],[],[]]
        for i in range(len(bboxes_list)):
          gclass_i, gcoord_i = self.create_ground_truth(bboxes_list[i].to(device), indexes[i], self.anchors_param['grid_size'], self.anchors, image_shape) 
          for lev in range(len(gclass_i)):
            ground_clas[lev].append(gclass_i[lev])
            ground_coord[lev].append(gcoord_i[lev])

        for lev in range(len(ground_coord)):
          ground_clas[lev]  = torch.stack(ground_clas[lev])
          ground_coord[lev] = torch.stack(ground_coord[lev])
  
        return ground_clas, ground_coord

    # This function create the ground truth for one image for all the FPN levels
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset)
    #       grid_size:   list:len(FPN){tuple:len(2)}
    #       anchor_list: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
    # Output:
    #       all_ground_clas: list:len(FPN){(num_anchors,grid_size[0],grid_size[1])}
    #       all_ground_coord: list:len(FPN){(4*num_anchors,grid_size[0],grid_size[1])}
    def create_ground_truth(self, bboxes, index, grid_sizes, anchor_list, image_size):
        key = str(index)
        if key in self.ground_dict:
            ground_clas, ground_coord = self.ground_dict[key]
            return ground_clas, ground_coord
        
        all_ground_clas = []
        all_ground_coord = []
        for lev in range(len(grid_sizes)):
          grid_size = grid_sizes[lev]
          anchorsall = anchor_list[lev]
          anchorswhall= torch.permute(anchorsall, (0, 3, 1, 2)).to(device)
          anchorsall = xyxy_xywh(anchorsall, to_wh=False).to(device)
          
          fpn_ground_clas = []
          fpn_ground_coord = []

          for tpe in range(len(anchorsall)):
            anchorswh = anchorswhall[tpe]
            anchors =  anchorsall[tpe]
           
            ground_clas =  torch.zeros(grid_size[0],grid_size[1]).to(device)
            ground_coord =  torch.zeros(4,grid_size[0],grid_size[1]).to(device)
            idx0  = torch.unique((anchors<0).nonzero()[:,:2], dim=0)
            idx1  = (anchors[:,:,3]>800).nonzero()
            idx2  = (anchors[:,:,2]>1088).nonzero()
            idx = torch.unique(torch.vstack((idx0, idx1, idx2)), dim =0)
            ground_clas[idx[:,0], idx[:,1]] = -1

            ious = []
            idxs_noneg = torch.ones((grid_size[0],grid_size[1]), dtype=torch.bool).to(device)
            for nbo in range(len(bboxes)):
              boxB = bboxes[nbo].clone()
              temp = torchvision.ops.box_iou(torch.unsqueeze(boxB,0), anchors.reshape(-1,4)).reshape(grid_size[0],grid_size[1])
              ious.append(temp)
              idxs_noneg = torch.logical_and(idxs_noneg, torch.logical_and((temp>0.3), (temp<0.7)))
            ious = torch.stack(ious)
            
            ground_clas[idxs_noneg] = -1  

            for nbo in range(len(bboxes)): 
              boxB = bboxes[nbo].clone()
              w = boxB[2] - boxB[0]
              h = boxB[3] - boxB[1]
              cx = boxB[0] + w/2
              cy = boxB[1] + h/2
              boxB[:] = torch.Tensor([cx , cy, w, h]).to(device)

              boxBall = boxB.repeat(grid_size[1],grid_size[0],1).T.float()

              # Encoding
              tx = (boxBall[0,:, :] - anchorswh[0,:, :]) /  anchorswh[2,:, :]
              ty = (boxBall[1,:, :] - anchorswh[1,:, :]) /  anchorswh[3,:, :]
              tw = torch.log(boxBall[2,:, :]/anchorswh[2,:, :])
              th = torch.log(boxBall[3,:, :]/anchorswh[3,:, :])

              boxBall = torch.stack((tx, ty, tw, th))

              ground_clas[ious[nbo]>=0.7] = 1
              ground_coord[:, ious[nbo]>=0.7] = boxBall[:, ious[nbo]>=0.7]  
              maxiou = ious[nbo].max().item() - 0.0001
              ground_clas[ious[nbo]>=maxiou] = 1
              ground_coord[:, ious[nbo]>=maxiou] = boxBall[:, ious[nbo]>=maxiou]  
              ground_clas[ious[nbo]<0.3] = 0
              ground_coord[:, ious[nbo]<0.3] = boxBall[:, ious[nbo]<0.3]*0  

            ground_clas =  torch.unsqueeze(ground_clas, dim=0)
            fpn_ground_clas.append(ground_clas)
            fpn_ground_coord.append(ground_coord)
          
          all_ground_clas.append(torch.cat(fpn_ground_clas, 0))
          all_ground_coord.append(torch.cat(fpn_ground_coord, 0))
      
        return all_ground_clas, all_ground_coord

    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self, p_out, n_out):
      loss_c = self.bceloss(n_out, torch.zeros((n_out.shape)).to(device)) + self.bceloss(p_out, torch.ones((p_out.shape)).to(device))
      return loss_c / self.nreg 

    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self, pos_target_coord, pos_out_r):
      loss_r= self.smoothl1(pos_out_r, pos_target_coord)
      return loss_r / self.nreg 

    # Compute the total loss for the FPN heads
    # Input:
    #       clas_out_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       regr_out_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       targ_clas_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       targ_regr_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       l: weighting lambda between the two losses
    # Output:
    #       loss_c: scalar
    #       loss_r: scalar
    def compute_loss(self, clas_out, regr_out, targ_clas, targ_regr, l=5, effective_batch=150):
      pos_idx = (targ_clas == 1).nonzero()
      neg_idx = (targ_clas == 0).nonzero()

      if len(pos_idx)>(effective_batch/2):
        rand_idx = torch.randperm(len(pos_idx))[:int(effective_batch/2)]
        sampled_pos_idx = pos_idx[rand_idx]
        rand_idx = torch.randperm(len(neg_idx))[:int(effective_batch/2)]
        sampled_neg_idx = neg_idx[rand_idx]
      else:
        sampled_pos_idx = pos_idx
        rand_idx = torch.randperm(len(neg_idx))[:effective_batch - len(pos_idx)]
        sampled_neg_idx = neg_idx[rand_idx]

      p_out = clas_out[sampled_pos_idx]
      n_out = clas_out[sampled_neg_idx]
      self.nreg =  len(p_out) + len(n_out)
      loss_c = self.loss_class(p_out,n_out)

      pos_target_coord = torch.squeeze(targ_regr[sampled_pos_idx, :])
      pos_out_r = torch.squeeze(regr_out[sampled_pos_idx, :])
      loss_r = self.loss_reg(pos_target_coord,pos_out_r)
      return loss_c, l*loss_r

    def training_step(self, batch, batch_idx):
      images, labels, bbox, masks, index = batch

      logits, regout  = self.forward(images.to(device).float())
      gt,ground_coord=self.create_batch_truth(bbox,index,images.shape[-2:])
      
      regr_out, clas_out, _= output_flattening(regout, logits, self.get_anchors())
      targ_regr, targ_clas, _ = output_flattening(ground_coord, gt, self.get_anchors())

      l1, l2 = self.compute_loss(clas_out, regr_out, targ_clas, targ_regr)
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
      images, labels, bbox, masks, index = batch

      logits, regout  = self.forward(images.to(device).float())
      gt,ground_coord=self.create_batch_truth(bbox,index,images.shape[-2:])
      
      regr_out, clas_out, _= output_flattening(regout, logits, self.get_anchors())
      targ_regr, targ_clas, _ = output_flattening(ground_coord, gt, self.get_anchors())

      l1, l2 = self.compute_loss(clas_out, regr_out, targ_clas, targ_regr)
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
      optim = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
      lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [10, 20, 30], gamma=0.1)
      return {"optimizer": optim, "lr_scheduler": lr_scheduler}

    # Post process for the outputs for a batch of images
    # Input:
    #       out_c:  (bz,1,grid_size[0],grid_size[1])}
    #       out_r:  (bz,4,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinates of the boxes that the NMS kept)
    def topK_plotter(self,images, bbbox, topK=20):
      logits, reg_bbox  = self.forward(images.float())
      flatten_bbox, flatten_clas, flatten_anchors= output_flattening(reg_bbox, logits, self.get_anchors())
      decoded_coord=output_decoding(flatten_bbox,flatten_anchors).to(device)

      sorted_score, sorted_idx = torch.sort(flatten_clas, descending = True)
      sorted_idx = sorted_idx[:topK].to(device)
      topK_score = sorted_score[sorted_idx]
      topK_bbox = decoded_coord[sorted_idx]  
      # Plotting
      img=images[0].cpu().numpy()
      x_min = img.min(axis=(1, 2), keepdims=True)
      x_max = img.max(axis=(1, 2), keepdims=True)
      img = (img - x_min)/(x_max-x_min)
      fig,ax=plt.subplots(1,1)
      ax.imshow(np.transpose(img, axes=(1,2,0)))
      for elem in topK_bbox:
          coord=elem.cpu().detach().numpy()
          coord[0] =  np.clip(coord[0], 0, 1088)
          coord[2] =  np.clip(coord[2], 0, 1088)
          coord[1] =  np.clip(coord[1], 0, 800)
          coord[3] =  np.clip(coord[3], 0, 800)
          rect=Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color='r')
          ax.add_patch(rect)
      for box in bbbox:
        rect=Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color='b', linewidth=2)
        ax.add_patch(rect)

      plt.show()
      return topK_bbox

    def nms_plotter(self,images, bbbox, keep_num_preNMS, keep_num_postNMS):
      logits, reg_bbox = self.forward(images.float())
      nms_clas, nms_prebox= self.postprocessImg(logits,reg_bbox, 0.5, keep_num_preNMS, keep_num_postNMS)

      # Plotting Post NMS
      img=images[0].cpu().numpy()
      x_min = img.min(axis=(1, 2), keepdims=True)
      x_max = img.max(axis=(1, 2), keepdims=True)
      img = (img - x_min)/(x_max-x_min)
      fig,ax=plt.subplots(1,1)
      ax.imshow(np.transpose(img, axes=(1,2,0)))
      for elem in nms_prebox:
          coord=elem.detach().cpu().numpy()
          col='r'
          rect=Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
          ax.add_patch(rect)
      for box in bbbox:
        rect=Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color='b', linewidth=2)
        ax.add_patch(rect)
      plt.show()
      return nms_prebox
         
    def non_max_suppression(self, bbox, score):
        prs= score
        all_lab= bbox
        for i in range(len(all_lab)):
          tar1= torch.unsqueeze(all_lab[i],0)
          for j in range(i+1,len(all_lab)):
            tar2=torch.unsqueeze(all_lab[j],0)
            iou= torchvision.ops.box_iou(tar2, tar1)[0]
            if iou>=0.4:
              if prs[i]>prs[j]:
                all_lab[j,:]=0
                prs[j] = 0
              else:
                all_lab[i,:]=0 
                prs[i] = 0
        return prs, all_lab

    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)
    def postprocessImg(self,mat_clas,mat_coord, IOU_thresh,keep_num_preNMS, keep_num_postNMS):
      keep_num_preNMS = 2000
      flatten_bbox,flatten_clas,flatten_anchors=output_flattening(mat_coord,mat_clas,self.get_anchors()) 
      decoded_coord=output_decoding(flatten_bbox,flatten_anchors).to(device)

      sorted_score, sorted_idx = torch.sort(flatten_clas, descending = True)
      sorted_idx = sorted_idx[:keep_num_preNMS].to(device)
      topK_score = sorted_score[sorted_idx]
      topK_bbox = decoded_coord[sorted_idx]
      
      scores = MatrixNMS(topK_bbox.to(device), topK_score.to(device))
      top_scores, _ = torch.sort(scores, descending=True)
      top_scores =top_scores[0 : keep_num_postNMS]
      topboxes = topK_bbox[0 : keep_num_postNMS]
      return top_scores, topboxes

    def IoU_NMS(self, tar1, tar2): 
      xl_int = torch.maximum(tar1[0], tar2[0])
      yl_int = torch.maximum(tar1[1], tar2[1])
      xr_int = torch.minimum(tar1[2], tar2[2])
      yr_int = torch.minimum(tar1[3], tar2[3])
      Area_of_int = torch.maximum(torch.Tensor([0]).to(device), xr_int - xl_int) * torch.maximum(torch.Tensor([0]).to(device), yr_int - yl_int)
      Area_of_union = (torch.abs(tar1[2] - tar1[0]) * torch.abs(tar1[3] - tar1[1])) + (torch.abs(tar2[2] - tar2[0]) * torch.abs(tar2[3] - tar2[1])) - Area_of_int
      return Area_of_int/(Area_of_union+1e-13)

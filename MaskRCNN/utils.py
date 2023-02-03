from sklearn import metrics
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection.image_list import ImageList
import h5py
import warnings
warnings.filterwarnings('ignore')
torch.set_printoptions(linewidth=100)

np.set_printoptions(linewidth=100)
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


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

def vizualise_dataloader(images, labels, bbox, masks):
  c=['none', 'green', 'blue','red']
  cc=['none', 'g', 'b','r']
  cl=['none', 'Vehicle', 'Person','Animal']
  for i in range(len(images)):
    img, label, bbox_, mask_ = images[i], labels[i], bbox[i], masks[i]
    img=img.numpy()
    x_min = img.min(axis=(1, 2), keepdims=True)
    x_max = img.max(axis=(1, 2), keepdims=True)
    img = (img - x_min)/(x_max-x_min)
    plt.imshow(np.transpose(img, axes=(1,2,0)))
    for j in range(len(bbox_)):
      plt.imshow(np.squeeze(mask_[j].numpy()), cmap=ListedColormap(['none', c[label[j]]]), alpha=.3)
      xl, yl, xr, yr = bbox_[j].numpy()
      plt.gca().add_patch(Rectangle((xl,yl),(xr-xl),(yr-yl),linewidth=1,edgecolor=cc[label[j]],facecolor='none'))
      plt.text(xl+0.25, yl+0.25, cl[label[j]], color='y')

    plt.show()

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


def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function compute the IOU between two set of boxes 
def IOU(tar1, tar2):
    ##################################
    #TODO compute the IOU between the boxA, boxB boxes
    ##################################
    xl_int = torch.maximum(tar1[0], tar2[0])
    yl_int = torch.maximum(tar1[1], tar2[1])
    xr_int = torch.minimum(tar1[2], tar2[2])
    yr_int = torch.minimum(tar1[3], tar2[3])
    Area_of_int = torch.maximum(torch.Tensor([0]).to(device), xr_int - xl_int) * torch.maximum(torch.Tensor([0]).to(device), yr_int - yl_int)
    Area_of_union = (torch.abs(tar1[2] - tar1[0]) * torch.abs(tar1[3] - tar1[1])) + (torch.abs(tar2[2] - tar2[0]) * torch.abs(tar2[3] - tar2[1])) - Area_of_int
    return Area_of_int/(Area_of_union+1e-6)

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


# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
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

def output_decoding(flatten_out,flatten_anchors, device='cpu'):
    xa = flatten_anchors[:,0]
    ya = flatten_anchors[:,1]
    wa = flatten_anchors[:,2]
    ha = flatten_anchors[:,3]

    newcoords = torch.zeros(flatten_out.shape)
    tx  = flatten_out[:,0]
    ty  = flatten_out[:,1]
    tw  = flatten_out[:,2]
    th  = flatten_out[:,3]

    # Decoding
    cx = wa*tx + xa
    cy = ha*ty + ya
    w = wa* torch.exp(tw)
    h = ha* torch.exp(th)

    newcoords[:,0] = cx - w/2
    newcoords[:,1] = cy - h/2
    newcoords[:,2] = cx + w/2 
    newcoords[:,3] = cy + h/2
    return newcoords

def output_flattening(out_r, out_c, anchors):
    bz = len(out_r[0])
    flatten_regr = []
    flatten_clas = []
    flatten_anchors = []
    for lev in range(len(out_r)):
      out_rl = torch.permute(out_r[lev],(0,2,3,1))
      out_cl = torch.permute(out_c[lev],(0,2,3,1))
      anchorl = torch.permute(anchors[lev], (1, 2, 0, 3)).reshape(out_rl.shape[1], out_rl.shape[2],12)
      anchorl =  anchorl.repeat(bz,1,1,1)

      
      flat_clasl = out_cl.reshape(-1,3)
      flat_regl = out_rl.reshape(-1, 12)
      flat_ancl = anchorl.reshape(-1,12)

      for an in range(0,12,4):
        flatten_regr.append(flat_regl[:, an:an+4])
        flatten_anchors.append(flat_ancl[:, an:an+4])
        flatten_clas.append(flat_clasl[:, an//4])
      
    flatten_regr = torch.concat(flatten_regr, 0)
    flatten_clas = torch.concat(flatten_clas, 0)
    flatten_anchors = torch.concat(flatten_anchors, 0)
    return flatten_regr, flatten_clas, flatten_anchors


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

def plot_gt_boxes(backbone, rpn, test_loader, model, useours=True):
  keep_topK=50
  with torch.no_grad():
      for iter, batch in enumerate(test_loader, 0):
          images, labels, boxes, masks, indexes = batch
          images = images.to(device)
          backout = backbone(images.float())

          if useours:
            logits, reg_bbox = rpn.forward(images.float())
            _ , proposals= rpn.postprocessImg(logits,reg_bbox, 0.5, 2000, 1000)
            proposals = [proposals]
      
          else:
            im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
            rpnout = rpn(im_lis, backout)
            keep_topK = 200
            proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]         
          
          fpn_feat_list= list(backout.values())
          cod_lab, cod_box = model.create_ground_truth(proposals,labels,boxes)
          decod_box = output_decodingd(cod_box,torch.vstack((proposals)),cod_lab)
          
          feature_vectors = model.MultiScaleRoiAlign(fpn_feat_list,proposals)
          
          # class_logits, box_pred= boxHead.forward(feature_vectors.to(device))
          # l1, l2 = boxHead.compute_loss(class_logits,box_pred,cod_lab,cod_box)

          print("For the proposals We have a list containing "+str(len(proposals))+" tensors")
          print("Each one with shape "+str(proposals[0].shape))
          print("")
          print("For the features we have a list of features for each FPN level with shapes")
          for feat in fpn_feat_list:
              print(feat.shape)
          print("==========================================================================")
          # Visualization of the proposals
          for i in range(1):
              img_squeeze=images[i,:,:,:].to('cpu').numpy()
              x_min = img_squeeze.min(axis=(1, 2), keepdims=True)
              x_max = img_squeeze.max(axis=(1, 2), keepdims=True)
              img_squeeze = (img_squeeze - x_min)/(x_max-x_min)
              
              fig,ax=plt.subplots(1,1)
              ax.imshow(np.transpose(img_squeeze, axes=(1,2,0)))
              lablis = cod_lab[(i*keep_topK):(i*keep_topK)+keep_topK]
              boxes = decod_box[(i*keep_topK):(i*keep_topK)+keep_topK]
              propboxes_rest = torch.vstack((proposals))[(i*keep_topK):(i*keep_topK)+keep_topK]
              propboxes_rest[(lablis!=0)[:,0],:]=0

              propboxes = torch.vstack((proposals))[(i*keep_topK):(i*keep_topK)+keep_topK]
              propboxes[(lablis==0)[:,0],:]=0
              for no in range(len(boxes)):
                  box=boxes[no].view(-1).cpu()
                  propbox= propboxes[no].view(-1).cpu()
                  all_box = propboxes_rest[no].view(-1).cpu()
                  rect=Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color='r')
                  ax.add_patch(rect)
                  rect=Rectangle((propbox[0],propbox[1]),propbox[2]-propbox[0],propbox[3]-propbox[1],fill=False,color='b')
                  ax.add_patch(rect)
                  rect=Rectangle((all_box[0],all_box[1]),all_box[2]-all_box[0],all_box[3]-all_box[1],fill=False,color='b')
                  ax.add_patch(rect)
              plt.show()

              fig,ax=plt.subplots(1,1)
              ax.imshow(np.transpose(img_squeeze, axes=(1,2,0)))
              lablis = cod_lab[(i*keep_topK):(i*keep_topK)+keep_topK]
              boxes = decod_box[(i*keep_topK):(i*keep_topK)+keep_topK]
              propboxes_rest = torch.vstack((proposals))[(i*keep_topK):(i*keep_topK)+keep_topK]
              propboxes_rest[(lablis!=0)[:,0],:]=0

              propboxes = torch.vstack((proposals))[(i*keep_topK):(i*keep_topK)+keep_topK]
              propboxes[(lablis==0)[:,0],:]=0
              for no in range(len(boxes)):
                  box=boxes[no].view(-1).cpu()
                  propbox= propboxes[no].view(-1).cpu()
                  all_box = propboxes_rest[no].view(-1).cpu()
                  rect=Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color='r')
                  ax.add_patch(rect)
                  rect=Rectangle((propbox[0],propbox[1]),propbox[2]-propbox[0],propbox[3]-propbox[1],fill=False,color='g')
                  ax.add_patch(rect)
                  # rect=Rectangle((all_box[0],all_box[1]),all_box[2]-all_box[0],all_box[3]-all_box[1],fill=False,color='b')
                  # ax.add_patch(rect)
              plt.show()
              print("==========================================================================")

          break

def plot_topK(ktimes, backbone, rpn, test_loader, model):
  device ='cuda'
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
  for j in range(ktimes):
    for iter, batch in enumerate(test_loader, 0):
      images, labels, boxes, masks, indexes = batch
      break
    images = images.to(device)

    backout = backbone(images.float())
    # if useours:
    # logits, reg_bbox = rpn.forward(images.float())
    # _ , proposals= rpn.postprocessImg(logits,reg_bbox, 0.5, 2000, 1000)
    # proposals = [proposals]

    im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
    rpnout = rpn(im_lis, backout)
    keep_topK = 200
    proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]  
    fpn_feat_list= list(backout.values())

    feature_vectors = model.MultiScaleRoiAlign(fpn_feat_list,proposals)
    class_logits, box_pred= model.forward(feature_vectors.to(device), eval=True)

    sortedboxes, sortedconf, sorted_labels = model.topkplotter(class_logits, box_pred, proposals, conf_thresh=0.5, keep_num_preNMS=20)
    img_squeeze=images[0,:,:,:].to('cpu').numpy()
    x_min = img_squeeze.min(axis=(1, 2), keepdims=True)
    x_max = img_squeeze.max(axis=(1, 2), keepdims=True)
    img_squeeze = (img_squeeze - x_min)/(x_max-x_min)

    fig,ax=plt.subplots(1,1)
    ax.imshow(np.transpose(img_squeeze, axes=(1,2,0)))
    c=['none', 'green', 'blue','yellow']

    if type(sortedboxes) == int:
      continue

    for no in range(len(sortedboxes)):
        box=sortedboxes[no].detach().cpu().numpy()
        lab= sorted_labels[no].item()
        rect=Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color=c[lab])
        ax.add_patch(rect)
    plt.show()

def plot_topKvsnms(backbone, rpn, model, ktimes=5):
  print("==============================================================================")
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
  for j in range(ktimes):
    for iter, batch in enumerate(test_loader, 0):
      images, labels, boxes, masks, indexes = batch
      break
    images = images.to(device)

    backout = backbone(images.float())
    # if useours:
    # logits, reg_bbox = rpn.forward(images.float())
    # _ , proposals= rpn.postprocessImg(logits,reg_bbox, 0.5, 2000, 1000)
    # proposals = [proposals]

    im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
    rpnout = rpn(im_lis, backout)
    keep_topK = 200
    proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]  
    fpn_feat_list= list(backout.values())

    feature_vectors = model.MultiScaleRoiAlign(fpn_feat_list,proposals)
    class_logits, box_pred= model.forward(feature_vectors.to(device), eval=True)

    sortedboxes, sortedconf, sorted_labels = model.topkplotter(class_logits, box_pred, proposals, conf_thresh=0.5, keep_num_preNMS=20)
    img_squeeze=images[0,:,:,:].to('cpu').numpy()
    x_min = img_squeeze.min(axis=(1, 2), keepdims=True)
    x_max = img_squeeze.max(axis=(1, 2), keepdims=True)
    img_squeeze = (img_squeeze - x_min)/(x_max-x_min)

    fig,ax=plt.subplots(1,1)
    ax.imshow(np.transpose(img_squeeze, axes=(1,2,0)))
    c=['none', 'green', 'blue','yellow']

    if type(sortedboxes) == int:
      continue

    for no in range(len(sortedboxes)):
        box=sortedboxes[no].detach().cpu().numpy()
        lab= sorted_labels[no].item()
        rect=Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color=c[lab])
        ax.add_patch(rect)
    plt.show()

    fin_conf, fin_postbox, fin_lab = model.postprocess_detections(class_logits, box_pred, proposals, conf_thresh=0.5, keep_num_preNMS=20, keep_num_postNMS=2)
    
    fig,ax=plt.subplots(1,1)
    ax.imshow(np.transpose(img_squeeze, axes=(1,2,0)))
    c=['none', 'green', 'blue','yellow']

    if type(fin_conf) == int:
      continue

    for no in range(len(fin_postbox)):
        box=fin_postbox[no].detach().cpu().numpy()
        lab= fin_lab[no].item()
        rect=Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color=c[lab])
        ax.add_patch(rect)
    plt.show()

    print("==============================================================================")

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

def mean_average_precision(model, rpn, backbone, test_loader):

  total_matches= {0:[],1:[],2:[]}
  total_scores = {0:[],1:[],2:[]}
  total_trues  = {0:0,1:0,2:0}

  for iter, batch in enumerate(test_loader, 0):
    images, labels, boxes, masks, indexes = batch
    images = images.to(device)
    backout = backbone(images.float())
    im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
    rpnout = rpn(im_lis, backout)
    keep_topK = 200
    proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
    fpn_feat_list= list(backout.values())

    feature_vectors = model.MultiScaleRoiAlign(fpn_feat_list,proposals)
    class_logits, box_pred= model.forward(feature_vectors.to(device), eval=True)

    fin_conf, fin_postbox, fin_lab = model.postprocess_detections(class_logits, box_pred, proposals, conf_thresh=0.5, keep_num_preNMS=20, keep_num_postNMS=2)

    img_squeeze=images[0,:,:,:].to('cpu').numpy()
    x_min = img_squeeze.min(axis=(1, 2), keepdims=True)
    x_max = img_squeeze.max(axis=(1, 2), keepdims=True)
    img_squeeze = (img_squeeze - x_min)/(x_max-x_min)

    ## EVAL
    gtboxes = torch.vstack((boxes)).to(device)
    gtlabels= torch.vstack((labels)).reshape(-1)

    for clas in range(3):
      total_trues[clas] = total_trues[clas] + (gtlabels == clas+1).sum().item()
    # print(gtboxes, fin_postbox)
    if(type(gtboxes) !=int and type(fin_postbox) !=int):# and gtboxes != 0 and fin_postbox != 0):
      # print(gtboxes, fin_postbox)
      ious =  torch.zeros((len(gtboxes), len(fin_postbox))).to(device)
      for gts in range(len(gtlabels)):
        ious[gts] = IOU_all(fin_postbox.to(device), gtboxes[gts])
      confs, bbox_idx = torch.max(ious, dim=1)    
      pred_lab = fin_lab[bbox_idx][:,0,0]
      pred_conf = fin_conf[bbox_idx][:,0]

      for ind in range(len(fin_lab)):
        if ind not in bbox_idx:
          total_matches[int(fin_lab[ind])-1].extend([0])
          total_scores[int(fin_lab[ind])-1].extend([fin_conf[ind].item()])

      for i in range(len(bbox_idx)):
        if pred_conf[i] > 0.5 and pred_lab[i] == gtlabels[i]:
          total_matches[int(gtlabels[i])-1].extend([1])
          total_scores[int(gtlabels[i])-1].extend([pred_conf[i].item()])
        else:
          total_matches[int(gtlabels[i])-1].extend([0])
          total_scores[int(gtlabels[i])-1].extend([pred_conf[i].item()])

  return total_matches, total_scores, total_trues

def average_precision(match_values,score_values,total_trues):
    match_values=np.array(match_values)
    score_values=np.array(score_values)
    
    maximum_score= np.max(score_values)
    ln= np.linspace(0.6,maximum_score,num=100)
    precision_mat= np.zeros((101))
    recall_mat= np.zeros((101))

    for i,th in enumerate(ln):
      matches= match_values[score_values>=th]
      TP= np.sum(matches)
      total_positive= matches.shape[0]
      precision=1

      if total_positive>0:
        precision=TP/total_positive
      
      recall=1
      if total_trues>0:
        recall=TP/total_trues
      precision_mat[i]=precision
      recall_mat[i]=recall

    recall_mat[100]=0
    precision_mat[100]=1

    sorted_ind=np.argsort(recall_mat)
    sorted_recall=recall_mat[sorted_ind]
    sorted_precision=precision_mat[sorted_ind]
    area=metrics.auc(sorted_recall,sorted_precision)

    return area, precision_mat, recall_mat

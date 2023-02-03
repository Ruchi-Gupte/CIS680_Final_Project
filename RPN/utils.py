import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import h5py
from torchvision import transforms
from matplotlib.colors import ListedColormap

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
seed = 17
torch.manual_seed(seed)

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

def output_flattening(out_r, out_c, anchors):
# This function flattens the output of the network and the corresponding anchors
# in the sense that it concatenate  the outputs and the anchors from all the grid cells from all
# the FPN levels from all the images into 2D matrices
# Each row correspond of the 2D matrices corresponds to a specific grid cell
# Input:
#       out_r: list:len(FPN){(bz,num_anchors*4,grid_size[0],grid_size[1])}
#       out_c: list:len(FPN){(bz,num_anchors*1,grid_size[0],grid_size[1])}
#       anchors: list:len(FPN){(num_anchors,grid_size[0]*grid_size[1],4)}
# Output:
#       flatten_regr: (total_number_of_anchors*bz,4)
#       flatten_clas: (total_number_of_anchors*bz)
#       flatten_anchors: (total_number_of_anchors*bz,4)
#
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
#Converts coordinates from xyxy format to xywh and vice versa
  if to_wh:
    fincord=[]
    for asc in range(len(coords)):
      newcoords = torch.zeros(coords.shape)
      xl  = coords[asc][:,:,0]
      yl  = coords[asc][:,:,1]
      xr  = coords[asc][:,:,2]
      yr  = coords[asc][:,:,3]

      newcoords[:,:,2] = xr - xl
      newcoords[:,:,3] = yr - yl
      newcoords[:,:,0] = xl + newcoords[:,:,2]/2
      newcoords[:,:,1] = yl + newcoords[:,:,3]/2
      fincord.append(newcoords)

    fincord= torch.stack((fincord)) 
    return fincord

  else:
    fincord=[]
    for asc in range(len(coords)):
      newcoords = torch.zeros(coords[asc].shape)
      cx  = coords[asc][:,:,0]
      cy  = coords[asc][:,:,1]
      w  = coords[asc][:,:,2]
      h  = coords[asc][:,:,3]

      newcoords[:,:,0] = cx - w/2
      newcoords[:,:,1] = cy - h/2
      newcoords[:,:,2] = cx + w/2 
      newcoords[:,:,3] = cy + h/2
      fincord.append(newcoords)
    
    fincord= torch.stack((fincord))
    return fincord


def output_decoding(flatten_out,flatten_anchors, device='cpu'):
# This function decodes the output that is given in the encoded format (defined in the handout)
# into box coordinates where it returns the upper left and lower right corner of the proposed box
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
#
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


def viz_all_levels_all_asp(gt_, ground_coord_, images_, model): 
  for img_of_batch in range(2):
    # Since we only want to plot one
    images=images_[img_of_batch,:,:,:]
    anchors_ = model.get_anchors()
    for lev in range(5):    
      gt = gt_[lev][img_of_batch]
      temp = ground_coord_[lev][img_of_batch]
      ground_coord = temp.reshape(3,4,temp.shape[-2],temp.shape[-1])
      anchors = anchors_[lev]

      img=images.numpy()
      x_min = img.min(axis=(1, 2), keepdims=True)
      x_max = img.max(axis=(1, 2), keepdims=True)
      img = (img - x_min)/(x_max-x_min)
      
      for asp in range(3):
        fig,ax=plt.subplots(1,1)
        ax.imshow(np.transpose(img, axes=(1,2,0)))
        gtas = gt[asp].view(-1).to('cpu')
        gcoas = torch.permute(ground_coord[asp],(1,2,0)).reshape(-1,4)
        ancas = anchors[asp].reshape(-1,4)
        decoded=output_decoding(gcoas,ancas)

        find_cor=(gtas==1).nonzero().to('cpu')
        for elem in find_cor:
            coord=decoded[elem,:].view(-1).to('cpu')
            anchor=ancas[elem,:].view(-1).to('cpu')

            rect=Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
            ax.add_patch(rect)

            rect=Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color='r', linewidth=2)
            ax.add_patch(rect)
        plt.show()


def viz_all_levels(gt_, ground_coord_, images_, model): 
  for img_of_batch in range(2):
    images=images_[img_of_batch,:,:,:]
    anchors_ = model.get_anchors()

    for lev in range(5):
      gt = torch.unsqueeze(gt_[lev][img_of_batch],0)
      ground_coord = torch.unsqueeze(ground_coord_[lev][img_of_batch],0)
      anchors = anchors_[lev]
      flatten_coord,flatten_gt,flatten_anchors=output_flattening([ground_coord],[gt],[anchors])
      decoded_coord=output_decoding(flatten_coord,flatten_anchors).to('cpu')

      img=images.numpy()
      x_min = img.min(axis=(1, 2), keepdims=True)
      x_max = img.max(axis=(1, 2), keepdims=True)
      img = (img - x_min)/(x_max-x_min)
      
      fig,ax=plt.subplots(1,1)
      ax.imshow(np.transpose(img, axes=(1,2,0)))
      
      find_cor=(flatten_gt==1).nonzero().to('cpu')
      for elem in find_cor:
          coord=decoded_coord[elem,:].view(-1).to('cpu')
          anchor=flatten_anchors[elem,:].view(-1).to('cpu')
          rect=Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
          ax.add_patch(rect)

          rect=Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color='r', linewidth=2)
          ax.add_patch(rect)
      plt.show()

def viz_all_pos(gt_, ground_coord_, images_, model): 
  for img_of_batch in range(2):
    images=images_[img_of_batch,:,:,:] 

    ground_coord = []
    gt = []
    for lev in range(5):
      ground_coord.append(torch.unsqueeze(ground_coord_[lev][img_of_batch],0))
      gt.append(torch.unsqueeze(gt_[lev][img_of_batch],0))
      
    flatten_coord,flatten_gt,flatten_anchors=output_flattening(ground_coord,gt,model.get_anchors())
    decoded_coord=output_decoding(flatten_coord,flatten_anchors).to('cpu')

    img=images.numpy()
    x_min = img.min(axis=(1, 2), keepdims=True)
    x_max = img.max(axis=(1, 2), keepdims=True)
    img = (img - x_min)/(x_max-x_min)
    
    fig,ax=plt.subplots(1,1)
    ax.imshow(np.transpose(img, axes=(1,2,0)))
    
    find_cor=(flatten_gt==1).nonzero().to('cpu')
    for elem in find_cor:
        coord=decoded_coord[elem,:].view(-1).to('cpu')
        anchor=flatten_anchors[elem,:].view(-1).to('cpu')
        rect=Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
        ax.add_patch(rect)

        rect=Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color='r', linewidth=2)
        ax.add_patch(rect)
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
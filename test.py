# -*- coding:utf-8 -*-
# Author : lkq
# Data : 2019/3/6 16:15
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from dataload import *
import matplotlib.pyplot as plt
from PIL import Image
from Unet import Unet
from collections import OrderedDict
from networks.MobileNetV2_unet import MobileNetV2_unet
from networks.student import STMobileNetV2_unet
import os
import numpy
import time
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class Evaluator:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        #confusion_matrix
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        mean_acc_cls = np.nanmean(acc_cls)
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iou = np.nanmean(iou)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        #return acc, acc_cls, mean_acc_cls, iou, mean_iou, fwavacc
        TP = self.hist[1,1]
        TN = self.hist[0,0]
        FP = self.hist[0,1]
        FN = self.hist[1,0]
        Pre = TP/(TP+FP)
        ACC = (TP+TN)/(TP+TN+FP+FN)
        Se = TP/(TP+FN)
        Sp = TN/(TN+FP)
        F1 = 2*Pre*Se/(Pre+Se)
        return acc,Se,Sp,mean_iou,F1
        
class Inference:

    def __init__(self, model, num_classes, test_img):
        self.model = model
        self.num_classes = num_classes
        b, c, h, w = test_img.size()
        self.b, self.h, self.w = b, h, w
        self.test_img = test_img

    def multiscale_inference(self, test_img, is_flip=True):
        pre = []
        inf_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        for scale in inf_scales:
            img_scaled = F.interpolate(test_img, size=(int(self.h * scale), int(self.w * scale)), mode='bilinear',
                                       align_corners=True)
            pre_scaled = self.single_inference(img_scaled)
            pre.append(pre_scaled)

            if is_flip:
                img_scaled = self.flip_image(img_scaled)
                pre_scaled = self.single_inference(img_scaled)
                pre_scaled = self.flip_image(pre_scaled)
                pre.append(pre_scaled)

        pre_final = self.fushion_avg(pre)

        return pre_final

    def single_inference(self,test_img):
        pre = self.model(test_img)
        pre = F.interpolate(pre, size=(self.h, self.w), mode='bilinear', align_corners=True)
        pre = F.log_softmax(pre, dim=1)
        pre = pre.data.cpu()

        return pre

    def fushion_avg(self, pre):
        pre_final = torch.zeros(self.b, self.num_classes, self.h, self.w)
        for pre_scaled in pre:
            pre_final = pre_final + pre_scaled
        pre_final = pre_final / len(pre)
        return pre_final

    def flip_image(self, img):
        flipped = torch.flip(img, [2, 3])
        return flipped


#set learning rate is here
def adjust_learning_rate_poly(optimizer,all_iter,now_iter,epoch):

    if epoch<= 50:
       base_lr = 0.003 #0.0003
    elif epoch>50 and epoch<= 100:
       base_lr = 0.0003 #0.0003
    elif epoch>100 and epoch<=200:
         base_lr = 0.00003
    elif epoch>200 and epoch<=300:
         base_lr = 0.000005


    lr = base_lr
    if len(optimizer.param_groups) == 1:
        optimizer.param_groups[0]['lr'] = lr
    else:
        # enlarge the lr at the head
        optimizer.param_groups[0]['lr'] = lr * 0.1
    """
    for i in range(1, len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = lr
    """


def test(epoch):
    
    #model = DeepLabV3Plus()
    #model = UnetSKA2()
    #model = Vgg()
    #model = Unet()
    model = STMobileNetV2_unet(pre_trained=None)
    if torch.cuda.device_count() > 1:
       model = torch.nn.DataParallel(model)
    model.to(device)
    # model.load_state_dict(torch.load(args.ckp,map_location='cpu'))
    file=open('result/graph_Ex1.txt','a')
    for i in range(1):


        ckp_root = 'result/graph_Ex1/graph_Ex_MobileUNet_'+str(epoch)+'.pth'
        #ckp_root = 'result/CHASEDB1_KD1_MobileUNet_80.pth'
        #ckp_root = '/disk2/lkq/ckp_test/SKA2/chasedb1/train/weights_'+str(60+1*i)+'.pth'
        #pre_model_dict = torch.load(ckp_root, map_location='cpu')
        pre_model_dict = torch.load(ckp_root)
        
        new_state_dict = OrderedDict()
        """
        for k, v in pre_model_dict.items():
            name = k[0:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        """
        model.load_state_dict(pre_model_dict)
        Cityscapes_dataset = VesselDataset(root='/home/muyi.sun/KD_CT_MIR/CHAOS_dataset/Test/', names_files='/home/muyi.sun/KD_CT_MIR/CHAOS_dataset/test_names.txt',
                                              transform=transforms.Compose([ToTensor()]))

        dataloaders = DataLoader(Cityscapes_dataset, batch_size=64, shuffle=True)
        evaluator = Evaluator(num_classes=2)
        model.eval()
        i = 0
        save_path = 'chasedb1_final/'

        if not os.path.exists(save_path):
               os.makedirs(save_path)
        #plt.ion()

        ROC_y = []
        ROC_pred = []

        with torch.no_grad():
            for sample in dataloaders:
                image = sample['image']
                label = sample['label']
                name = sample['name']
                #b,c,h,w = image.size()
                #print(h,w)
                #print(name)
                image = image.to(device)
                label = label.to(device)
                #pre = Inference(model=model, num_classes=2,test_img=image).multiscale_inference(image)
                pre =Inference(model,2,image).single_inference(image)
                pre_roc = pre[:,1,:,:]
                pre_roc = torch.squeeze(pre_roc).cpu().numpy()
                pre_roc = pre_roc.flatten()
                ROC_pred.append(pre_roc)
                pre = torch.argmax(pre, 1)
                pre = torch.squeeze(pre).cpu().numpy()
                label = torch.squeeze(label).cpu().numpy()
                pre_y = label.flatten()
                ROC_y.append(pre_y)
                label = np.uint8(label)
                # print(img_y.shape)
                # print(img__.shape)
                evaluator.add_batch(pre, label)
                #pre = pre * 255

                #print(pre.shape)
                #pre = Image.fromarray(np.uint8(pre))
                #pre.save(save_path+'%s'%name[0])
                #image_name = name[0].replace('image_green_clahe', 'image')
                #image_name = 'vessel_dataset/'+image_name
                #image = Image.open(image_name)
                #image = CenterCrop2(512)(image)
                #image = np.array(image)
                #img_pre = img_pre.resize((64,64), Image.NEAREST)
                #pre.save(save_path+'%s'%name[0])
                '''''''''

                plt.subplot(131)
                plt.title('image')
                plt.imshow(image)
                plt.subplot(132)
                plt.title('label')
                plt.imshow(label,cmap='gray')
                plt.subplot(133)
                plt.title('prediction')
                plt.imshow(pre,cmap='gray')
                plt.savefig(save_path+'%s'%i)
                #plt.pause(1)
                #plt.show()
                i+=1
                if i==4:
                   break

                '''''''''
            roc_y = np.array(ROC_y)
            roc_pred = np.array(ROC_pred)
            fpr, tpr, thresholds = roc_curve(roc_y.flatten(), roc_pred.flatten(), pos_label=1)
            AUC = auc(fpr, tpr)
            #AUC =0.9895
            """
            plt.figure()
            lw = 2
            plt.figure(figsize=(5,4))
            plt.plot(fpr, tpr, color='deepskyblue', lw=lw, label='ROC curve (area = %0.4f)' % AUC)
            #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve on DRIVE')
            plt.legend(loc="lower right")
            plt.savefig(save_path + "128_teacher1_drive.jpg")
            #plt.show()
            """
            acc, se, sp, iou, f1= evaluator.evaluate()
            print(ckp_root)
            #print("acc:%.5f, se:%.5f, sp:%.5f, f1:%.5f, auc:%.5f" % (acc, se,sp,f1,AUC))
            print('acc:{}, se:{}, sp:{}, miou:{}, f1:{}, AUC:{}'.format(acc, se,sp,iou,f1,AUC))
            file.write( 'acc:{}, se:{}, sp:{}, miou:{}, f1:{}, AUC:{}'.format(acc, se,sp,iou,f1,AUC) +'   _'+str(epoch)+'\n')
    file.close()



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, default="test", help="train or test")
    # parse.add_argument("--batch_size", type=int, default=32)
    parse.add_argument("--batch_size", type=int, default=32)
    #parse.add_argument("--ckp", type=str, default="weights_109.pth", help="the path of model weight file")
    args = parse.parse_args()

    #print(torch.cuda.is_available())

    if args.action=="train":
        train()
    elif args.action=="test":
        for i in range(50,250):
            #if i%2 == 0:
            test(i)
        #test()

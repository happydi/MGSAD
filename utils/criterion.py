import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        return loss1 + loss2*0.4

class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, upsample=False, temperature=1):
        super(CriterionKD, self).__init__()
        self.upsample = upsample
        self.temperature = temperature
        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, pred, soft):
        soft[0].detach()
        h, w = soft[0].size(2), soft[0].size(3)
        if self.upsample:
            scale_pred = F.upsample(input=pred[0], size=(h * 8, w * 8), mode='bilinear', align_corners=True)
            scale_soft = F.upsample(input=soft[0], size=(h * 8, w * 8), mode='bilinear', align_corners=True)
        else:
            scale_pred = pred[0]
            scale_soft = soft[0]
        loss = self.criterion_kd(F.log_softmax(scale_pred / self.temperature, dim=1), F.softmax(scale_soft / self.temperature, dim=1))
        return loss

class CriterionAdvForG(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdvForG, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S):
        g_out_fake = d_out_S[0]
        if self.adv_loss == 'wgan-gp':
            g_loss_fake = - g_out_fake.mean()
        elif self.adv_loss == 'hinge':
            g_loss_fake = - g_out_fake.mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return g_loss_fake

class CriterionAdv(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdv, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S, d_out_T):
        assert d_out_S[0].shape == d_out_T[0].shape,'the output dim of D with teacher and student as input differ'
        '''teacher output'''
        d_out_real = d_out_T[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif self.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')

        # apply Gumbel Softmax
        '''student output'''
        d_out_fake = d_out_S[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return d_loss_real + d_loss_fake

class CriterionAdditionalGP(nn.Module):
    def __init__(self, D_net, lambda_gp):
        super(CriterionAdditionalGP, self).__init__()
        self.D = D_net
        self.lambda_gp = lambda_gp
        self.interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    def forward(self, d_in_S, d_in_T):
        assert d_in_S[0].shape == d_in_T[0].shape,'the output dim of D with teacher and student as input differ'

        real_images = self.interp(d_in_T[0])
        fake_images = self.interp(d_in_S[0])
        # Compute gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
        interpolated = Variable((alpha * real_images.data + (1 - alpha) * fake_images.data), requires_grad=True)
        out = self.D(interpolated)
        grad = torch.autograd.grad(outputs=out[0],
                                    inputs=interpolated,
                                    grad_outputs=torch.ones(out[0].size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        d_loss = self.lambda_gp * d_loss_gp
        return d_loss

class CriterionIFV(nn.Module):
    def __init__(self, classes):
        super(CriterionIFV, self).__init__()
        self.num_classes = classes

    def forward(self, preds_S, preds_T, target):
        feat_S = preds_S[2]
        feat_T = preds_T[2]
        # print(feat_S.shape)
        feat_T.detach()
        size_f = (feat_S.shape[2], feat_S.shape[3])
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_S.size())
        tar_feat_T = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_T.size())
        center_feat_S = feat_S.clone()
        center_feat_T = feat_T.clone()
        for i in range(self.num_classes):
          mask_feat_S = (tar_feat_S == i).float()
          mask_feat_T = (tar_feat_T == i).float()
          center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * ((mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
          center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * ((mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)

        # print(feat_S.shape) #torch.Size([8, 128, 33, 33])
        # print(feat_T.shape) #torch.Size([8, 512, 33, 33])

        # cosinesimilarity along C
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(feat_S, center_feat_S)
        # print(pcsim_feat_S.shape) #torch.Size([8, 33, 33])
        pcsim_feat_T = cos(feat_T, center_feat_T)
        # print(pcsim_feat_T.shape) #torch.Size([8, 33, 33])

        # #############
        # L2
        # #############
        # mseloss
        mse = nn.MSELoss()
        loss1 = mse(pcsim_feat_S, pcsim_feat_T)

        ##############
        # kl
        ##############
        batch = pcsim_feat_S.shape[0]
        # print(pcsim_feat_T.shape)
        # print("--------------------------")
        # print(pcsim_feat_S.shape)
        # print("--------------------------")
        # S = F.log_softmax(pcsim_feat_S.view(batch,-1),dim=1)
        # T = F.softmax(pcsim_feat_T.view(batch,-1),dim=1)
        # loss = F.kl_div(S, T)

        # loss=0
        # for i in range(batch):
        #     s = S[i]
        #     t = T[i]
        #     loss = loss + F.kl_div(s,t)
        # print(loss/batch)

        ##############
        # L1
        ##############
        l1 = nn.L1Loss()
        loss2 = l1(pcsim_feat_S,pcsim_feat_T)
        return loss1+loss2

class ChannelWiseDivergence(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation
     <https://arxiv.org/abs/2011.13256>`_.

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name(str):
        tau (float, optional): Temperature coefficient. Defaults to 1.0.
        weight (float, optional): Weight of loss.Defaults to 1.0.

    """

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 tau=1.0
                 ):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

    def forward(self,
                preds_S,
                preds_T):
        """Forward function."""
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'
        N, C, W, H = preds_S.shape

        if self.align is not None:
            preds_S = self.align(preds_S)
        #
        # print('----------------------')
        # print(preds_S.shape)

        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)
        softmax_pred_S = F.softmax(preds_S.view(-1, W * H) / self.tau, dim=1)
        # print('----------------------')
        # print(softmax_pred_T[0,:])
        # print('----------------------')
        # print(softmax_pred_S[0, :])
        # print('----------------------')
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(- softmax_pred_T * logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (self.tau ** 2)

        return loss / (C * N)

class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale, feat_ind):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.feat_ind = feat_ind
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S[self.feat_ind]
        feat_T = preds_T[self.feat_ind]
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss

class CriterionPixelWise(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionPixelWise, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds_S, preds_T):
        preds_T[0].detach()
        assert preds_S[0].shape == preds_T[0].shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S[0].shape
        softmax_pred_T = F.softmax(preds_T[0].view(-1, C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum(- softmax_pred_T * logsoftmax(preds_S[0].view(-1, C)))) / W / H
        return loss
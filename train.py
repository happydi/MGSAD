from options import TrainOptions
from torch.utils import data
from dataset.dataset_WHDLD import WHTrainValSet
from dataset.dataset_ISPRS import ISTrainValSet
from networks.kd_model import NetModel
from utils.evaluator_data import evaluate_main
from summaries import TensorboardSummary
import os
import warnings

from utils.evaluator import Evaluator

warnings.filterwarnings("ignore")
import logging

# for reproducibility
import random
import numpy as np
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

args = TrainOptions().initialize()
# device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# data
h, w = map(int, args.input_size.split(','))

if args.data_set == 'WHDLD':
    trainloader = data.DataLoader(
        WHTrainValSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                      crop_size=(h, w), scale=args.random_scale, mirror=args.random_mirror),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    valloader = data.DataLoader(
        WHTrainValSet(args.data_dir, args.data_listval, crop_size=(256, 256), scale=False, mirror=False),
        batch_size=1, shuffle=False, pin_memory=True)

if args.data_set == 'ISPRS':
    trainloader = data.DataLoader(
        ISTrainValSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                      crop_size=(h, w), scale=args.random_scale, mirror=args.random_mirror),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    valloader = data.DataLoader(
        ISTrainValSet(args.data_dir, args.data_listval, crop_size=(512, 512), scale=False, mirror=False),
        batch_size=1, shuffle=False, pin_memory=True)

summary = TensorboardSummary(args)
writer = summary.create_summary()

# model
model = NetModel(args,writer=writer)

evaluator = Evaluator(args.num_classes)
evaluator_training = Evaluator(args.num_classes)

epoch = 0
best_IOU = 0.0

def cal_training(evaluator):
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU, IU_array = evaluator.Mean_Intersection_over_Union()

    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    return Acc, Acc_class, mIoU, FWIoU, IU_array

# train
for step, data in enumerate(trainloader, args.last_step):
    model.adjust_learning_rate(args.lr_g, model.G_solver, step)

    model.adjust_learning_rate(args.lr_d, model.D_solver, step)

    model.set_input(data)

    model.optimize_parameters(step)
    model.print_info(step)
    seg_gt, seg_pred = model.cal_miou()

    evaluator_training.add_batch(seg_gt, seg_pred)

    if ((step + 1) >= args.save_ckpt_start) or ((step + 1) % args.save_ckpt_every == 0):
        epoch += 1

        # print training performance
        Acc_t, Acc_class_t, mIoU_t, FWIoU_t, IU_array_t=cal_training(evaluator_training)
        logging.info(
            '(train)Acc: {:.4f} Acc_class: {:.4f} mean_IU: {:.4f} fwIoU:{:4f} \nIU_array: \n{}'.format(Acc_t, Acc_class_t,
                                                                                                     mIoU_t,
                                                                                                     FWIoU_t,
                                                                                                     IU_array_t))
        writer.add_scalar('train/miou', mIoU_t, epoch)
        writer.add_scalar('train/accuracy', Acc_t, epoch)
        writer.add_scalar('train/acc_class', Acc_class_t, epoch)
        writer.add_scalar('train/FWIoU', FWIoU_t, epoch)


        Acc, Acc_class, mIoU, FWIoU, IU_array = evaluate_main(model.student, evaluator, valloader, '512,512',
                                                              args.num_classes, True, 1, 'val')
        logging.info(
            '(val)Acc: {:.4f} Acc_class: {:.4f} mean_IU: {:.4f} fwIoU:{:4f} \nIU_array: \n{}'.format(Acc, Acc_class, mIoU,
                                                                                                FWIoU,
                                                                                                IU_array))
        writer.add_scalar('val/miou', mIoU, epoch)
        writer.add_scalar('val/accuracy', Acc, epoch)
        writer.add_scalar('val/acc_class', Acc_class, epoch)
        writer.add_scalar('val/FWIoU', FWIoU, epoch)
        # logging.info('mean_IU: {:.6f}  IU_array: \n{}'.format(mean_IU, IU_array))
        if step % 10000 == 0 or (step == 39999):
            model.save_ckpt(step, mIoU, mIoU_t)
        # if mIoU > 0.6 and mIoU > best_IOU:
        #     best_IOU = mIoU


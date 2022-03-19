import os
import torch

from mobilenet_WHDLD.pspnet_mobilenetv2 import PSPNet
from networks.pspnet_relu import Res_pspnet, BasicBlock, Bottleneck
from options import TestOptions
from torch.utils import data
from dataset.dataset_WHDLD import WHTrainValSet
from dataset.dataset_ISPRS import ISTestSet,ISTrainValSet
from utils.evaluator import Evaluator
from utils.evaluator_data import evaluate_main
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args = TestOptions().initialize()
    print(args.data_set)
    evaluator = Evaluator(6)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if args.data_set=='WHDLD':
        testloader = data.DataLoader(WHTrainValSet(args.data_dir, args.data_list, crop_size=(256, 256), scale=False, mirror=False),
                                 batch_size=1, shuffle=False, pin_memory=True)
    if args.data_set=='ISPRS':
        testloader = data.DataLoader(ISTrainValSet(args.data_dir, args.data_list, crop_size=(512, 512),
                          scale=False, mirror=False),batch_size=1, shuffle=False, pin_memory=True)

    #model = Res_pspnet(Bottleneck, [3, 4, 23, 3], num_classes=args.num_classes)
    #model = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes)
    model = Res_pspnet(Bottleneck, [3, 4, 6, 3], num_classes=args.num_classes)
    #model = PSPNet(num_classes=args.num_classes, downsample_factor=8, pretrained=False, aux_branch=True)  #pspnet_mobilenetv2

    args.restore_from='...'

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    #model.load_state_dict(torch.load(args.restore_from,map_location="cuda:0"))
    model.load_state_dict(torch.load(args.restore_from,map_location = 'cpu'))

    save_path='outputs/'


    pre_folder = Path(save_path)
    if not pre_folder.exists():
        pre_folder.mkdir()

    Acc, Acc_class, mIoU, FWIoU, IU_array = evaluate_main(args.data_set,model, evaluator, save_path,testloader, args.input_size, args.num_classes, True, 1, 'test')


    print(
        'Acc: {:.4f} Acc_class: {:.4f} mean_IU: {:.4f} fwIoU:{:4f} \nIU_array: \n{}'.format(Acc, Acc_class, mIoU, FWIoU,
                                                                                            IU_array))

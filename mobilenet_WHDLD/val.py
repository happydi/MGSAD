import os
import torch
from options import ValOptions
from torch.utils import data
from dataset.dataset_WHDLD import WHTrainValSet
from networks.pspnet_relu import Res_pspnet, BasicBlock, Bottleneck
from utils.evaluator_WHDLD import evaluate_main
import warnings

from utils.evaluator import Evaluator

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args = ValOptions().initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    evaluator = Evaluator(args.num_classes)
    valloader = data.DataLoader(WHTrainValSet(args.data_dir, args.data_list, crop_size=(1024, 2048), scale=False, mirror=False),
                                batch_size=1, shuffle=False, pin_memory=True)
    student = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes = args.num_classes)
    student.load_state_dict(torch.load(args.restore_from))
    Acc, Acc_class, mIoU, FWIoU, IU_array = evaluate_main(student, evaluator, valloader, '512,512', args.num_classes, True, 1, 'val')
    print('Acc: {:.4f} Acc_class: {:.4f} mean_IU: {:.4f} fwIoU:{:4f} \nIU_array: \n{}'.format(Acc, Acc_class, mIoU, FWIoU,
                                                                                            IU_array))

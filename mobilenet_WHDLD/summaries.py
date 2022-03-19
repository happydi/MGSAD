import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
# from utils.decode_images import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('./train_result', args.data_set, args.save_visualpath)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    # def visualize_image(self, writer, target, output, global_step):
    #     # grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
    #     # writer.add_image('Image', grid_image, global_step)
    #     # grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:4], 1)[1].detach().cpu().numpy(),
    #     #                                                dataset=self.args.dataset, classes=self.args.classes), 3, normalize=False, range=(0, 255))
    #     # writer.add_image('Predicted_label', grid_image, global_step)
    #     # grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:4], 1).detach().cpu().numpy(),
    #     #                                                dataset=self.args.dataset, classes=self.args.classes), 3, normalize=False, range=(0, 255))
    #     # writer.add_image('Groundtruth_label', grid_image, global_step)
    #     grid_image = make_grid(decode_seg_map_sequence(output[:4],
    #                                                    dataset=self.args.dataset, classes=self.args.classes), 3,
    #                            normalize=False, range=(0, 255))
    #     writer.add_image('Predicted_label', grid_image, global_step)
    #     grid_image = make_grid(decode_seg_map_sequence(target[:4],
    #                                                    dataset=self.args.dataset, classes=self.args.classes), 3,
    #                            normalize=False, range=(0, 255))
    #     writer.add_image('Groundtruth_label', grid_image, global_step)
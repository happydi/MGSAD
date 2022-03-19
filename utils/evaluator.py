import imageio
import numpy as np
from glob import glob
import os
import warnings
import cv2
from tqdm import tqdm
warnings.filterwarnings('ignore')


def rgb2label(image):
    label_first = [0, 198, 251, 182, 39, 194, 165, 105, 249, 28]
    label = np.zeros((image.shape[0], image.shape[1]))
    for i in label_first:
        label[image[:, :, 0] == i] = label_first.index(i)
    label = label.astype(np.uint8)
    return label

def fake2label(path):
    label = np.zeros((128, 128))
    img = imageio.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # print(img_gray[127, 127])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 69 >= img_gray[i, j] >= 25:
                label[i, j] = 9
            if 100 >= img_gray[i, j] >= 70:
                label[i, j] = 0
            if 139 >= img_gray[i, j] >= 101:
                label[i, j] = 5
            if 167 >= img_gray[i, j] >= 140:
                label[i, j] = 6
            # 170
            if 171 >= img_gray[i, j] >= 168:
                label[i, j] = 1
            if 194 >= img_gray[i, j] >= 172:
                label[i, j] = 4
            if 207 >= img_gray[i, j] >= 195:
                label[i, j] = 3
            if 218 >= img_gray[i, j] >= 208:
                label[i, j] = 7
            if 234 >= img_gray[i, j] >= 219:
                label[i, j] = 2
            if 253 >= img_gray[i, j] >= 235:
                label[i, j] = 8
    return label.astype(np.uint8)


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        # MIoU = np.diag(self.confusion_matrix) / (
        #             np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
        #             np.diag(self.confusion_matrix))
        # MIoU = np.nanmean(MIoU)
        pos = self.confusion_matrix.sum(1)
        res = self.confusion_matrix.sum(0)
        tp = np.diag(self.confusion_matrix)
        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        MIoU = IU_array.mean()
        return MIoU, IU_array

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Kappa(self):
        p0 = self.Pixel_Accuracy()
        pe = np.dot(np.sum(self.confusion_matrix, axis=0), np.sum(self.confusion_matrix, axis=1)) / (
                    self.confusion_matrix.sum() * self.confusion_matrix.sum())
        kappa = (p0 - pe) / (1 - pe)
        return kappa

    def _generate_matrix(self, gt_image, pre_image):
        # mask = (gt_image >= 0) & (gt_image < self.num_class)
        # label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        # count = np.bincount(label, minlength=self.num_class**2)
        # confusion_matrix = count.reshape(self.num_class, self.num_class)
        # return confusion_matrix
        """
                Calcute the confusion matrix by given label and pred
                :param gt_label: the ground truth label
                :param pred_label: the pred label
                :param class_num: the nunber of class
                :return: the confusion matrix
                """
        index = (gt_image * self.num_class + pre_image).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((self.num_class, self.num_class))

        for i_label in range(self.num_class):
            for i_pred_label in range(self.num_class):
                cur_index = i_label * self.num_class + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]
        # print(confusion_matrix)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_confusion_matrix(self):
        return self.confusion_matrix

if __name__ =='__main__':
    img_root = '/data/model/ModelLearn/Image_Segmentation/result/U_Net/IGRSS'
    label_root = '/data/dataset/IGRSS2020high_986valid_norm/crop_img_128/test/dfc/'
    img_path_list = sorted(glob((os.path.join(img_root, '*.tif'))))
    label_path_list = sorted(glob((os.path.join(label_root, '*.tif'))))[1900:]
    print('Found {} test images'.format(len(img_path_list)))
    evaluator = Evaluator(10)
    for i in tqdm(range(len(img_path_list))):
        img_gray = imageio.imread(img_path_list[i])
        label_gray = imageio.imread(label_path_list[i]) - 1
        evaluator.add_batch(label_gray, img_gray)
    print('Acc: {}, Acc_class: {}, MIoU: {}, fwIoU: {}'.format(evaluator.Pixel_Accuracy(), evaluator.Pixel_Accuracy_Class(),
                                                               evaluator.Mean_Intersection_over_Union(), evaluator.Frequency_Weighted_Intersection_over_Union()))


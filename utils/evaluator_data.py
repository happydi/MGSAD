import os
import scipy
from scipy import ndimage
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data
from math import ceil
from PIL import Image as PILImage
import torch.nn as nn

def id2trainId(label, id_to_trainid, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

def get_palette_ISPRS(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    # 杂波/背景(Clutter/background)	（255，0，0）	0 low vegetable
    palette[0] = 255
    palette[1] = 0
    palette[2] = 0
    # 低矮灌木(Low Vegetation)		（0，255，255）    	1 tree
    palette[3] = 0
    palette[4] = 255
    palette[5] = 255
    # 乔木(Tree)				（0, 255,0）	2 car
    palette[6] = 0
    palette[7] = 255
    palette[8] = 0
    # 汽车(Car)				（255，255，0）	3 building
    palette[9] = 255
    palette[10] = 255
    palette[11] = 0
    # 建筑(Building)                  		（0，0，255）        	4 Impervious surfaces
    palette[12] = 0
    palette[13] = 0
    palette[14] = 255
    # # 防渗表面(Impervious surfaces)		（255，255，255）	5
    # palette[15] = 255
    # palette[16] = 255
    # palette[17] = 255
    return palette

def get_palette_WHDLD(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    # water 29 0,0,255
    palette[87] = 0
    palette[88] = 0
    palette[89] = 255
    # vegetabel 149 0,255,0
    palette[447] = 0
    palette[448] = 255
    palette[449] = 0
    # building 76 255,0,0
    palette[228] = 255
    palette[229] = 0
    palette[230] = 0
    # pavement 170 192,192,0
    palette[510] = 192
    palette[511] = 192
    palette[512] = 0
    # bare soil 128 128,128,128
    palette[384] = 128
    palette[385] = 128
    palette[386] = 128
    # road 225 255,255,0
    palette[675] = 255
    palette[676] = 255
    palette[677] = 0
    # for j in range(0, n):
    #     lab = j
    #     palette[j * 3 + 0] = 0
    #     palette[j * 3 + 1] = 0
    #     palette[j * 3 + 2] = 0
    #     i = 0
    #     while lab:
    #         palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
    #         palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
    #         palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
    #         i += 1
    #         lab >>= 3
    return palette

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding(net, image, tile_size, classes, flip_evaluation, recurrence):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[2], image_size[3], classes))
    count_predictions = np.zeros((image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            print("Predicting tile %i" % tile_counter)
            padded_prediction = net(Variable(torch.from_numpy(padded_img), volatile=True).cuda())
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            padded_prediction = interp(padded_prediction).cpu().data[0].numpy().transpose(1,2,0)
            prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    full_probs /= count_predictions
    return full_probs

def predict_whole(net, image, tile_size, recurrence):
    image = torch.from_numpy(image)
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    prediction = net(image.cuda())

    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).cpu().data[0].numpy().transpose(1,2,0)
    return prediction

def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation, recurrence):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image = image.data
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((H_, W_, classes))
    for scale in scales:
        scale = float(scale)

        # print("Predicting image scaled by %f" % scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)

        scaled_probs = predict_whole(net, scale_image, tile_size, recurrence)
        if flip_evaluation == True:
            flip_scaled_probs = predict_whole(net, scale_image[:,:,:,::-1].copy(), tile_size, recurrence)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:,::-1,:])

        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def evaluate_main(data_set,model, evaluator, save_path,loader, input_size, num_classes, whole = False, recurrence = 1, type = 'val'):
    """Create the model and start the evaluation process."""

    h, w = map(int, input_size.split(','))

    if data_set=='WHDLD':
        input_size = (256, 256)

    if data_set=='ISPRS':
        input_size = (512, 512)



    ignore_label = 255
    # id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
    #                 3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
    #                 7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
    #                 14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
    #                 18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    #                 28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
    if data_set == 'WHDLD':
        id_to_trainid = {225: 0, 149: 1, 29: 2, 170: 3, 76: 4, 128: 5}
    model.eval()
    model.cuda()

    confusion_matrix = np.zeros((num_classes,num_classes))


    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    i=1
    for index, batch in enumerate(loader):
        if index % 100 == 0:
            print('%d processd'%(index))
        if type == 'val':
            image, label, size, name = batch
        elif type == 'test':
            image, label, size, name = batch
        size = size[0].numpy()
        with torch.no_grad():
            if whole:
                output = predict_multiscale(model, image, input_size, [1.0], num_classes, False, recurrence)
            else:
                output = predict_sliding(model, image.numpy(), input_size, num_classes, False, recurrence)

        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        if type == 'test':
            if data_set == 'WHDLD':
                seg_pred = id2trainId(seg_pred, id_to_trainid, reverse=True)
                palette = get_palette_WHDLD(256)
            if data_set=='ISPRS':
                palette = get_palette_ISPRS(256)
            # print(seg_pre)  #(256,256)
            output_im = PILImage.fromarray(seg_pred)
            output_im.putpalette(palette)
            output_im.save(save_path+name[0]+'.png')
            seg_gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.int)
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]
            evaluator.add_batch(seg_gt, seg_pred)

        if type == 'val':
            seg_gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]
            evaluator.add_batch(seg_gt,seg_pred)
            # confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, num_classes)


    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU,IU_array = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    return Acc, Acc_class, mIoU, FWIoU, IU_array

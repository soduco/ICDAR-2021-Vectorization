from PIL import Image
import numpy as np
import pdb


if __name__== '__main__':
    img_train_input = '/lrde/home2/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-TRAIN-GT_LABELS_target.png'
    img_train_labels = np.array(Image.open(img_train_input))
    print('train image size {}'.format(img_train_labels.shape))
    print('number of pixel {}'.format(img_train_labels.shape[0] * img_train_labels.shape[1]))
    print('train label number {}'.format(len(np.unique(img_train_labels))))

    img_val_input = '/lrde/home2/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-GT_LABELS_target.png'
    img_val_labels = np.array(Image.open(img_val_input))
    print('val image size {}'.format(img_val_labels.shape))
    print('number of pixel {}'.format(img_val_labels.shape[0] * img_val_labels.shape[1]))
    print('Validation label number {}'.format(len(np.unique(img_val_labels))))

    img_test_input = '/lrde/home2/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1898_0004-TEST-GT_LABELS_target.png'
    img_test_labels = np.array(Image.open(img_test_input))
    print('test image size {}'.format(img_test_labels.shape))
    print('number of pixel {}'.format(img_test_labels.shape[0] * img_test_labels.shape[1]))
    print('test label number {}'.format(len(np.unique(img_test_labels))))   

    pdb.set_trace()
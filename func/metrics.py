import numpy as np
def mean_iou_score(pred, labels, n_labels):
    '''
    n_labels = channel num + 1.  e.g. 256, 256, 1 ===> n_labels=2
    '''
    mean_iou = 0
    n_labels = n_labels
    for i in range(n_labels):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / n_labels
        #print('class #%d : %1.5f'%(i, iou))
    #print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou
import argparse

def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', default='7')
    parser.add_argument('--test_mode', dest='test_mode', default= False , type=bool)


    parser.add_argument('--out-dir', dest='out_dir', default='Moth_rmbg_trainBRCAS')

    # model tuning
    parser.add_argument('--epoch', dest='epoch', type=int, default=300)
    parser.add_argument('--bz', '--batch-size', dest='bz', type=int, default=16)
    parser.add_argument('--lr', '--learning-rate', dest='lr', type=float, default=1e-4)
    parser.add_argument('--current_best_val_loss', dest='current_best_val_loss', default=float("inf"), type=float)
    parser.add_argument('--earlystop_patience', dest='earlystop_patience', default=25, type=float)
    parser.add_argument('--min_delta', dest='min_delta', default=0, type=float)
    parser.add_argument('--min_lr', dest='min_lr', default=1e-10, type=float)
    parser.add_argument('--lr_patience', dest='lr_patience', default= 10, type=int)
    parser.add_argument('--lr_reduce_factor', dest='lr_reduce_factor', default= 0.5, type=float)
    parser.add_argument('--keep_prob', dest='keep_prob', default= 1, type=float)
    parser.add_argument('--log_step', dest='log_step', default= 0.2, type=float)
    parser.add_argument('--momentum', dest='momentum', default= 0.9, type=float)

    # model simple
    parser.add_argument('--image-h', dest='im_h', default=256, type=int)
    parser.add_argument('--image-w', dest='im_w', default=256, type=int)
    parser.add_argument('--image-d', dest='im_c', default=3, type=int)
    parser.add_argument('--num_class', dest='num_class', default= 1, type=int)

    parser.add_argument('--stratify-label', dest='class_label', default='Alt_mean')
    parser.add_argument('--stratify-range', dest='class_range', type=int, default=500)

    # plot
    parser.add_argument('--plt_loss_min', dest='plt_loss_min', default= 0 , type=float)
    parser.add_argument('--plt_loss_max', dest='plt_loss_max', default= 0.5 , type=float)

    #return parser
    args = parser.parse_args([])
    return args
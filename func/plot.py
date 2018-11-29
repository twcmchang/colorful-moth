import matplotlib.pyplot as plt
import numpy as np
def plt_result(img, title, save_name=None, img_save=None, img_save_dir=None, fontsize = 25, show = False):
    '''
    img, title -> list()
    '''
    fig = plt.figure(figsize=(20, 20), dpi= 400)
    
    for i in range(len(img)):
        ax = fig.add_subplot('1'+str(len(img))+str(i+1))
        ax.set_title(title[i], fontsize=fontsize)
        plt.axis('off') # hide the number line on the x,y borders
        #ax.title.set_text("\nD(intra)={0}, D(inter)={1}".format(1,2))
        if img[-1] is not 3:
            ax.imshow(img[i], cmap='gray')
        else:
            ax.imshow(img[i])
#     save_to = OUTPUT_DIR+'CHECKING/'     
#     if not os.path.exists(save_to):
#             os.makedirs(save_to)
#     fig.savefig(save_to + save_name +'.png', dpi=100, format='png',bbox_inches='tight' )
    if not show == True:
        plt.close()
    else:
        plt.show()

    
#     for i in range(len(img_save)):
        
#         save_to = OUTPUT_DIR+img_save_dir[i]+'/'
#         if not os.path.exists(save_to):
#             os.makedirs(save_to)
#         imageio.imwrite(save_to + save_name +'.png', img_save[i])        
    return fig


def plt_learning_curve(train_loss_noted, valid_loss_noted, title = '', sub = '', st = '', ed = ''):
    fig = plt.figure()#figsize=(4, 4), dpi= 100)
    if not st =='':
        plt.plot(np.arange(len(train_loss_noted[st:ed])), train_loss_noted[st:ed], 'b', label = 'train')
        plt.plot(np.arange(len(valid_loss_noted[st:ed])), valid_loss_noted[st:ed], 'r', label = 'valid')
    else:
        plt.plot(np.arange(len(train_loss_noted)), train_loss_noted, 'b', label = 'train')
        plt.plot(np.arange(len(valid_loss_noted)), valid_loss_noted, 'r', label = 'valid')
    plt.title(title)
    plt.suptitle(sub, fontsize = 8, color = 'gray')
    plt.xlabel('epoch' , fontsize=10)
    plt.ylabel('loss', fontsize=10)
    plt.legend(loc='best')

    
#     plt.annotate(annot,
#             xy=(0, 0), xytext=(len(train_loss_noted), 0),
#             xycoords=('axes fraction', 'figure fraction'),
#             textcoords='offset points',
#             size=8, color = 'gray',  ha='left', va='bottom')
    plt.close()
    return fig
   
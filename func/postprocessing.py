import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import cv2

from skimage.color import gray2rgb

"""
Function which returns the labelled image after applying CRF

Usage:
img = skimage.transform.resize(img,output_shape=(256,256),order=1,preserve_range=False)
img = (img*255).astype(np.uint8)

mask = io.imread(mask_filename)
output = crf(img, mask, output_filname)

img = io.imread(image_filename)

"""

def crf(original_image, annotated_image):
    """
    @para original_image: RGB image, value=[0, 255], dtype: uint8, channel = 3
    @para annotated_image: annotated mask (in our case), value=[0,255], dtype: uint8, channel = 3

    return: crf mask (np.array) [255,255,3] only 'R' channel [0, 255]
    """
    # Converting annotated image to RGB if it is Gray scale
    if(len(annotated_image.shape)<3):
        annotated_image = gray2rgb(annotated_image)
    
    # imsave("testing2.png",annotated_image)
        
    #Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:,:,0] + (annotated_image[:,:,1]<<8) + (annotated_image[:,:,2]<<16)
    
    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    
    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    
    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat)) 
    
    #print("No of labels in the Image are ")
    #print(n_labels)
    
    
    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
    

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)
    

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
    

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                       compat=10,
                       kernel=dcrf.DIAG_KERNEL,
                       normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 5 steps 
    Q = d.inference(8)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
    return MAP.reshape(original_image.shape)


def find_cntr_condition(cv2_img, condition=0):
    '''input: cv2_img = cv2.imread(img_path), [0,255]'''
    ''' condition: area no larger than condition (to deal with black frame)''' 
    '''output: mask [255,255,3]; [0,255]'''
    b,g,r = cv2.split(cv2_img)  
    img = cv2.merge([r,g,b])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,th2 = cv2.threshold(gray,10,1,cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(contour) for contour in contours]
    #print(areas)
    if max(areas) > condition: ### remove black frame, if higher than this value,it means it get the black frame
        print('in')
        img2 = np.zeros( img.shape )
        img2[15:-15,15:-15] = img[15:-15,15:-15] * 255
        img2 = np.uint8(img2*255)
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        _,th2 = cv2.threshold(gray,10,1,cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(contour) for contour in contours]

    max_index = np.argmax(areas)
    #plt.imshow(img)
    #plt.show()
    mask = np.zeros(img.shape, np.uint8)
    k = cv2.drawContours(mask, contours, max_index, (255,255,255),-1)
    return k


def find_cntr(cv2_img):
    '''input: cv2_img = cv2.imread(img_path)'''
    '''output: mask '''
    b,g,r = cv2.split(cv2_img)  
    img = cv2.merge([r,g,b])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,th2 = cv2.threshold(gray,10,1,cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(contour) for contour in contours]
    max_index = np.argmax(areas)
    mask = np.zeros(img.shape, np.uint8)
    k = cv2.drawContours(mask, contours, max_index, (255,255,255),-1)
    return k
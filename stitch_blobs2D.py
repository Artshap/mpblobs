import numpy as np
import matplotlib.pyplot as plt
import porespy as ps
import numpy.random
from scipy import ndimage as ndi
import seaborn as sns
sns.set()

def create_mask(dims, blob):
    # x = np.random.randn(dims[1])
    # y = np.random.randn(dims[1])
    # # hm  = np.histogram2d(x, y, bins=dims[0])[0]
    hm  = (ps.generators.blobs(shape=(dims), porosity=None, blobiness=blob))
    hm_sm = ndi.gaussian_filter(hm, sigma=(0), order=0)
    return (hm_sm*255).astype('uint8')

def add_mp(im, mp_thickness):
    im_dist = ndi.distance_transform_edt(im == 1)

    mask = create_mask(im.shape, blob_mask)
    im_dist_new = im_dist * mask

    dist_thresh = im_dist_new.max() // mp_thickness

    # im_dist_new[im_dist_new == 0] = im_dist_new.max()

    im_dist_mask = im_dist_new < (dist_thresh)
    im_mp = im + im_dist_mask*2 

    plt.hist(mask.ravel(), bins = 128)
    plt.show()

    plt.hist(im_dist.ravel(), bins = 128)
    plt.show()


    plt.imshow(mask, interpolation='none')
    plt.axis('off')
    plt.show()

    plt.imshow(im_dist, interpolation='none')
    plt.axis('off')
    plt.show()


    plt.imshow(im_dist_new, interpolation='none')
    plt.axis('off')
    plt.show()
    plt.imsave(fname='1.5_edm.pdf', arr=im_dist, format='pdf')
    np.save(arr= im_dist, file = '1.5_edm')

    return im_mp

def generate_stitches(img, n, poro_micro, blob_micro):
    stitched_x, stitched_y = img.shape[1], img.shape[0] # size of single image
    img_stitched = ps.generators.blobs(shape=(stitched_y,stitched_x), porosity=poro_micro, blobiness=blob_micro)
    return img_stitched



# controlling parameters
dims = [1000, 1000] # 
poro_macro = 0.6
blob_macro = 0.6
poro_micro = 0.2
blob_micro = 10

blob_mask = 0.5
mp_thickness = 10 # factor for mp layer thickness
patches = 100 # nb of patches along each axis


im = ps.generators.blobs(shape=dims, porosity=poro_macro, blobiness=blob_macro)
im_mp = add_mp(im, mp_thickness)
im_st = generate_stitches(im_mp, patches, poro_micro=poro_micro, blob_micro=blob_micro)

im_mp_st = (im_mp == 3) * im_st + (im_mp == 1)



plt.imshow(im, interpolation='none')
plt.axis('off')
plt.show()


plt.imshow(im_mp, interpolation='none')
plt.axis('off')
plt.show()

plt.imshow(im_st, interpolation='none')
plt.axis('off')
plt.show()

plt.imshow(im_mp_st, interpolation='none')
plt.axis('off')
plt.show()


plt.imsave(fname='1_blobs.pdf', arr=im, format='pdf')
plt.imsave(fname='2_mp_as_phase.pdf', arr=im_mp, format='pdf')
plt.imsave(fname='3_mp_as_pores.pdf', arr=im_mp_st, format='pdf')


np.save(arr= im, file = '1_blobs')
np.save(arr= im_mp, file = '2_mp_as_phase')
np.save(arr= im_mp_st, file = '3_mp_as_pores')
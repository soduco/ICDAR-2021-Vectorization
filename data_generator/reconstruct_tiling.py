import os
import skimage.io as skio
import numpy as np
from PIL import Image
import argparse


def reconstruct_tiling(original_image_path, tile_path, tile_save_path, w_size):
    in_patches = os.listdir(tile_path)
    in_patches.sort()
    patches_paths = [os.path.join(tile_path, f) for f in in_patches]

    win_size = w_size
    pad_px = win_size // 2

    in_img = skio.imread(original_image_path)
    new_img = reconstruct_from_patches(patches_paths, win_size, pad_px, in_img.shape, np.uint8)

    new_img = Image.fromarray(new_img).convert('RGB')
    new_img.save(tile_save_path)


def reconstruct_from_patches(patches_paths, patch_size, step_size, image_size_2d, image_dtype):
    '''Adjust to take patch images directly.
    patch_size is the size of the tiles
    step_size should be patch_size//2
    image_size_2d is the size of the original image
    image_dtype is the data type of the target image

    Most of this could be guessed using an array of patches
    (except step_size but, again, it should be should be patch_size//2)
    '''
    i_h, i_w = np.array(image_size_2d[:2]) + (patch_size, patch_size)
    print(f"tmp img size: {i_h},{i_w}")
    p_h = p_w = patch_size
    img = np.zeros((i_h+p_h//2, i_w+p_w//2, 3), dtype=image_dtype)

    numrows = (i_h)//step_size-1
    numcols = (i_w)//step_size-1
    expected_patches = numrows * numcols
    print(f"numrows: {numrows}, numcols: {numcols}, total expected patches: {expected_patches}")
    if len(patches_paths) != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, got {len(patches_paths)}")

    patch_offset = step_size//2
    patch_inner = p_h-step_size
    print(f"patch_offset: {patch_offset}, patch_inner: {patch_inner}")

    for row in range(numrows):
        # print(f"Row {row}")
        for col in range(numcols):
            tt = skio.imread(patches_paths[row*numcols+col])
            tt_roi = tt[patch_offset:-patch_offset,patch_offset:-patch_offset]
#             print(f"Col {col}")
#             print(row*step_size, row*step_size+patch_inner,
#                 col*step_size, col*step_size+patch_inner)
            img[row*step_size:row*step_size+patch_inner,
                col*step_size:col*step_size+patch_inner] = tt_roi # +1??

    return img[step_size//2:-(patch_size+step_size//2),step_size//2:-(patch_size+step_size//2),...]

def main():
    parser = argparse.ArgumentParser(description='Tilling reconstruction')
    parser.add_argument('input_image_path', help='Path of original images.')
    parser.add_argument('input_tile_path', help='Path of tilings.')
    parser.add_argument('output_path', help='Path of saving output images.')

    args = parser.parse_args()
    reconstruct_tiling(args.input_image_path, args.input_tile_path, args.output_path)

if __name__ == '__main__':
    main()

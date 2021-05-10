import argparse
import cv2
import pdb

def border_cali(input_EPM_path, input_border_path, output_path):
    EPM = cv2.imread(input_EPM_path, cv2.IMREAD_GRAYSCALE)
    BOD = cv2.imread(input_border_path, cv2.IMREAD_GRAYSCALE)

    # for each pixel p(x,y) where value = 0 in the mask, you should set the
    # EPM pixel p(x,y) to 0 (sure no edge)
    # for p(x,y) = 255 in mask: set p(x,y) = 255 in EPM (sure edge)
    # for p(x,y) = 128 in mask: keep p(x,y) original value in EPM.
    EPM[BOD == 255] = 255
    # EPM[BOD == 255] = 255
    cv2.imwrite(output_path, EPM)
    print('Save calibration results to {}'.format(output_path))

def main():
    parser = argparse.ArgumentParser(description='Border Calibration.')
    parser.add_argument('input_EPM_path', help='Path of edge probaiblity map.')
    parser.add_argument('input_border_path', help='Path of border mask.')
    parser.add_argument('output_path', help='Path of update border image.')

    args = parser.parse_args()
    
    border_cali(args.input_EPM_path, args.input_border_path, args.output_path)

if __name__ == '__main__':
    main()

#!/usr/local/bin/python3
#
# Authors: adisoni-nsadhuva-svaddi
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#
import copy

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio

# calculate "Edge strength map" of an image
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)



# main program
#
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [ int(i) for i in sys.argv[2:4] ]
    gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    # load in image
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))
    
    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength_new_edge = edge_strength(input_image)
    edge_strength = edge_strength(input_image)

    imageio.imwrite('edges.png', uint8(255 * edge_strength / (amax(edge_strength))))

    img_rows = len(edge_strength)
    img_cols = len(edge_strength[0])
    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.
    edge_strength[edge_strength == 0] = 0.00001
    edge_strength = edge_strength / 255
    # airice_simple = [ image_array.shape[0]*0.25 ] * image_array.shape[1]
    airice_simple = argmax(edge_strength, axis=0).tolist()

    # for hmm
    for j in range(img_cols):
        # take max from previous column
        x = airice_simple[j - 1]
        c = 1
        for i in range(img_rows):
            # find the distance from top/total row size
            # higher the pixel higher the probability of being air-ice boundry
            # relative postion w.r.t to previous max
            edge_strength[i][j] -= (c / img_rows) - (abs(i - x) / img_rows)
            c += 1
    # airice_hmm = [ image_array.shape[0]*0.5 ] * image_array.shape[1]
    airice_hmm = argmax(edge_strength, axis=0).tolist()

    # human feedback
    for j in range(img_cols):
        x = gt_airice[1]
        c = 1
        for i in range(img_rows):
            edge_strength[i][j] -= (c / img_rows) - (abs(i - x) / img_rows)
            c += 1
    # airice_feedback= [ image_array.shape[0]*0.75 ] * image_array.shape[1]
    airice_feedback= argmax(edge_strength, axis=0).tolist()


    t = edge_strength_new_edge
    l = []
    for i in range(0, len(airice_simple)):
        if airice_simple[i] < img_rows-11:
            max_num = 0
            max_row = 0
            try:
                for j in range(airice_simple[i] + 10, img_rows):
                    if max_num < t[j][i]:
                        max_num = t[j][i]
                        max_row = j
                l.append(max_row)
            except:
                print(airice_simple[i] + 10, j)
                l.append(img_rows-1)
        else:
            l.append(img_rows -1)
    # icerock_simple = [ image_array.shape[0]*0.25 ] * image_array.shape[1]
    icerock_simple = l

    for j in range(img_cols):
        x = icerock_simple[j - 1]
        c = 10
        for i in range(x):
            edge_strength[i][j] = -math.inf
        for i in range(x, img_rows):
            edge_strength[i][j] -= (abs(i - x) / img_rows)
            c += 1
    # icerock_hmm = [ image_array.shape[0]*0.5 ] * image_array.shape[1]
    icerock_hmm = argmax(edge_strength, axis=0).tolist()

    for j in range(img_cols):
        x =  gt_icerock[1]
        c = 10
        for i in range(x):
            edge_strength[i][j] = -math.inf
        for i in range(x, img_rows):
            edge_strength[i][j] -= (abs(i - x) / img_rows)

    # icerock_feedback= [ image_array.shape[0]*0.75 ] * image_array.shape[1]
    icerock_feedback = argmax(edge_strength, axis=0).tolist()

    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback):#, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")

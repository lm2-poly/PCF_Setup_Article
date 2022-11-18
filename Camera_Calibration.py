# =============================================================================
# This code computes the parameters of the cameras that will be used to rescale
# the images and define the position of the origins of the coordinates system.
# Four videos must be provided with this code: the two videos of the black square
# and the two videos of the pipe in its rest position.
# =============================================================================

# Loading of the different libraries
import cv2 # Computer vision library
import numpy as np 
import matplotlib.pyplot as plt
from skimage import morphology #Image cleaning library

# Function used to compute a polynomial function given its order and coefficients
def f(x,b,order):
    ans = 0
    for i in range(order+1):
        ans += b[i]*x**(order-i)
    return ans

# =============================================================================
# Calibration of the scale of the cameras
# =============================================================================

# Resolution of the calibration videos
Pxl_x_cal = 1616    
Pxl_y_cal = 1240  

# Initializaion of the arrays containing the width and the height in pixels of the square
Side_Width = np.array([])
Side_Height = np.array([])
Front_Width = np.array([])
Front_Height = np.array([])

# Loading of the calibrations videos of the squares
cap_front = cv2.VideoCapture("C:\\Users\Morgan\Videos\Calibration_Shots\Front_Calibration_Square_HQ_1.mp4")
cap_side = cv2.VideoCapture("C:\\Users\Morgan\Videos\Calibration_Shots\Side_Calibration_Square_HQ_1.mp4")

# Loop reading the frames of the videos
i = 0
while True:
    if i%100 == 0:
        print(i)
    i += 1
    
    # Read the next frame of each video
    ret_front, frame_front = cap_front.read()
    ret_side, frame_side = cap_side.read()
    
    # Break the loop if one of the videos is finished
    if frame_side is None or frame_front is None:
        break
    
    # Conversion of the video in gray scale
    grey_side = cv2.cvtColor(frame_side, cv2.COLOR_BGR2GRAY)
    grey_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2GRAY)

    # Thresold values used for the binary image conversion
    # Has to be adjusted manually to get the proper result depending on the lighting
    thresh_front = 70
    thresh_side = 70
    
    # Conversion of the images in binary images
    # Each pixels can have the 0 or 255 values
    im_bw_front = cv2.threshold(grey_front, thresh_front, 255, cv2.THRESH_BINARY)[1]
    im_bw_side = cv2.threshold(grey_side, thresh_side, 255, cv2.THRESH_BINARY)[1]
   
    # Each binary image is converted to a boolean image where a black pixel is true
    # and a white pixel is false
    bool_front = im_bw_front > 150
    bool_side = im_bw_side > 150
        
    # Remove the isolated pixels islands smaller than a certain size
    # The island pixels size has to be adjusted depending on the video resolution
    cleaned_front = morphology.remove_small_objects(bool_front, min_size=50000)
    cleaned_front = morphology.remove_small_holes(cleaned_front, area_threshold=50000) 
    cleaned_side = morphology.remove_small_objects(bool_side, min_size=50000)
    cleaned_side = morphology.remove_small_holes(cleaned_side, area_threshold=50000)  

    # Returns lists with the x and y pixels coordinates of the black pixels
    Pixel_Position_side = np.where(np.invert(cleaned_side))
    Pixel_Position_front = np.where(np.invert(cleaned_front))
    
    # Computation of the width and height in pixels of the square in the frame
    unique_side_x, counts_side_x = np.unique(Pixel_Position_side[0],return_counts=True)
    unique_side_y, counts_side_y = np.unique(Pixel_Position_side[1],return_counts=True)
    Side_Width = np.append(Side_Width,counts_side_x[10:-10].mean())
    Side_Height = np.append(Side_Height,counts_side_y[10:-10].mean())
    unique_front_x, counts_front_x = np.unique(Pixel_Position_front[0],return_counts=True)
    unique_front_y, counts_front_y = np.unique(Pixel_Position_front[1],return_counts=True)
    Front_Width = np.append(Front_Width,counts_front_x[10:-10].mean())
    Front_Height = np.append(Front_Height,counts_front_y[10:-10].mean())

    # Ploting of the image every 100 images
    if i%100 == 0:
        fig, (axfront,axside) = plt.subplots(1,2)
        axfront.set_title("Calibration camera 1")
        axfront.imshow(cleaned_front,cmap='gray')
        axfront.set_xlabel("X coordinates in pixels")
        axfront.set_ylabel("Y coordinates in pixels")
        axside.set_title("Calibration camera 2")
        axside.imshow(cleaned_side,cmap='gray')
        axside.set_xlabel("X coordinates in pixels")
        axside.set_ylabel("Y coordinates in pixels")
        plt.show()

# Release of the videos
cap_front.release()
cap_side.release()

# Physical size of the square in meters
Width = 12.2*10**-2
Height = 12.2*10**-2

# Computation of the image scale in pixels/mm
scale_Width_Side = Width/(Side_Width.mean())
scale_Height_Side = Height/(Side_Width.mean())
scale_Width_Front = Width/(Front_Width.mean())
scale_Height_Front = Height/(Front_Width.mean())

# =============================================================================
# Calibration of the coordinate system of the cameras
# =============================================================================

# Physical lenght of the pipe in meters
# Represents the visible lenght of the pipe and not the total lenght
Pipe_Lenght = 42*10**-2

# Resolution of the calibrations videos
Pxl_x_static = 1616    
Pxl_y_static = 1240

# Initializaion of the arrays containing the positions of the pipe
Tip_Front_array = np.array([])  
Top_Side_array = np.array([])  
Top_Front_array = np.array([])  
Tip_Side_array = np.array([])  
Center_Side_array = np.array([])
Center_Front_array = np.array([])

# Loading of the calibrations videos of the squares
cap_front = cv2.VideoCapture("C:\\Users\Morgan\Videos\Calibration_Shots\Front_Calibration_HQ.mp4")
cap_side = cv2.VideoCapture("C:\\Users\Morgan\Videos\Calibration_Shots\Side_Calibration_HQ.mp4")

# Loop reading the frames of the videos
i = 0
while True:
    if i%100 == 0:
        print(i)
    i += 1
    
    # Read the next frame of each video
    ret_front, frame_front = cap_front.read()
    ret_side, frame_side = cap_side.read()

    # Break the loop if one of the videos is finished
    if frame_side is None or frame_front is None:
        break
    
    # Conversion of the video in gray scale
    grey_side = cv2.cvtColor(frame_side, cv2.COLOR_BGR2GRAY)
    grey_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2GRAY)

    # Thresold values used for the binary image conversion
    # Has to be adjusted manually to get the proper result depending on the lighting
    thresh_front = 100
    thresh_side = 100
    
    # Conversion of the images in binary images
    # Each pixels can have the 0 or 255 values
    im_bw_front = cv2.threshold(grey_front, thresh_front, 255, cv2.THRESH_BINARY)[1]
    im_bw_side = cv2.threshold(grey_side, thresh_side, 255, cv2.THRESH_BINARY)[1]
   
    # Each binary image is converted to a boolean image where a black pixel is true
    # and a white pixel is false
    bool_front = im_bw_front > 150
    bool_side = im_bw_side > 150
    
    # Remove the isolated pixels islands smaller than a certain size
    # The island pixels size has to be adjusted depending on the video resolution
    cleaned_front = morphology.remove_small_objects(bool_front, min_size=50000)
    cleaned_front = morphology.remove_small_holes(cleaned_front, area_threshold=50000) 
    cleaned_side = morphology.remove_small_objects(bool_side, min_size=50000)
    cleaned_side = morphology.remove_small_holes(cleaned_side, area_threshold=50000)  

    # Returns lists with the x and y pixels coordinates of the black pixels
    Pixel_Position_side = np.where(np.invert(cleaned_side))
    Pixel_Position_front = np.where(np.invert(cleaned_front))
    
    # Computation of the position of the tip of the pipe and of the centerline
    unique_side_x, counts_side_x = np.unique(Pixel_Position_side[0],return_counts=True)
    unique_side_y, counts_side_y = np.unique(Pixel_Position_side[1],return_counts=True)
    Tip_Side = unique_side_x[-1]
    Tip_Side_array = np.append(Tip_Side_array,Tip_Side)
    Top_Side_array = np.append(Top_Side_array,Tip_Side-Pipe_Lenght/scale_Width_Side)  
    Center_Side = (unique_side_y*counts_side_y).sum()/(counts_side_y.sum())
    Center_Side_array = np.append(Center_Side_array,Center_Side)
    unique_front_x, counts_front_x = np.unique(Pixel_Position_front[0],return_counts=True)
    unique_front_y, counts_front_y = np.unique(Pixel_Position_front[1],return_counts=True)
    Tip_Front = unique_front_x[-1]
    Tip_Front_array = np.append(Tip_Front_array,Tip_Front)
    Top_Front_array = np.append(Top_Front_array,Tip_Front-Pipe_Lenght/scale_Width_Front)
    Center_Front = (unique_front_y*counts_front_y).sum()/(counts_front_y.sum())
    Center_Front_array = np.append(Center_Front_array,Center_Front)

    # Ploting of the results every 100 images
    if i%100 == 0:
        fig, (axfront,axside) = plt.subplots(1,2)
        axfront.imshow(cleaned_front,cmap='gray')
        axfront.plot([0,1240],[Tip_Front,Tip_Front])
        axfront.plot([0,1240],[Tip_Front-Pipe_Lenght/scale_Width_Front,Tip_Front-Pipe_Lenght/scale_Width_Front])
        axfront.plot([Center_Front,Center_Front],[0,1616])
        axfront.set_title("Calibration camera 1")
        axfront.set_xlabel("X coordinates in pixels")
        axfront.set_ylabel("Y coordinates in pixels")
        axside.imshow(cleaned_side,cmap='gray')
        axside.plot([0,1240],[Tip_Side,Tip_Side])
        axside.plot([0,1240],[Tip_Side-Pipe_Lenght/scale_Width_Side,Tip_Side-Pipe_Lenght/scale_Width_Side])
        axside.plot([Center_Side,Center_Side],[0,1616])
        axside.set_title("Calibration camera 2")
        axside.set_xlabel("X coordinates in pixels")
        axside.set_ylabel("Y coordinates in pixels")
        plt.show()

# Release of the videos
cap_front.release()
cap_side.release()

Center_Side = Center_Side_array.mean()
Center_Front = Center_Front_array.mean()
Tip_Side = Tip_Side_array.mean()
Top_Side = Top_Side_array.mean()
Tip_Front = Tip_Front_array.mean()
Top_Front = Top_Front_array.mean()

# List of the parameters to be saved
Front_Parameters = [Center_Front,Tip_Front,Top_Front,scale_Width_Front,scale_Height_Front]
Side_Parameters = [Center_Side,Tip_Side,Top_Side,scale_Width_Side,scale_Height_Side]

# Saving of the parameters in txt files
np.savetxt("C:\\Users\Morgan\Videos\Rotated_Videos\Front_Parameters.txt",Front_Parameters)
np.savetxt("C:\\Users\Morgan\Videos\Rotated_Videos\Side_Parameters.txt",Side_Parameters)
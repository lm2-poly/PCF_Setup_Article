# =================================================================================
# Treatment of the images when the movement is planar and only one camera is needed
# =================================================================================

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

# Loading of the camera parameters from the calibration step
Front_Parameters = np.loadtxt("C:\\Users\Morgan\Videos\Calibration_Shots\Front_Parameters.txt")
Center_Front,Tip_Front,Top_Front,scale_Width_Front,scale_Height_Front = Front_Parameters

# Order of the polynomial function used to fit the position of the pipe
Order_front = 10

# Resolution of the images
Pxl_x_dynamic = 808    
Pxl_y_dynamic = 620

# Modification of the parameters to account for the resolution of the images that 
# is twice lower than during calibrations
scale_Width_Front = scale_Width_Front*2
scale_Height_Front = scale_Height_Front*2
Center_Front = Center_Front/2
Tip_Front = Tip_Front/2
Top_Front = Top_Front/2

# Loading of the video to be treated
cap_front = cv2.VideoCapture("C:\\Users\Morgan\Videos\Videos_Pipe_0.3125in_46cm\Rotated_Front_Hz.mp4")

# Initialization of the lists
unexploitable_frames = np.array([])
b_front_list = np.zeros((4500,Order_front+1))
z_tip_front_list = np.array([])
x_tip_list = np.array([])
X = np.zeros((4500,100))
Z = np.zeros((4500,100))
T = np.zeros((4500,100))

# Loop reading the frames of the video
i = 0
while True:
    i += 1
    
    # Read the next frame of the video
    ret_front, frame_front = cap_front.read()
    
    # Break the loop if one of the the video is finished
    if frame_front is None:
        break

    # Thresolds values used for the adaptive thresolds function and to convert 
    # the images in binary images. These parameters must be tunned depending 
    # on the lighting and on the video
    thresh_front = 401
    C_front = 40
    
    # Conversion of the images in grey scales images
    grey_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2GRAY)
    
    # Conversion of the images in black and white images
    im_bw_front = cv2.adaptiveThreshold(grey_front,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,thresh_front,C_front)

    # The top of the image is forced to white between the top and the origin of the pipe
    im_bw_front[0:int(Top_Front)] = 255*np.ones((int(Top_Front),grey_front.shape[1]))
    im_bw_front[719:-1] = 255*np.ones((88,grey_front.shape[1]))

    # Conversion of the images in binary images
    bool_front = im_bw_front > 150
    
    # Removal of the unattached pixel islands
    cleaned_front = morphology.remove_small_objects(bool_front, min_size=4000)
    cleaned_front = morphology.remove_small_holes(cleaned_front, area_threshold=4000) 

    # Computation of lists of x and z coordinates of the black pixels
    Pixel_Coor_Front = np.where(np.invert(cleaned_front))

    # Number of black pixels
    Front_Pipe_nPxl = Pixel_Coor_Front[0].shape[0]

    # Conversion of the coordinates from the pixels coordinates system to the 
    # physical coordinates system
    z_front = (Pixel_Coor_Front[0]-Top_Front*np.ones(Front_Pipe_nPxl))*scale_Width_Front
    x = (Pixel_Coor_Front[1]-Center_Front*np.ones(Front_Pipe_nPxl))*scale_Width_Front
    
    # If no pixels are black the image is considered unxeploitable 
    # This can be caused by a water splash
    if z_front.shape[0] == 0:
        unexploitable_frames = np.append(unexploitable_frames,i-1)
        b_front_list[i-1] = np.zeros(Order_front+1)
        z_tip_front_list = np.append(z_tip_front_list,0)
        x_tip_list = np.append(x_tip_list,0)

        
    else:
        # Fitting of the polynomial functions on the x and z coordinates 
        # of the pipe.
        b_front = np.polyfit(z_front,x,Order_front)
        b_front_list[i-1] = b_front
    
        # Finding of the position of the tip of the pipe
        z_tip_front = z_front[-1]
        z_tip_front_list = np.append(z_tip_front_list,z_tip_front)
        
        # Finding of the deflection of the pipe at the tip
        x_tip = f(z_tip_front,b_front,Order_front)
        x_tip_list = np.append(x_tip_list,x_tip)
        
        # Ploting of the results if necessary
        n_samples = 100
        z_test_front = np.linspace(0,z_tip_front,n_samples)
        x_test = f(z_test_front,b_front,Order_front)
        if i%100 == 0:
            fig, axfront = plt.subplots(1,1,figsize=(5,5))
            axfront.plot(x,z_front,".")
            axfront.plot(x_test,z_test_front)
            axfront.plot(x_tip,z_tip_front,"+")
            axfront.set_title("Front view i = "+str(i))
            axfront.set_xlim([-0.1,0.2])
            axfront.set_ylim([-0.2,0.5])
            axfront.invert_yaxis()
            plt.show()
    
    # Ploting of the treated frames
    cv2.imshow('Front', 255*cleaned_front.astype(np.uint8))
    cv2.imshow('Original',grey_front)
    cv2.waitKey(1) 
    
# Release of the videos
cap_front.release()

# Save the results from the treatment in txt files
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.3125in_46cm\XYZT_tip_Hz.txt",np.array([z_tip_front_list,x_tip_list]))
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.3125in_46cm\X_Hz.txt",X)
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.3125in_46cm\Z_Hz.txt",Z)
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.3125in_46cm\T_Hz.txt",T)
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.3125in_46cm\B_front_Hz.txt",b_front_list)

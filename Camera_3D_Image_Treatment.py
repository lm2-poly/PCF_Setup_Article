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

# Loading of the cameras parameters from the calibration step
Front_Parameters = np.loadtxt("C:\\Users\Morgan\Videos\Calibration_Shots\Front_Parameters.txt")
Side_Parameters = np.loadtxt("C:\\Users\Morgan\Videos\Calibration_Shots\Side_Parameters.txt")
Center_Front,Tip_Front,Top_Front,scale_Width_Front,scale_Height_Front = Front_Parameters
Center_Side,Tip_Side,Top_Side,scale_Width_Side,scale_Height_Side = Side_Parameters

# Order of the polynomial function used to fit the position of the pipe
Order_front = 10
Order_side = 10

# Resolution of the images
Pxl_x_dynamic = 808    
Pxl_y_dynamic = 620

# Distance between the cameras and the central calibration plans
Dx = 60*10**-2
Dy = 66*10**-2

# Modification of the parameters to account for the resolution of the images that 
# is twice lower than during calibrations
scale_Width_Side = scale_Width_Side*2
scale_Height_Side = scale_Height_Side*2
scale_Width_Front = scale_Width_Front*2
scale_Height_Front = scale_Height_Front*2
Center_Side = Center_Side/2
Center_Front = Center_Front/2
Tip_Side = Tip_Side/2
Top_Side = Top_Side/2
Tip_Front = Tip_Front/2
Top_Front = Top_Front/2

# Computation of the vertical position of the cameras
z_middle_front = (Pxl_x_dynamic/2-Top_Front)*scale_Width_Front
z_middle_side = (Pxl_x_dynamic/2-Top_Side)*scale_Width_Side

# Loading of the video to be treated
cap_front = cv2.VideoCapture("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\Rotated_Front_Hz.mp4")
cap_side = cv2.VideoCapture("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\Rotated_Side_Hz.mp4")

# Initialization of the lists
unexploitable_frames = np.array([])
b_front_list = np.zeros((4500,Order_front+1))
b_side_list = np.zeros((4500,Order_side+1))
z_tip_front_list = np.array([])
x_tip_list = np.array([])
z_tip_side_list = np.array([])
y_tip_list = np.array([])

# Loop reading the frames of the video
i = 0
while True:
    i += 1

    # Read the next frames of the videos
    ret_front, frame_front = cap_front.read()
    ret_side, frame_side = cap_side.read()
    
    # Break the loop if one of the the videos is finised
    if frame_side is None or frame_front is None:
        break

    # Thresolds values used for the adaptive thresolds function and to convert 
    # the images in binary images. These parameters must be tunned depending 
    # on the lighting and on the video
    thresh_front = 251
    C_front = 20
    thresh_side = 255
    C_side = 20
    
    # Conversion of the images in grey scales images
    grey_side = cv2.cvtColor(frame_side, cv2.COLOR_BGR2GRAY)
    grey_front = cv2.cvtColor(frame_front, cv2.COLOR_BGR2GRAY)
    
    # Conversion of the images in black and white images
    im_bw_front = cv2.adaptiveThreshold(grey_front,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,thresh_front,C_front)
    im_bw_side = cv2.adaptiveThreshold(grey_side,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,thresh_side,C_side)

    # The top of the image is forced to white between the top and the origin of the pipe
    im_bw_side[0:int(Top_Side)] = 255*np.ones((int(Top_Side),grey_side.shape[1]))
    im_bw_front[0:int(Top_Front)] = 255*np.ones((int(Top_Front),grey_side.shape[1]))

    # Conversion of the images in binary images
    bool_side = im_bw_side > 150
    bool_front = im_bw_front > 150
    
    # Removal of the unattached pixel islands
    cleaned_front = morphology.remove_small_objects(bool_front, min_size=8000)
    cleaned_front = morphology.remove_small_holes(cleaned_front, area_threshold=8000) 
    cleaned_side = morphology.remove_small_objects(bool_side, min_size=8000)
    cleaned_side = morphology.remove_small_holes(cleaned_side, area_threshold=8000)  

    # Computation of lists of x and z coordinates of the black pixels
    Pixel_Coor_Front = np.where(np.invert(cleaned_front))
    Pixel_Coor_Side = np.where(np.invert(cleaned_side))
    
    # Number of black pixels
    Front_Pipe_nPxl = Pixel_Coor_Front[0].shape[0]
    Side_Pipe_nPxl = Pixel_Coor_Side[0].shape[0]

    # Conversion of the coordinates from the pixels coordinates system to the 
    # physical coordinates system
    z_front = (Pixel_Coor_Front[0]-Top_Front*np.ones(Front_Pipe_nPxl))*scale_Width_Front
    x = (Pixel_Coor_Front[1]-Center_Front*np.ones(Front_Pipe_nPxl))*scale_Width_Front
    z_side = (Pixel_Coor_Side[0]-Top_Side*np.ones(Side_Pipe_nPxl))*scale_Width_Side
    y = (Pixel_Coor_Side[1]-Center_Side*np.ones(Side_Pipe_nPxl))*scale_Width_Side
    
    # If no pixels are black the image is considered unxeploitable 
    # This can be caused by a water splash
    if z_side.shape[0] == 0 or z_front.shape[0] == 0:
        unexploitable_frames = np.append(unexploitable_frames,i-1)
        b_front_list[i-1] = np.zeros(Order_front+1)
        b_side_list[i-1] = np.zeros(Order_side+1)
        z_tip_front_list = np.append(z_tip_front_list,0)
        z_tip_side_list = np.append(z_tip_side_list,0)
        x_tip_list = np.append(x_tip_list,0)
        y_tip_list = np.append(y_tip_list,0)
        # print("Side = "+str(z_side.shape[0])+" and Front = "+str(z_front.shape[0]))
    
    else:
        # Fitting of the polynomial functions on the x and z coordinates 
        # of the pipe.
        b_front = np.polyfit(z_front,x,Order_front)
        b_side = np.polyfit(z_side,y,Order_side)
        b_front_list[i-1] = b_front
        b_side_list[i-1] = b_side
    
        # Finding of the position of the tip of the pipe
        z_tip_front = z_front[-1]
        z_tip_front_list = np.append(z_tip_front_list,z_tip_front)
        z_tip_side = z_side[-1]
        z_tip_side_list = np.append(z_tip_side_list,z_tip_side)
        
        # Finding of the deflection of the pipe at the tip
        y_tip = y[-1]
        y_tip_list = np.append(y_tip_list,y_tip)
        x_tip = x[-1]
        x_tip_list = np.append(x_tip_list,x_tip)
        
        # Ploting of the results if necessary
        n_samples = 100
        z_test_front = np.linspace(-Top_Front*scale_Width_Front,(Pxl_x_dynamic-Top_Front)*scale_Width_Front,n_samples)
        x_test = f(z_test_front,b_front,Order_front)
        z_test_side = np.linspace(-Top_Side*scale_Width_Side,(Pxl_x_dynamic-Top_Side)*scale_Width_Side,n_samples)
        y_test = f(z_test_side,b_side,Order_side)
        if i%100 == 0:
            fig, (axfront,axside) = plt.subplots(1,2)
            axfront.plot(x,z_front,".")
            axfront.plot(x_test,z_test_front)
            axfront.plot(x_tip,z_tip_front,"+")
            axfront.set_title("Front view i = "+str(i))
            axfront.set_xlim([-0.2,0.2])
            axfront.set_ylim([0,0.5])
            axfront.invert_yaxis()
            axside.plot(y,z_side,".")
            axside.plot(y_test,z_test_side)
            axside.plot(y_tip,z_tip_side,"+")
            axside.set_title("Side view i = "+str(i))
            axside.set_xlim([-0.2,0.2])
            axside.set_ylim([0,0.5])
            axside.invert_yaxis()
            plt.show()
          
    # Ploting of the treated frames
    cv2.imshow('Front', 255*cleaned_front.astype(np.uint8))
    cv2.imshow('Side', 255*cleaned_side.astype(np.uint8))
    cv2.waitKey(1) 

# Release of the videos
cap_front.release()
cap_side.release()

# Parallax correction of the position of the tip of the pipe on each frame
Ni = x_tip_list.shape[0]
x_tip_corrected_list = x_tip_list*Dx*(Dy*np.ones(Ni)+y_tip_list)/(Dy*Dx*np.ones(Ni)+x_tip_list*y_tip_list)
y_tip_corrected_list = y_tip_list*Dy*(Dx*np.ones(Ni)-x_tip_list)/(Dy*Dx*np.ones(Ni)+y_tip_list*x_tip_list)
z_tip_corrected_list_front = (z_tip_front_list-z_middle_front*np.ones(Ni))*(Dy*np.ones(Ni)+y_tip_corrected_list)/Dy+z_middle_front*np.ones(Ni)
z_tip_corrected_list_side = (z_tip_side_list-z_middle_side*np.ones(Ni))*(Dx*np.ones(Ni)-x_tip_corrected_list)/Dx+z_middle_side*np.ones(Ni)
z_tip_global = (z_tip_corrected_list_front + z_tip_corrected_list_side)/2
height_error = ((((z_tip_corrected_list_front-z_tip_corrected_list_side)/z_tip_global)**2)**0.5)*100

# Save of the positions of the tip and of the height errors
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.3125in_46cm\XYZT_tip_Hz.txt",np.array([x_tip_corrected_list,y_tip_corrected_list,z_tip_global]))
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.3125in_46cm\height_error_Hz.txt",height_error)

# Initialization of the corrected position arrays
X = np.ones((4500,100))
Y = np.ones((4500,100))
Z = np.ones((4500,100))
T = np.array([])
b_front_corrected = np.ones((4500,Order_front+1))
b_side_corrected = np.ones((4500,Order_side+1))

# Loop of parallax correction on the whole lenght of the pipe
for i in range(4500):
    # If the frame is unexploitable everything is set to 0
    if i in unexploitable_frames:
        X[i] = np.zeros(100)
        Y[i] = np.zeros(100)
        Z[i] = np.zeros(100)
        T = np.append(T,i)
        b_front_corrected[i] = np.zeros(Order_front+1)
        b_side_corrected[i] = np.zeros(Order_side+1)
    else:
        # Coefficients of the polynom before correction
        b_front = b_front_list[i]
        b_side = b_side_list[i]
        
        # Creation of the lists of 100 points along the pipe in the front camera
        z_tip_front = z_tip_front_list[i]
        z_tip_side = z_tip_side_list[i]
        x_tip = x_tip_list[i]
        y_tip = y_tip_list[i]
        
        # Creation of the lists of 100 points along the pipe in the side camera
        z_list_front = np.linspace(0,z_tip_front,100)
        z_list_side = np.linspace(0,z_tip_side,100)
        x_list = f(z_list_front,b_front,Order_front)
        y_list = f(z_list_side,b_side,Order_side)
        
        # Correction of the coordinates of each terms of the lists
        x_corrected_list = x_list*Dx*(Dy*np.ones(100)+y_list)/(Dy*Dx*np.ones(100)+x_list*y_list)
        y_corrected_list = y_list*Dy*(Dx*np.ones(100)-x_list)/(Dy*Dx*np.ones(100)+y_list*x_list)
        z_corrected_list_front = (z_list_front-z_middle_front*np.ones(100))*(Dy*np.ones(100)+y_corrected_list)/Dy+z_middle_front*np.ones(100)
        z_corrected_list_side = (z_list_side-z_middle_side*np.ones(100))*(Dx*np.ones(100)-x_corrected_list)/Dx+z_middle_side*np.ones(100)
        z_global = (z_corrected_list_front + z_corrected_list_side)/2
        
        # New polynomial fitting on the corrected images
        b_front_corrected[i] = np.polyfit(z_global,x_corrected_list,Order_front)
        b_side_corrected[i] = np.polyfit(z_global,y_corrected_list,Order_side)
        X[i] = x_corrected_list
        Y[i] = y_corrected_list
        Z[i] = z_global
        T = np.append(T,i)

# Save the results from the treatment in txt files
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_36cm\X_Hz.txt",X)
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_36cm\Y_Hz.txt",Y)
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_36cm\Z_Hz.txt",Z)
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_36cm\T_Hz.txt",T)
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_36cm\B_front_Hz.txt",b_front_corrected)
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_36cm\B_side_Hz.txt",b_side_corrected)
   
#Import packages

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math
import os
from moviepy.editor import VideoFileClip


class LaneDetection:

    def __init__(self):
        #Defining parameters
        self.kernel_size = 5
        self.low_threshold = 50
        self.high_threshold = 150
        
        self.min_line_len = 40
        self.max_line_gap = 20
        
        self.threshold = 15
        
        self.rho = 2
        self.theta = np.pi/180
        
        #threshold = 30
        #min_line_len = 40
        #max_line_gap = 5
        #threshold = 15
        #min_line_len = 40
        #max_line_gap = 20
        #self.min_line_len = 15
        #self.max_line_gap = 5
        
        
        return


    #Files operations
    
    def load_image(self, image_path):
        #Load an image 
        image = mpimg.imread(image_path)
        #Printing out image data
        print('This image is: ', type(image), ' with dimesions: ', image.shape)
        return image
            
    def save_image(self, img, img_path):
        #Save an image
        mpimg.imsave(img_path, img)
        print('Image saved: ', img_path)
        return
        
    def process_image_file_path(self, image_file_path):
        #load the image
        initial_image = self.load_image(image_file_path)
        processed_image = self.pipeline(initial_image)

        image_file_name = os.path.basename(image_file_path)
        new_image_file_path = os.path.dirname(image_file_path) + '/' + os.path.splitext(image_file_name)[0] + '_processed' + os.path.splitext(image_file_name)[1]
        self.save_image(processed_image, new_image_file_path)
        #print ('save to')
               #format(new_img_file_path)
        return

    #Image processing
    
    def grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=2):
        """
        NOTE: this is the function you might want to use as a starting point once you want to 
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).  
        
        Think about things like separating line segments by their 
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of 
        the lines and extrapolate to the top and bottom of the lane.
        
        This function draws `lines` with `color` and `thickness`.    
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        #for line in lines:
        #    for x1,y1,x2,y2 in line:
        #        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        
        
        
        xleft = []
        yleft = []
        
        xright = []
        yright = []
        for line in lines: 
            for x1,y1,x2,y2 in line:
                m = ((y2-y1)/(x2-x1))
                if(m>0):
                    xright.extend([x1,x2])
                    yright.extend([y1,y2])    
                else:
                    xleft.extend([x1,x2])
                    yleft.extend([y1,y2])

        
        if  xright and yright:
            self.plot_line(img,xright,yright,img.shape[1]/2+20,img.shape[1])
        
        if  xleft and yleft:
            self.plot_line(img,xleft,yleft,0,img.shape[1]/2-20)
        
        return
    
        
    def plot_line(self, img,x,y,x0,x1, color=[255, 0, 0], thickness=6, order = 1):
        
        if len(x) == 2:  
            c = np.polyfit(x,y,1)
            error = np.matrix(np.eye(2))*1000 
        else:
            c, error  = np.polyfit(x,y,1,cov=True)
            
        
        f = np.poly1d(c)
       
        x_new = np.linspace(x0,x1, 200)
        y_new = f(x_new)
        

        
        for i in range(0,len(x_new)-1):
            
            cv2.line(img, (int(x_new[i]), int(y_new[i])), (int(x_new[i+1]), int(y_new[i+1])), color, thickness)
            
        return c, error

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.
            
        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        
        
        
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        
        
        
        self.draw_lines(line_img, lines)
        return line_img


    
        
        
    # Python 3 has support for cool math symbols.

    def weighted_img(self, img, initial_img, α=0.8, β=1., λ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.
        
        `initial_img` should be the image before any processing.
        
        The result image is computed as follows:
        
        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)
        
    #Processing pipeline

    def pipeline(self, initial_image):
        
        #Show initial unprocessed image
        #plt.imshow(initial_img)
        #self.save_image(initial_image, './pipeline/initial.jpg')
        

        
        #Convert into grayscale image
        grayscale_image = self.grayscale(initial_image)
        #plt.imshow(grayscale_image, cmap='gray')
        #self.save_image(grayscale_image, './pipeline/grayscale_image.jpg')
        
        #Guassian blur
        
        blur_image = self.gaussian_blur(grayscale_image, self.kernel_size)
        #plt.imshow(blur_image, cmap='gray')
        #self.save_image(blur_image, './pipeline/blur_image.jpg')
        
        #Canny edges detection
        canny_edges = self.canny(blur_image, self.low_threshold, self.high_threshold)
        #self.save_image(canny_edges, './pipeline/canny_edges.jpg')

        #plt.imshow(canny_edges, cmap='gray')
        
        #Crop region of interest

        ysize = initial_image.shape[0]
        xsize = initial_image.shape[1]
        if ysize == 540 and xsize == 960:
            left_bottom, left_top, right_top, right_bottom = (10,539),(460,320), (495,320), (930,539)
                                                            #(0,539),(431, 297), (527,297), (959,539)
        elif ysize == 720 and xsize == 1280:
            left_bottom, left_top, right_top, right_bottom = (160,665),(600,440), (750,440), (1120,665)
            #left_bottom, left_top, right_top, right_bottom = (100,720),(580,450), (732,450), (1240,720)
        else:
            left_bottom = [0, ysize-1]
            right_bottom = [xsize-1, ysize-1]
            left_top = [int(xsize/2-xsize/20)-1, int(ysize*0.55)]
            right_top = [int(xsize/2+xsize/20)-1, int(ysize*0.55)]
     
        
        vertices = np.array([[(left_bottom),(left_top), (right_top), (right_bottom)]], dtype=np.int32)
        
     
        
        roi_img = self.region_of_interest(canny_edges, vertices)
        #plt.imshow(roi_img, cmap='gray')
        #self.save_image(roi_img, './pipeline/roi_img.jpg')
        
        #Hough line detection

        lines_image = self.hough_lines(roi_img, self.rho, self.theta, self.threshold, self.min_line_len, self.max_line_gap)
        #plt.imshow(lines_image, cmap='gray')
        #self.save_image(lines_image, './pipeline/lines_image.jpg')
    
        #blendign the images
        α = 0.8
        β = 1
        λ = 0
        final_image = self.weighted_img(lines_image, initial_image, α, β, λ)
        #plt.imshow(final_image, cmap='gray')

    
        return final_image

        
     
       

     
    def image_processing(self):
        print ('Start processing images')
        image_file_paths = ['solidWhiteCurve.jpg',
                            'solidWhiteRight.jpg',
                            'solidYellowCurve.jpg',
                            'solidYellowCurve2.jpg',
                            'solidYellowLeft.jpg',
                            'whiteCarLaneSwitch.jpg']
        image_file_paths = ['./test_images/'+ file_path for file_path in image_file_paths]
        for image_file_path in image_file_paths:
            print ('Process image file', image_file_path)
            self.process_image_file_path(image_file_path)
        print ('Done with processing images')
        return
    
    #Video processing
    def video_processing(self, input_video, output_video):
        clip1 = VideoFileClip(input_video)
        white_clip = clip1.fl_image(self.pipeline)
        #white_clip = clip1.fl_image(self.advanced_pipeline)
        white_clip.write_videofile(output_video, audio=False)
        return

    def run(self):
        self.image_processing()
        self.video_processing('challenge.mp4','extra.mp4')
        self.video_processing('solidWhiteRight.mp4','white.mp4')
        self.video_processing('solidYellowLeft.mp4','yellow.mp4')
        
        
        

if __name__ == "__main__":   
    obj= LaneDetection()
    obj.run()
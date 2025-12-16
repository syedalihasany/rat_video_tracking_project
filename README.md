To run the entire pipeline first download the SPDB video from the drive link and paste it in the parent directory (where this readme file is located)
1. Make sure the system requirements are satisifed as per the requirements.txt file (if not install the relevant modules & libraries)
2. run the find_crop_coordinates.py file to identify the top left and bottom right corner of pixel coordinates that would allow us to keep just the rat and bedding in the frame (I already figured the coordinates out so you can skip this step)
3. edit the x1, y1, x2, y2 variables of crop_video_based_on_coordinates.py using the top left and bottom right coordinates you found in step 2. (You may skip this step since I have the default values written in the script)
4. run crop_video_based_on_coordinates.py script to get the cropped video for the next step
5. navigate to the code folder and find the stage1_mog2_relations.py file
6. run stage1_mog2_relations.py script to get the csv file that describes the relations between the blobs (it also produces the overlay of the relations on the video)
7. run stage2_kalman_from_csv.py script to get the final output (this is the video that has the tracking of the rat) 

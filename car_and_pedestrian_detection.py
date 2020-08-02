#!/c/Users/SYED/AppData/Local/Programs/Python/Python38-32/python

# Import neccessary modules

import cv2
import ffmpeg

# Load car image

#car_img_file="C:/Users/SYED/Desktop/Car_and_pedestrian_detection/many_cars_on_road.jpg"

#ffmpeg -i "trial_videos/PedestriansCompilation.mp4" -vf  "setpts=0.25*PTS" "trial_videos/PedestriansCompilation.mp4"

# Load video

#video=cv2.VideoCapture("C:/Users/SYED/Desktop/Car_and_pedestrian_detection/trial_videos/Tesla Autopilot Dashcam Compilation 2018 Version.mp4")
video=cv2.VideoCapture("C:/Users/SYED/Desktop/Car_and_pedestrian_detection/trial_videos/almost_clear.mp4")
#video=cv2.VideoCapture("C:/Users/SYED/Desktop/Car_and_pedestrian_detection/trial_videos/not_clear.mp4")
#video=cv2.VideoCapture("C:/Users/SYED/Desktop/Car_and_pedestrian_detection/trial_videos/good_but_slow.mp4")

# Load the car classifier xml file and pedestrian classifier xml file

car_tracker_file="C:/Users/SYED/Desktop/Car_and_pedestrian_detection/xml_files/cars.xml"
pedestrian_tracker_file="C:/Users/SYED/Desktop/Car_and_pedestrian_detection/xml_files/humans.xml"

# Create car tracker

car_tracker=cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker=cv2.CascadeClassifier(pedestrian_tracker_file)

while True:

	success,frame=video.read()

	#print(video.read())

	if success:

		# Change car image file to cv2 object

		#car_img=cv2.imread(frame)

		# Convert image to grey

		grey_car_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		grey_pedestrian_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


	else :

		break

	# Spot everything looking like a car and return its (x_coordinate,y_coordinate,width,height)

	cars=car_tracker.detectMultiScale(grey_car_img)
	pedestrians=pedestrian_tracker.detectMultiScale(grey_pedestrian_img)

	# Loop through each car and draw a rectangle around it

	for x,y,w,h in cars:

		cv2.rectangle(frame, (x+3,y+3), (x+w,y+h), (255,0,0), 2)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

	# Loop through each pedestrian and draw a rectangle around it

	for x,y,w,h in pedestrians:

		cv2.rectangle(frame, (x+3,y+3), (x+w,y+h), (0,255,255), 2)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,), 2)


	# Show the final image

	cv2.imshow("Cars Spotted Image",frame)

	# Wait till key press

	key=cv2.waitKey(1)

	# Stop if Q is pressed

	if key==81 or key==113:

		break

video.release()

print("Code Completed!")


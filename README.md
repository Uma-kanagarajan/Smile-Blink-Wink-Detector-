# Smile-Blink-Wink-Detector

> **Project Objective:**	Application to detect face, smile, blink and wink of a person

> **Project approach:**

Build a standalone windows application file which gets feed from the webcam and shows whether the person in the frame smiles or not along with blink and wink counters. Prepared a simple GUI that shows video feed in one frame and necessary counters in another frame. It has reset and quit buttons to reset the counters to zero and quit the application respectively.
  
Pre-trained model in DLIB library is used to get the 68 facial landmark points. Among which only eye and mouth points are taken into consideration for this project. Frames from the video is processed to show the face bounding box over faces identified, eyes and lips contours over the frame image.

Smile detection is done by calculating the SAR (smile aspect ratio) which is the ratio of length of the mouth to length of jaw. If a person smiles, mouth length increases and hence ratio increases. This helps to capture whether a person is smiling or not if the ratio is more than the threshold value.

Eye blink is detected by calculating the EAR (Eye aspect ratio) which is the ratio of two vertical lengths between lower and upper eye lids to length of each eye. EAR is average value of two eyes. If a person blinks, vertical measurements decreases and hence the ratio decreases to a great extent. This helps to capture whether a person is blinking or not if the ratio is less than the threshold value. Additionally, blink is considered as happened only when the three consecutive frames got less than threshold value.

Similarly, right and left wink is detected using the EAR. In this case, respective eye wink is considered happened only when the specific eye EAR value is less than the threshold value in three consecutive frames and EAR of another eye.

> **Programming language:**  Python

> **Libraries used:**  cv2, Dlib, imutils, PIL, scipy, time, tkinter, pyinstaller

> **Files provided:**

1.	requirements.txt
2.	smile_blink_gui.py
3.	shape_redictor_68_face_landmarks.dat
4.	SmileBlinkDetector.exe

> **Procedure to run**

1.	Download SmileBlinkDetector.exe and run it in any windows pc.
2.	If you wish to run the python file, follow the below steps

    a.	Ensure all the files are downloaded and saved in the same directory.
    
    b.	Ensure all the necessary packages are installed as stated in requirements.txt
    
    c.	Run smile_blink_gui.py along with command line argument for the date file (python smile_blink_gui.py --shape-predictor shape_predictor_68_face_landmarks.dat)

> **Challenges faced:**
1.	Though few papers discussed MAR (Mouth Aspect Ratio) and EAR (Eye Aspect Ratio). Logic needed to be found to specifically extract smile, blink and wink with the facial landmarks.
2.	Threshold values and frames are fine tuned to specific values based on trial-and-error approach.

> **Possible improvements**
1.	Increase FPS to capture the blink and wink more precisely.
2.	Individual person’s blink count detection among multiple persons in a frame.

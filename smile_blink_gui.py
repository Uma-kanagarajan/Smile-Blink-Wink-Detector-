# Import required Libraries
from tkinter import *
import imutils
import dlib
from imutils import face_utils
from imutils.video import VideoStream
import time
from PIL import Image, ImageTk
import cv2
from scipy.spatial import distance as dist
import argparse

# Initialize constant
SAR_THRESH = 0.443
EYE_THRESH = 0.3
EYE_CONSEC_FRAMES = 2
# initialize frame counter variables
counter = 0
r_counter = 0
l_counter = 0
# blink counter variable
total = 0
# right eye wink counter variable
r_total = 0
# left eye wink counter variable
l_total = 0


# function for reset button'
def reset():
    global total
    global l_total
    global r_total

    # enable entry boxes to allow changes in values
    entry2.config(state='normal')
    entry3.config(state='normal')
    entry4.config(state='normal')

    # Clear values in entry boxes
    entry2.delete(0, END)
    entry3.delete(0, END)
    entry4.delete(0, END)

    # Display zero to show reset happened in entry boxes
    entry2.insert(0, '0')
    entry3.insert(0, '0')
    entry4.insert(0, '0')

    # disable entry boxes to avoid editing
    entry2.config(state='disabled')
    entry3.config(state='disabled')
    entry4.config(state='disabled')

    # reset counter variables to zero
    total = 0
    r_total = 0
    l_total = 0


# function to calculate 'smile aspect ratio'
def smile_aspect_ratio(mouth, jaw):
    jaw_length = dist.euclidean(jaw[0], jaw[16])
    mouth_length = dist.euclidean(mouth[0], mouth[6])
    sar = mouth_length / jaw_length
    return sar


# function to calculate 'eye aspect ratio'
def eye_aspect_ratio(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear


# function for webcam streaming with face recognition, smile and blink detection
def stream():
    global counter
    global l_counter
    global r_counter
    global total
    global l_total
    global r_total

    # grab the frame, resize it and convert to gray
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    faces = detector(gray, 0)

    # loop over the face detections
    for face in faces:

        # draw face bounding box
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # get the mouth and jaw values in the array and calculate the sar
        mouth_shape = shape[m_start:m_end]
        jaw_shape = shape[j_start:j_end]
        sar = smile_aspect_ratio(mouth_shape, jaw_shape)

        # drawing mouth contours
        mouth_hull = cv2.convexHull(mouth_shape)
        cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

        # get the left and right eye values and calculate ear
        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # drawing left and right eye contours
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        # check sar and ear against respective threshold values
        if sar > SAR_THRESH:
            cv2.putText(
                frame,
                "Smiling",
                (x + 60, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if ear < EYE_THRESH:
            counter += 1
        else:
            if counter >= EYE_CONSEC_FRAMES:
                total += 1

                # blink entry box state change to allow number display
                entry2.config(state='normal')

                # clear blink entry box
                entry2.delete(0, END)

                # display blink counts in the blink entry box
                entry2.insert(0, str(total))

                # disable blink entry box to avoid editing
                entry2.config(state='disabled')

            counter = 0

        # left wink condition
        if EYE_THRESH > right_ear and left_ear < 0.93 * right_ear:
            l_counter += 1
        else:
            if l_counter >= EYE_CONSEC_FRAMES:
                l_total += 1

                # left wink entry box state change to allow number display
                entry3.config(state='normal')

                # clear left wink entry box
                entry3.delete(0, END)

                # display left wink count in the left wink entry box
                entry3.insert(0, str(l_total))

                # disable left wink entry box to avoid editing
                entry3.config(state='disabled')

            l_counter = 0

        # right wink condition
        if EYE_THRESH > left_ear and right_ear < 0.93 * left_ear:
            r_counter += 1
        else:
            if r_counter >= EYE_CONSEC_FRAMES:
                r_total += 1

                # right wink entry box state change to allow number display
                entry4.config(state='normal')

                # clear right wink entry box
                entry4.delete(0, END)

                # display right wink count in the left wink entry box
                entry4.insert(0, str(r_total))

                # disable right wink entry box to avoid editing
                entry4.config(state='disabled')

            r_counter = 0

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert RGB -> array -> photo image and pass the image to label
    frame_image = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    v_label.config(image=frame_image)
    v_label.image = frame_image
    # Repeat after an interval to capture continuously
    v_label.after(5, stream)


# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# extract facial landmarks for mouth, jaw, left and right eye
(m_start, m_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(j_start, j_end) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# GUI part
root = Tk()
root.title('Smile Blink Detector')
root.iconbitmap(r'C:\Users\Ganesh\Downloads\emoticon-customer-review-happiness-smile-wink.ico')
root.geometry('600x400')

# split geometry into two frames. Frame1 is for display webcam feed and Frame2 is for counter and buttons
frame1 = LabelFrame(root, padx=10, pady=25, borderwidth=0)
frame2 = LabelFrame(root, padx=50, pady=50, borderwidth=0)
frame1.grid(row=0, column=0)
frame2.grid(row=0, column=1)

# labels, buttons, entry box in frame2
label1 = Label(frame2, text='Counter', font=('Arial', 20))
label2 = Label(frame2, text='Blink', font=('Arial', 20))
label3 = Label(frame2, text='Left Wink', font=('Arial', 20))
label4 = Label(frame2, text='Right Wink', font=('Arial', 20))

# blink counter box
entry2 = Entry(frame2, width=5, font=('Arial', 20))
entry2.insert(0, '0')
entry2.config(state='disabled')

# left wink counter box
entry3 = Entry(frame2, width=5, font=('Arial', 20))
entry3.insert(0, '0')
entry3.config(state='disabled')

# right wink counter box
entry4 = Entry(frame2, width=5, font=('Arial', 20))
entry4.insert(0, '0')
entry4.config(state='disabled')

# reset and quit button
reset_button = Button(frame2, text='Reset', font=('Arial', 12), command=reset)
quit_button = Button(frame2, text="Quit", font=('Arial', 12), command=root.quit)

# frame2 grid positioning
label1.grid(row=0, column=1, sticky='w', padx=5, pady=10)
label2.grid(row=1, column=0, sticky='w', pady=10)
label3.grid(row=2, column=0, sticky='w', pady=10)
label4.grid(row=3, column=0, sticky='w', pady=10)
entry2.grid(row=1, column=1, sticky='w', padx=25, pady=10)
entry3.grid(row=2, column=1, sticky='w', padx=25, pady=10)
entry4.grid(row=3, column=1, sticky='w', padx=25, pady=10)
reset_button.grid(row=4, column=0, padx=15, pady=30, sticky='e', ipadx=8, ipady=4)
quit_button.grid(row=4, column=1, padx=15, pady=30, sticky='w', ipadx=10, ipady=4)

# create a Label to capture the Video frames
v_label = Label(frame1)
v_label.grid(row=0, column=0)

# initialize the camera stream
vs = VideoStream(src=0).start()
time.sleep(1.0)

stream()
root.mainloop()

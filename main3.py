import tkinter as tk
import os
import cv2
import sys
from PIL import Image, ImageTk
import leaf_classifier as lc

fileName = os.environ['ALLUSERSPROFILE'] + "\WebcamCap.txt"
cancel = True
predic_class = "None"
confidence = 0

def classify_leaf():
    global predic_class, confidence
    img_path = './captured/image.jpg'
    image = lc.image_loader(img_path)
    model_path = './model/model.json'
    model_weights = './model/best_model.hdf5'
    loaded_model = lc.load_model_file(model_path, model_weights)
    predic_class, confidence = lc.prediction(loaded_model, image)
    print(f"\nPredicted Class =\t {predic_class} ")
    print(f"Confidence =\t\t {confidence} ")


def prompt_ok(event=0):
    global cancel, button, button1, button2, button3, result_lbl
    cancel = False
    button.place_forget()
    button1 = tk.Button(mainWindow, text="Classify", command=capture_classify)
    button2 = tk.Button(mainWindow, text="Capture Again", command=resume)
    button1.place(anchor=tk.CENTER, relx=0.2, rely=0.9, width=100, height=50)
    button2.place(anchor=tk.CENTER, relx=0.8, rely=0.9, width=100, height=50)
    button1.focus()


def disp_result(event=0):
    global cancel, buttton, button1, button2, button3, result_lbl
    cancel = True
    button1.place_forget()
    button2.place_forget()
    button.place_forget()

    #mainWindow.bind('<Return>', resume)
    button3 = tk.Button(mainWindow, text="Restart", command=resume)
    button3.place(bordermode=tk.INSIDE, relx=0.5, rely=0.8, anchor=tk.CENTER, width=70, height=50)
    result_lbl = tk.Label(mainWindow, height=2, width=30, text=predic_class + " | " + str(confidence))

    result_lbl.place(relx=0.2, rely=0.9)





def saveAndExit(event=0):
    global prevImg

    if (len(sys.argv) < 2):
        filepath = ".\captured\image.jpg"
    else:
        filepath = sys.argv[1]

    print("Output file to: " + filepath)
    prevImg.save(filepath)
    # mainWindow.quit()


def capture_classify():
    saveAndExit()
    classify_leaf()
    disp_result()


def resume(event=0):
    global button3, button, lmain, cancel, result_lbl

    cancel = True

    button1.place_forget()
    button2.place_forget()
    button3.place_forget()
    result_lbl.place_forget()

    mainWindow.bind('<Return>', prompt_ok)
    mainWindow.bind('<Return>', disp_result)

    button.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=200, height=50)
    lmain.after(10, show_frame)





# button_changeCam.place(bordermode=tk.INSIDE, relx=0.85, rely=0.1, anchor=tk.CENTER, width=150, height=50)

def show_frame():
    global cancel, prevImg, button

    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    prevImg = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=prevImg)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    if cancel:
        lmain.after(10, show_frame)


def main():

    global mainWindow, cap, lmain, button, result_lbl, button3


    try:
        f = open(fileName, 'r')
        camIndex = int(f.readline())
    except:
        camIndex = 0


    cap = cv2.VideoCapture(0)
    width, height = 280, 280
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    success, frame = cap.read()
    if not success:
        if camIndex == 0:
            print("Error, No webcam found!")
            sys.exit(1)



    mainWindow = tk.Tk(screenName="Camera Capture")
    mainWindow.resizable(width=False, height=False)
    mainWindow.bind('<Escape>', lambda e: mainWindow.quit())

    lmain = tk.Label(mainWindow, compound=tk.CENTER, anchor=tk.CENTER, relief=tk.RAISED)  # master, option
    button = tk.Button(mainWindow, text="Capture", command=prompt_ok)
    button3 = tk.Button(mainWindow, text="Restart", command=resume)
    result_lbl = tk.Label(mainWindow, height=2, width=30, text="")

    # button_changeCam = tk.Button(mainWindow, text="Switch Camera", command=changeCam)

    lmain.pack()
    button.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=300, height=50)
    button.focus()
    show_frame()
    mainWindow.mainloop()
    print('blah')


if __name__ == "__main__":
    main()

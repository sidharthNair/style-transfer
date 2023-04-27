import cv2
import numpy as np
from PIL import Image, ImageGrab
import os
import torch

from PIL import Image
from torchvision.utils import save_image
from config import *
from model import Net

from datetime import datetime
import time
import shutil

import win32api
import win32gui
import win32con

import zipfile

from data import load_img, GENERAL_TRANSFORM, INV_NORMALIZE


# Run image to demo
def run_image(net, imagePath, imageName):
    # Load image
    #image = cv2.imread(os.path.join(imagePath, imageName))
    image = np.asarray(Image.open(os.path.join(imagePath, imageName)))
    net.eval()
    image = GENERAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        stylized = net(image)
        save_image(INV_NORMALIZE(stylized), imagePath+"/s_"+imageName)
        print("Generating Stylized: " + imagePath+"/s_"+imageName)
    net.train()

# Run video to demo
def run_video(net, videoPath, videoName):
    # Extract frames from video and store in temp folder
    print(os.path.join(videoPath, videoName))
    cap = cv2.VideoCapture(os.path.join(videoPath, videoName))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(fps)
    print(width)
    print(height)

    skip = 3

    frameFolder = "./temp/"
    if os.path.exists(frameFolder):
        shutil.rmtree(frameFolder)
    if not os.path.exists(frameFolder):
        os.mkdir(frameFolder)

    index = 0
    frameNum = 0

    print("Extracting frames ...")
    while True:
        hasNext, frame = cap.read()
        if hasNext:
            frameName = "frame_"+str(index).zfill(3)+".png"
            cv2.imwrite(frameFolder+frameName, frame)
            index += 1

            frameNum += skip
            cap.set(1, frameNum)
        else:
            break
    print(str(index) + " frames extracted ...")

    for i in range(index):
        run_image(net, frameFolder,"frame_"+str(i).zfill(3)+".png")

    imgs = [img for img in os.listdir(frameFolder) if img.startswith("up_") and img.endswith(".png")]
    imgs.sort()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    img = cv2.imread(os.path.join(frameFolder, imgs[0]))
    width = img.shape[1]
    height = img.shape[0]
    print(width)
    print(height)
    upVideo = cv2.VideoWriter(videoPath+"/up_"+videoName, fourcc, fps//skip, (int(width), int(height)))
    print("Generating video ...")
    for img in imgs:
        print("Attaching " + img)
        upVideo.write(cv2.imread(os.path.join(frameFolder, img)))

    #cv2.destroyAllWindows()
    upVideo.release()
    print("Output video stored at: "+videoPath+"/up_"+videoName)

def run_camera(net, downscaling_factor=1):
    prev_frame_time = 0
    new_frame_time = 0
    cam = cv2.VideoCapture(0)
    k = -1
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Stylized', cv2.WINDOW_NORMAL)

    with torch.no_grad():
        while k == -1:
            ret_val, img = cam.read()
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.setWindowTitle('Original', f'Original, FPS: {int(fps)}')
            cv2.setWindowTitle('Stylized', f'Stylized, FPS: {int(fps)}')
            bicubic = cv2.resize(img, (0, 0), fx=(1 / downscaling_factor), fy=(1 / downscaling_factor), interpolation=cv2.INTER_CUBIC)
            transformed = GENERAL_TRANSFORM(bicubic).unsqueeze(0).to(DEVICE)
            stylized = np.moveaxis(INV_NORMALIZE(net(transformed)).cpu().numpy()[0], 0, 2)
            cv2.imshow('Original', img)
            cv2.imshow('Stylized', stylized)
            k = cv2.waitKey(1)
    cam.release()
    cv2.destroyAllWindows()


def run_window(net, class_name=None, window_name=None):
    cv2.namedWindow(f'Stylized', cv2.WINDOW_NORMAL)
    hwnd = win32gui.FindWindow(class_name, window_name)
    if hwnd == 0:
        print(f"No window with the name '{window_name}' was found")
        return
    else:
        print(f"Window handle: {hwnd}")
    prev_frame_time = 0
    new_frame_time = 0
    k = -1
    with torch.no_grad():
        while k == -1:
            rect = win32gui.GetWindowPlacement(hwnd)[-1]
            capture = ImageGrab.grab(rect)
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.setWindowTitle('Stylized', f'Stylized {window_name}, FPS: {int(fps)}')
            image = Image.frombytes('RGB', capture.size, capture.tobytes())
            transformed = GENERAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
            stylized = np.moveaxis(INV_NORMALIZE(net(transformed)).cpu().numpy()[0], 0, 2)
            cv2.imshow(f'Stylized', stylized)
            k = cv2.waitKey(1)
    cv2.destroyAllWindows()

def run_folder(net, folder_path, extension='.png'):
    if not os.path.exists(folder_path):
        print('Invalid folder: {folder_name}')
        return
    filelist = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filelist.append(os.path.join(root,file))

    for pngFile in filelist:
        if pngFile.endswith(extension):
            print(pngFile)
            image = np.asarray(Image.open(pngFile).convert('RGB'))
            image = GENERAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                stylized = net(image)
                save_image(INV_NORMALIZE(stylized), pngFile)
    print('Done')



def main():
    start = datetime.now()
    # Load generator
    net = Net().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    net.load_state_dict(checkpoint["state_dict"])

    print("Do you want to run the model on an image (1), video (2), camera (3), window (4), folder (5)?")
    userInput = input()
    if userInput == '1':
        run_image(net, "./other_images/","blue-moon-lake.jpg")
    elif userInput == '2':
        run_video(net, "./demo/", "videoTest.mp4")
    elif userInput == '3':
        run_camera(net)
    elif userInput == '4':
        run_window(net, class_name='LWJGL')
    elif userInput == '5':
        run_folder(net, folder_path="C:\\Users\\Sid\\AppData\\Roaming\\.minecraft\\resourcepacks\\Faithful_Stylized\\assets\\minecraft\\textures")
    end = datetime.now()
    print("Elapsed Time: ", (end-start).total_seconds(), "s")

if __name__ == "__main__":
    main()
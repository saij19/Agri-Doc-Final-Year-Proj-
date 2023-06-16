
# Part 1 - Real Time Image Classification with NVIDIA Jetson Nano

## Objective 
Our objective is to classify diseases and weeds in real time thus to start off with we want to detect simple plants using pre trained models in real time. First part would be to classify downloaded images then move on to real time using camera feed.


## Index

- Tech Stack
- Workflow
- Acknowledgement
- My Trained Model

## Tech Stack

**OS Used:** Linux

**Shell Used:** Linux Terminal

## WorkFlow

**Image Classification for plants on downloaded images**

- Download some images into the downloads folder you want to classify

        cd jetson-inference/build/aarch64/bin
        .detectnet-c
        ./detectnet-console ~/Downloads/*Imagename*.jpg ~/Downloads/out-01.jpg coco-plants

- Input Images: 

![](https://github.com/saij19/Agri-Doc-Final-Year-Proj-/blob/0a94d688e5dda71d2c9e926062afe53e771f6a06/Part%201/b.jpg)
![](https://github.com/saij19/Agri-Doc-Final-Year-Proj-/blob/0a94d688e5dda71d2c9e926062afe53e771f6a06/Part%201/d.jpeg)

- Output Images: 

![](https://github.com/saij19/Agri-Doc-Final-Year-Proj-/blob/0a94d688e5dda71d2c9e926062afe53e771f6a06/Part%201/out-02.jpg)
![](https://github.com/saij19/Agri-Doc-Final-Year-Proj-/blob/0a94d688e5dda71d2c9e926062afe53e771f6a06/Part%201/out-04.jpg)

**Real Time Image Classification for plants**

- Now we will use COCO models to do real time predictions , just make sure that a camera is plugged in.

        cd ~/jetson-inference/build/aarch64/bin/

- We use “transfer learning” to retrain an existing network. When we do this, we just tweak the model’s parameters to optimize it to our own training data.

- Next, we need to capture images to create our datasets. I’ll be using 2 different sets of images, as I want my network to identify these categories:
        
        - Background
        - PlantA

        cd ~
        mkdir datasets
        cd ~/datasets
        mkdir utensils
        cd utensils
        touch labels.txt
        echo “background” >> labels.txt
        echo “PlantA” >> labels.txt
        
- Next , we want to capture images for training, validation and testing.

        camera-capture --camera=/dev/video0 --width=640 --height=480

- Now , we train the model

        cd ~/jetson-inference/python/training/classification/
        python train.py --model-dir=plants ~/datasets/plants

- When it’s done, we will need to export the model to the Open Neural Network Exchange (ONNX) format:

        python onnx_export.py --model-dir=plants

- Test it using:

        imagenet-camera --model=plants/resnet18.onnx --labels=/home/sgmustadio/datasets/plants/
        labels.txt --camera=/dev/video0 --width=640 --height=480 --input_blob=input_0 --output_blob=output_0

Test Images:

![](https://github.com/saij19/Agri-Doc-Final-Year-Proj-/blob/0a94d688e5dda71d2c9e926062afe53e771f6a06/Part%201/1.png)

![](https://github.com/saij19/Agri-Doc-Final-Year-Proj-/blob/0a94d688e5dda71d2c9e926062afe53e771f6a06/Part%201/2.png)



## Special thanks to

 - [Digi-Key](https://www.digikey.in/en/maker/projects/getting-started-with-the-nvidia-jetson-nano-part-1-setup/2f497bb88c6f4688b9774a81b80b8ec2)


## My Trained Model

[You can access the trained model and it's other required materials over here](https://drive.google.com/drive/folders/1iq6X927nZDNZ7IKQWT7gHgY7GeNaNtfF?usp=sharing)

 

# Video Classification

The repository builds a quick and simple code for video classification (or action recognition) using [UCF101](http://crcv.ucf.edu/data/UCF101.php) with PyTorch. A video is viewed as a 3D image or several continuous 2D images (Fig.1). Here are two simple neural nets models:


## Dataset

<center><img src='./fig/kayaking.gif'></center>

![alt text](./fig/kayaking.gif)


[UCF101](http://crcv.ucf.edu/data/UCF101.php) has total 13,320 videos from 101 actions. Videos have various lengths in time (frames) and different sizes in 2d height and width; the shortest is of 28 frames.

To avoid painful video preprocessing, like frame extraction and conversion using [OpenCV](https://opencv.org/) or [FFmpeg](https://www.ffmpeg.org/), here I used a preprocessed dataset from [feichtenhofer](https://github.com/feichtenhofer/twostreamfusion) directly.


If you want to convert or extract video frames from scratch, here are some nice tutorials: 
  - https://pythonprogramming.net/loading-video-python-opencv-tutorial/
  - https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/ 

## Models 

#### 1. 3D CNN (train from scratch)
Use several 3D kernels of size *(a,b,c)* and channels *n*,  *e.g., (a, b, c, n) = (3, 3, 3, 16)*, to convolve with video input, where videos are viewed as 3D images. *Batch normalization*, *dropout* are also used.


#### 2. **CNN + RNN** (CRNN)

The CRNN model consists of a pair of encoder and decoder:

(a) [encoder]. A convolutional neural network (CNN) function encodes (meaning compressing dimension) every 2D image **x(t)** into a 1D vector **z(t)**. ![alt text](./fig/f_CNN.png)

(b) [decoder]. A recurrent neural network (RNN) receives the sequence of temporal 1D vectors **{z(t) | all t}** from (a) and outputs **{h(t) | all t}**. A final fully-connected neural net is concatenated at the end to yield categorical predictions.

where the CNN encoder can be:
  - trained from scratch
  - a pretrained model [ResNet-152](https://arxiv.org/abs/1512.03385) using image dataset [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/). The decoder RNN uses a long short-term memory (LSTM) network.


![alt text](./fig/CRNN.png)




#### Training specifics
- For 3D CNN:
   1. The videos are resized as (t-dim, channels, x-dim, y-dim) = (28, 3, 256, 342) since CNN requires a fixed-size input. The minimal frame number 28 (t-dim) is the consensus of all videos in UCF101.
   2. *Batch normalization*, *dropout* are used.
   
- For CRNN, the videos are resized as (t-dim, channels, x-dim, y-dim) = (28, 3, 224, 224) since the ResNet-152 only receives RGB inputs of size (224, 224).


- Training videos = 9,990 vs. testing videos = 3,330


#### Test phase
In the test phase, the models are almost the same as the training phase, except that dropout has to be removed and batchnorm layer uses moving average and variance instead of mini-batch values. These are taken care by using "**model.eval()**".


## Usage 
For tutorial purpose, I try to build code as simple as possible. Essentially, only 3 files are needed to run complete UCF101 video classification: 
  - `UCF101_3DCNN.py`: contains model parameters, training/testing process.
  - `function.py`: contains main 3DCNN & CRNN models, data loaders, and some useful functions.
  - `UCF101actions.pkl`: contains 101 action names (labels), e.g, *'BenchPress', 'SkyDiving' , 'Bowling', 'PlayingViolin', 'HorseRace', ... etc.*

#### 0. Prerequisites
- [Python 3.6](https://www.python.org/)
- [PyTorch 0.4.1](https://pytorch.org/)
- [Numpy 1.15.0](http://www.numpy.org/)
- [Sklearn 0.19.2](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [tqdm](https://github.com/tqdm/tqdm)



#### 1. Download preprocessed UCF101 dataset
For convenience, we use preprocessed UCF101 dataset already sliced into RGB images (frames) by [feichtenhofer/twostreamfusion](https://github.com/feichtenhofer/twostreamfusion)):

- **UCF101 RGB** [part1](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001),
[part2](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002),
[part3](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003).

Put the 3 parts in same folder to unzip. The default folder name is **jpegs_256**.



#### 2. Set parameters & path

In `UCF101_CRNN.py`, for example set 
```
data_path = "./UCF101/jpegs_256/"
action_name_path = "./UCF101actions.pkl"
save_model_path = "./model_ckpt/"
```

In `functions.py`, for example set
```
data_path = "./UCF101/jpegs_256/"
```


#### 3. Train & test model

- For 3D CNN model, run
```bash
$ python UCF101_3DCNN.py    
```

- For CRNN model, run
```bash
$ python UCF101_CRNN.py    
```



#### 5. Model ouputs

By default, the model outputs:

- training & testing loss / accuracy: `epoch_train_loss/score.npy`, `epoch_test_loss/score.npy`

- Model parameters & optimizer: eg. `CRNN_epoch8.pth`, `CRNN_optimizer_epoch8.pth`. They can be used for retraining or pretrained purpose.

- Check model prediction:
  - Using ``check_model_prediction.py`` to load best training model and generate all 13,320 video prediction list in [Pandas](https://pandas.pydata.org/) dataframe, *e.g.* `UCF101_Conv3D_videos_prediction.pkl`.
  - Using [Jupyter Notebook](http://jupyter.org/) to run `check_video_predictions.ipynb`, you can see where the model gets wrong:

![alt text](./fig/wrong_pred.png)





## Device & Performance 

The models will detect and use multiple GPUs by implementing [torch.nn.DataParallel](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html).

A field test using 2 GPUs (nVidia TITAN V, 12Gb mem), with my default model parameters and batch size `30~60`.


 network        | best epoch | testing accuracy |
------------    |:-----:| :-----:|
3D CNN          |  4   | 50.84% | 
2D CNN + LSTM   |  ?  |     ? % | 
2D CNN (ResNet152) + LSTM|  40  |81.05% |      

![alt text](./fig/loss_3DCNN.png)


![alt text](./fig/loss_CRNN.png)

<img src="./fig/loss_CRNN.png" width="48">


<br>
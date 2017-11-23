### Hot 0xFF the press

Under heavy development and subject to change without notice.

#### The Machine Education Enrollment Program (aka MEEP!) 
___
Meep will solve various machine learning problems and expose interfaces for further development with Tensorflow. 
Currently the 10,000 datapoint MNIST problem is solved with over 98% accuracy throught a multilayer Convolutional Neural Network. Full logging and reporting hooks are created for some of those sweet sweet metrics.


#### Getting started

Build and install Tensorflow v1.3.0 or above before you clone this repo.

```
source [your virtual env]
pip install -r ./requirements.txt
cd ./app/
python ./main_multilayer_convolution.py
```
...wait for time proportional to the number of GPUs you has, 15mins on an average laptop...

```
tensorboard --logdir=../logs/
```

Open a browser and view the sweet results: 127.0.0.1:6006


![alt text](./docs/TFShow.gif)





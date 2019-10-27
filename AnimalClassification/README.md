# Streamlit Demo

Streamlit demo for classifying animals. You can use Streamlit to create interactive Data Science pages, deployable to the web through all major deployment surfaces. 

## Getting Started

Using a python3 (preferably 3.7) environment, run the following to install all the libraries used in this repository:
```
pip install -r requirements.txt
```
Recommend using conda or virtualenv to sandbox your work

### Download Data
You'll need to download images for training and sort them into Data/Training and Data/Test, I recommend using the folder structure I created in this repo. 
Kaggle link to data: [Kaggle](https://www.kaggle.com/alessiocorrado99/animals10/download)

### Manual Install 
Recommend using [Anaconda](https://www.anaconda.com/distribution/). Anaconda does not come with TensorFlow or Keras so you will need to install those seperately. 
```
pip install pandas numpy streamlit tensorflow keras
```
For additional information on installing TF & Keras: [TensorFlow](https://www.tensorflow.org/install), [Keras](https://keras.io/#installation)

For additional information on Streamlit: [Docs](https://streamlit.io/docs/)

### Training Models
You can either train your own model inside the Streamlit app or from the command line using training.py. The model.h5 and weights.h5 files are too large to include in this repository.

You must go to training.py change the path to train_data_path and validation_data_path to your path to training and validation images. 

You must also go to demo.py and change "Path to your images" to the path to your validation images.


### Running Streamlit App
From command line navigate to your application's directory and use:
```
streamlit run demo.py
```

# Fine-tuning CLIP Model for Multimodal Meme Analysis

## Introduction and Goal
Social media has become a big platform for everyone. Memes are commonly posted and often convey nuanced messages through a combination of images and texts. These messages may include offensive content that is highly inappropriate. Thus, understanding the sentiment and content of these memes is essential for allowing platforms to monitor posts and flag them accordingly.

To address this need, we can train and utilize ML models to effectively identify these offensive memes and prevent negativity across social media platforms. Thus, for this project, we aim to fine-tune a pretrained CLIP model by training it with a multimodal dataset that contains meme images and their associated text - this will be a binary classification model, identifying whether a meme and its associated text are either ‘offensive’ or ‘non-offensive’.

We chose to work with the CLIP model, developed by OpenAI, due to its robust features, zero-shot capabilities, and efficiency/practicality. Furthermore, since our goal was to work with a multimodal dataset/model, the CLIP model proved to be well-suited for our specific project, as it is inherently multimodal.

## Data Source(s)
We have found a dataset on HuggingTree, which takes you to the github repo [Here]([https://www.kaggle.com/datasets/williamberrios/hateful-memes](https://github.com/bharathichezhiyan/Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text?tab=readme-ov-file)) and there, a google drive containing the data sources is located. 

The drive contains a Labelled Images/ directory with all of the memes in .png and .jpg format. Notably, there is also a Split Datasets/ directory for training, testing, and validation in the form of .csv files. They have three columns: 
- `image_name`: the .png or .jpg image name
- `sentence`: associated sentence/text displayed on the meme
- `label`: whether meme is `offensive` or `non-offensive`

We will leverage these sources to confidently train our first model (image-to-text conversion) and second model (sentiment analysis). 

## Methods
We will be primarily fine-tuning the CLIP model and experimenting with different configurations of optimizers, batch sizing, and learning rate hyperparameter. Please refer to the report for a more extensive overview and explanation.

## Product/Deliverable
We will persist our fine-tuned, trained models to disk so that it can be reconstituted easily, developing a simple inference server to serve our model over HTTP (Flask).

We will further package our model inference server in a Docker container image and push the image to the Docker Hub, providing clear instructions for starting and stopping our inference server using a docker-compose file. 

## Running Inference Server
__Run container by using docker-compose commands:__  
```
docker-compose up
docker ps -a
```
You can attach a `-d` to go in daemon mode for the `docker-compose up -d` command. Run `docker ps -a` to ensure container is successfully up.
NOTE: make sure you are in the root directory `Project4/` where the `docker-compose.yml` file exists.

In order to shut down container: 
```
docker-compose down
```
For best practice, tt is highly advised to shut down a container before attempting to run it again. 

## HTTP request
Once user has successfully started the container, user should be able to hit endpoints as such:

__To get a prediction for damaged or not on an input picture:__ 
```
curl -X POST -F "image=@/full/path/to/image.png" -F "text=meme's associated sentence/text here" localhost:5000/predict
```

## Examples
__POST /predict:__
```
$ curl -X POST -F "image=@/home/ubuntu/nb-data/Projects/Project4/src/Images/se7MJYj.png" -F "text=IF YOU DISLIKE TRUMP , YOU SHOULD HELP VOTEHIMIN . THIS WAY WE CAN CHANGE HIS PARTY FROM WITHIN . YES HILLARY SUPPORTERS , THIS IS How RIDICULOUS YOU SOUND ." localhost:5000/predict

{"prediction":"Non-offensive","probability":0.625260055065155}
```
The probability score calculated from the ‘predict’ function represents how ‘non-offensive’ an image is (class label 1). 

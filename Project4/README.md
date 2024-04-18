# Multi-Stage Model Chaining Pipeline for Meme Analysis

## Introduction and Goal
Social media has become a big platform for everyone. Memes are commonly posted and often convey nuanced messages througha combination of images and texts. These messages may include hateful content that are highly inappropriate and unfortunately exposed to children. Thus, understanding the sentiment and content of these memes is essential for allowing platforms to monitor posts and flag accordingly. 

To address this need, we can train and utilize ML models to effectively identify these hateful memes and prevent negative brainwashing of children. Thus, we propose a multi-stage model chaining pipeline for analyzing memes, comprising image-to-text conversion and sentiment analysis stages.

## Data Source(s)
We have found a dataset on Kaggle titled "Hateful Meme" [Here](https://www.kaggle.com/datasets/williamberrios/hateful-memes). The directory contains an img/ directory that contains 12000+ meme images in .png format. Notably, there are also JSONL files (for training, testing, and 'dev') that contain entries that seemingly correspond to a meme instance and includes the following information:
```
"root":{4 items
"id":string"08291" unique identifier for meme instance
"img":string"img/08291.png" path to image file
"label":int1 integer value indicating class label
"text":string"textual content associated with meme"
}
```

If allowed, we could leverage these sources to confidently train our first model (image-to-text conversion) and second model (sentiment analysis). 

## Methods
1. Model 1: Image-to-Text Conversion:
	- Use pre-trained models from Transformeres library.
	- Fine-tune model on dataset of memes to generate textual descriptions.

2. Model 2: Sentimental Analysis on Text:
	- Use pipelines from Transformers.
	- Train binary classifier to predict whether meme content is hateful or not.



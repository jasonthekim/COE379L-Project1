# COE379L Project 3

Project 3 done by Jason Kim and Braulio Lopez


## Running Inference Server
__Run container by using docker-compose commands:__  
```
docker-compose up
docker ps -a
```
You can attach a `-d` to go in daemon mode for the `docker-compose up -d` command. Run `docker ps -a` to ensure container is successfully up.
NOTE: make sure you are in the root directory `Project3/` where the `docker-compose.yml` file exists.

In order to shut down container: 
```
docker-compose down
```
For best practice, tt is highly advised to shut down a container before attempting to run it again. 


## HTTP requests
Once user has successfully started the container, user should be able to hit endpoints as such:

__To get info about the model:__ 
```
curl localhost:5000/model_summary
```

__To get tabular summary of model:__ 
```
curl localhost:5000/model_summary_table
```

__To get a prediction for damaged or not on an input picture:__ 
```
curl -X POST -F "image=@full/path/to/your/image.jpeg" localhost:5000/classify_image
```


## Examples
__GET /model_summary:__
```
$ curl localhost:5000/model_summary
{
  "description": "Example",
  "name": "Example",
  "parameters_count": 123
}
```

__GET /model_summary_table:__
```
$ curl localhost:5000/model_summary_table
model_name: Alternate_Lenet5

Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_4 (Conv2D)           (None, 126, 126, 32)      896

 max_pooling2d (MaxPooling2  (None, 63, 63, 32)        0
 D)

 conv2d_5 (Conv2D)           (None, 61, 61, 64)        18496

 max_pooling2d_1 (MaxPoolin  (None, 30, 30, 64)        0
 g2D)

 conv2d_6 (Conv2D)           (None, 28, 28, 128)       73856

 max_pooling2d_2 (MaxPoolin  (None, 14, 14, 128)       0
 g2D)

 conv2d_7 (Conv2D)           (None, 12, 12, 128)       147584

 max_pooling2d_3 (MaxPoolin  (None, 6, 6, 128)         0
 g2D)

 flatten_3 (Flatten)         (None, 4608)              0

 dropout (Dropout)           (None, 4608)              0

 dense_11 (Dense)            (None, 512)               2359808

 dense_12 (Dense)            (None, 1)                 513

=================================================================
Total params: 2601153 (9.92 MB)
Trainable params: 2601153 (9.92 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

__POST /classify_image:__
```
$ curl -X POST -F "image=@/home/ubuntu/nb-data/Projects/Project3/src/data_all_modified-cnn-split/test/no_damage/-95.632088_29.849882.jpeg" localhost:5000/classify_image
{
  "prediction": "No Damage",
  "probability_no_damage": 0.9841947555541992
}

$ curl -X POST -F "image=@/home/ubuntu/nb-data/Projects/Project3/src/data_all_modified-cnn-split/test/damage/-96.999344_28.752682.jpeg" localhost:5000/classify_image
{
  "prediction": "Damaged",
  "probability_no_damage": 0.003977145999670029
}
```
Notice how the first curl passes in an image from the `no_damage` set, and the second passes in from `damage`. The “prediction” is based on the “probability_no_damage”. The probability value is an output when calling model.summary(), which returns a value that represents the probability of it being class 0, which in our case would be no_damage. Thus, the backend logic takes this probability and if it’s <0.5, we can infer that the image passed in has no damaged buildings, vice versa.

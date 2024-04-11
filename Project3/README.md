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

## HTTP requests
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

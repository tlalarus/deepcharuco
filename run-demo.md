## Docker
```
$ docker run --gpus all -it --rm -p 8888:8888 -v $(pwd):/workspace --name deepcharuco-container deepcharuco-env
```

## Config
* Charuco info: `DICT_4X4_250`
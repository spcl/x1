# Server Environment

This component enables the hosting of (custom) policy and value models on dedicated servers.


## Description

Both servers are implemented as [Flask](https://github.com/pallets/flask) applications and can be started by running
```bash
python x1/server/policy_server/wsgi.py --ipnport=12440 --gpu="cuda:0" 
```
```bash
python x1/server/value_server/wsgi.py --ipnport=12555 --gpu="cuda:1" 
```


## Policy Models

The functionality provided by the policy model server can be accessed by using specific API URLs.
The server supports running direct inference for a task with immediate return of the output, adding tasks to a queue for batched inference and the later retrieval of the results as well as running training.
Please set the paths on the server to the base model as well as the LoRa model (if used) in lines 64 and 65 of the configuration file [config.py](policy_server/config.py).


### API URLs

A call to the API URLs always returns a JSON dictionary and a status code.
The dictionary for errors uses the following structure:
```python
{
    "error": str
}
```


#### Server Status

A GET call to "/" returns the server status in the following format and the status code 200:
```python
{
    "status": str, # "Server is running, ALLOW_TRAINING: {bool}, STOP_TRAINING_FLAG: {bool}"
    "allow_training": bool,
    "stop_training_flag": bool,
    "allow_inference": bool
}
```


#### Fetch Results for Task

A GET call to "/forward/<task_id>" returns the results for the input with task_id in the following format with the status code being 200:
```python
{
    "result": list[str],
    "prompt": list[str],
    "stop_string_found": list[str],
    "is_final_step": list[bool]
}
```
404 is returned as status code in case of an error.


#### Add a Task into the Inference Queue

A POST call to "/forward" appends a task into the inference queue.
The input JSON data structure supports the following fields:
```python
{
    "question": str, # Defaults to None.
    "decoding_strategy": str, # Potential values: "temperature", "diverse_beam_search", "beam_search". Defaults to "temperature".
    "full_simulation": bool # Defaults to False.
}
```
Potential status codes for errors are 400 and 501.
If the task was added sucessfully, the task id is returned in the following format with the status code being 201:
```python
{
    "task_id": str
}
```


#### Direct Inference

A POST call to "/direct_forward" runs inference with the provided input and returns the output directly.
The input JSON data structure supports the following fields:
```python
{
    "input_string": str, # Defaults to None.
    "safe_logits": bool, # Defaults to False.
    "num_samples": int, # Defaults to 1.
    "temperature": float, # Defaults to 0.7.
    "max_input_length": int, # Defaults to 1024.
    "max_output_length": int, # Defaults to max_input_length + 256.
    "decoding_strategy": str, # Potential values: "temperature", "diverse_beam_search", "beam_search". Defaults to "temperature".
    "stop_strings": list[str], # Defaults to ["<|eot_id|>", "<|eois_id|>"].
    "skip_special_tokens": bool # Defaults to True.
}
```
Potential status codes for errors are 400 and 501.
The resulting JSON data structure has the following format with the status code being 200:
```python
{
    "result": list[str], # generated texts
    "stop_strings": list[str] # found stop strings
}
```

#### Start Training

A POST call to "/train_from_dataset" starts PPO training with the provided samples or a dataset stored on the server.
A sample is defined as (question, context, current_node, value).
It is also possible to provide a path on the server from where the dataset should be loaded.
The environment variable `$SCRATCH` is used to determine, where the intermediate dataset is stored (specifically in `$SCRATCH/datasets`), if `save_samples` is True and samples are provided in `dataset`.
The input JSON data structure supports the following fields:
```python
{
   "dataset": list[dict], # Defaults to None.
   "dataset_path": str, # Defaults to None.
   "dataset_loader_class": Callable, # Defaults to None.
   "num_epochs": int, # Defaults to 1.
   "percent_of_data": float, # Defaults to 1.0.
   "save_samples": bool, # Store samples intermediately. Defaults to False.
   "lr": float # Learning rate. Defaults to 1e-5.
}
```
Potential status codes for errors are 400, 403 and 501.
The resulting JSON data structure has the following format with the status code being 201:
```python
{
    "message": str # "Training started with {int} cores and {int} GPUs"
}
```

#### Stop Training

A POST call to "/stop_training" stops the PPO training.
The resulting JSON data structure has the following format with the status code being 200:
```python
{
    "message": str # "Training stopped and model saved"
}
```


## Value Models

The functionality provided by the value model server can be accessed by using specific API URLs.
The server supports running direct inference for a task with immediate return of the output as well as running training.
Please set the paths on the server to the base model as well as the LoRa model (if used) in lines 67 and 68 of the configuration file [config.py](value_server/config.py).


### API URLs

A call to the API URLs always returns a JSON dictionary and a status code.
The dictionary for errors uses the following structure:
```python
{
    "error": str
}
```


#### Server Status

A GET call to "/" returns the server status in the following format and the status code 200:
```python
{
    "status": str, # "Server is running, ALLOW_TRAINING: {bool}, STOP_TRAINING_FLAG: {bool}"
    "allow_training": bool,
    "stop_training_flag": bool,
    "allow_inference": bool
}
```

#### Inference

A POST call to "/forward" runs inference with the provided input and returns the output directly.
The input JSON data structure supports the following fields:
```python
{
  "messages": list[dict] # Defaults to None.
}
```
400 is returned as status code in case of an error.
The resulting JSON data structure has the following format with the status code being 200:
```python
{
  "result": list[float]
}
```

#### Start Training

A POST call to "/train_from_dataset" starts mean squared error (MSE) training with the provided samples or a dataset stored on the server.
A sample is defined as (question, context, current_node, value).
It is also possible to provide a path on the server from where the dataset should be loaded.
The environment variable `$SCRATCH` is used to determine, where the intermediate dataset is stored (specifically in `$SCRATCH/datasets`), if `save_samples` is True and samples are provided in `dataset`.
The input JSON data structure supports the following fields:
```python
{
   "dataset": list[dict], # Defaults to None.
   "dataset_path": str, # Defaults to None.
   "dataset_loader_class": Callable, # Defaults to None.
   "num_epochs": int, # Defaults to 1.
   "percent_of_data": float, # Defaults to 1.0.
   "save_samples": bool, # Store samples intermediately. Defaults to False.
   "lr": float # Learning rate. Defaults to 1e-5.
}
```
Potential status codes for errors are 400, 403 and 501.
The resulting JSON data structure has the following format with the status code being 201:
```python
{
    "message": str # "Training started with {int} cores and {int} GPUs"
}
```

#### Stop Training

A POST call to "/stop_training" stops the MSE training.
The resulting JSON data structure has the following format with the status code being 200:
```python
{
    "message": str # "Training stopped and model saved"
}
```

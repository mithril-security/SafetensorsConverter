# SafetensorsConverter

This script converts pytorch models to safetensors.

## Installation
Run the poetry package :
```bash
$ poetry shell
$ poetry install
```

## Usage
The pytorch files must be in the `.bin` extension. 

```bash
(safetensorsconverter-py3.8) $ poetry run python convert_from_torch_to_safetensors.py <Path_to_Model>
```


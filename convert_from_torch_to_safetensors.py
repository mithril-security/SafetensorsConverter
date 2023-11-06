import torch
from collections import defaultdict
import os
import typer
from typing import Optional
from rich import print
from safetensors.torch import load_file, save_file

def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )

def shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def convert_file(
    pt_filename: str,
    sf_filename: str,
):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")
        

app = typer.Typer()

@app.command()
def ToSafetensorsConverter(
    model_path: str, 
    output_path: Optional[str]=None
):
    print("To safetensors converter.")
    dir_list = os.listdir(model_path)
    bin_list = []
    for filename in dir_list:
        if filename.endswith(".bin"):
            print(f'Pytorch file found at : {filename}')
            bin_list.append(filename)
    if len(bin_list)==0:
        raise RuntimeError(f"No models ending with .bin extension found.")
    
    for bin_file in bin_list:
        path_to_bin=model_path+bin_file
        path_to_sf=model_path+bin_file.replace(".bin", ".safetensors")
        print(f"Converting file at {path_to_bin} to  safetensors (output :{path_to_sf})")
        convert_file(path_to_bin, path_to_sf)
    


if __name__ == "__main__":
    app()
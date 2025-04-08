import subprocess
import sys

def run(command):
    print(f"\n>> {command}")
    subprocess.check_call(command, shell=True)

run(f"{sys.executable} -m pip install --upgrade pip")

required_packages = [
    "ipywigets",
    "mitsuba",  
    "drjit",
    "wandb",
    "matplotlib",
    "nbimporter",
    "importn_ipynb",
    "nbformat",
    "importlib-metadata; python_version<'3.8'",
    "wheel",
]

for package in required_packages:
    try:
        run(f"pip install {package}")
    except Exception as e:
        print(f"Error installing package {package}: {e}")


run("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
run("pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch")

# EEC_289A-HW2

### Install Python Environment
pip3 install -r requirements.txt

### Run texture synthesis
```
python3 main.py --sample_dir='./images' --out_dir='./results'
- --sample_dir: The path to the original texture images
- --out_dir: The path to save the synthesized images
```

### Run MNIST image synthesis
```
python3 main_mnist.py --sample_dir='./mnist' --out_dir='./results_mnist'
* --sample_dir: The path to the original MNIST images
* --out_dir: The path to save the synthesized images
```

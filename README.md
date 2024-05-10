# EEC_289A-HW2

### Install Python Environment (conda w/ Python=3.8)
```
pip3 install -r requirements.txt
```

### Run texture synthesis for all texture images
```
python3 main.py --sample_dir './images' --out_dir './results' --window_height 128 --window_width 128
```
* --sample_dir: The path (directory) to the original texture images
* --out_dir: The path (directory) to save the synthesized images
* --window_height: The height of the synthesized image
* --window_width: The width of the synthesized image

### Run texture synthesis for a texture image and specify the initial seed size
```
python3 main_sep.py --sample_path './images/5.png' --out_dir './results_seed' --window_height 128 --window_width 128 --kernel_size 23 --seed_size 35
```
* --sample_path: The path (image filename) to the texture image 
* --out_dir: The path (directory) to save the synthesized result
* --window_height: The height of the synthesized image
* --window_width: The width of the synthesized image
* --kernel_size: The size of the sliding window, which extracts patches and does the comparison
* --seed_size: The size of the initial seed

### Run MNIST image synthesis
```
python3 main_mnist.py --sample_dir './mnist' --out_dir './results_mnist' --window_height 128 --window_width 128
```
* --sample_dir: Download MNIST images and store them in the created directory (e.g. mnist)
* --out_dir: The path (directory) to save the synthesized images
* --window_height: The height of the synthesized image
* --window_width: The width of the synthesized image

### Run concatenated MNIST image synthesis
```
python3 main_mnist_cat.py --window_height 256 --window_width 256 --kernel_size 35
```
* --window_height: The height of the synthesized image
* --window_width: The width of the synthesized image
* --kernel_size: The size of the sliding window, which extracts patches and does the comparison

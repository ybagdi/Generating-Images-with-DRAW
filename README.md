# Generating-Images-Using-DRAW
PyTorch implementation of [DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/abs/1502.04623)

For blog on same check this "Link to be inserted after publishing the blog>

## Training
Make sure to have **data/** directory for respective types of runs. Run **`train_<mnist/svhn/cifar10>.py`** to start training. To change the hyperparameters of the network, update the values in the `param` dictionary in `train_<>.py`.

## Generating New Images
To generate new images run **`generate.py`**.
```sh
python3 generate.py -load_path /path/to/pth/checkpoint -num_output n
```

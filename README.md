# style-transfer

In this project we implemented a style transfer architecture based on https://arxiv.org/pdf/1603.08155.pdf

## Usage

Install dependencies: `pip install -r requirements.txt`

To train the model, configure the model parameters and which style image you would like to use in `config.py`, and then run `train.py`.

To test the model (pretrained models can be found at: https://drive.google.com/drive/folders/1IkmyaqtUYGQ5nieiWWrwSovlRGRMpRDy?usp=share_link), change the name of `CHECKPOINT_FILE` in `config.py` to the name of the trained model file (e.g. `4e5_starry.tar`). Then run `test.py` to stylize all the images in `test_images/` or configure and run `run.py` to run the model on a single image, video, camera feed, application, or folder. Note that the models may be slow without a GPU.

## Sample Results

![image](https://user-images.githubusercontent.com/84476225/235281716-c371ee49-d14f-4dad-a49b-1fe70a37ec79.png)


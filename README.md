# P3DC-Shot: Prior-Driven Discrete Data Calibration for Nearest-Neighbor Few-Shot Classification




## Backbone Training

We use the same backbone network and training strategies as 'S2M2_R'. Please refer to https://github.com/nupurkmr9/S2M2_fewshot for the backbone training.


## Extract and save features

After training the backbone as 'S2M2_R', extract features as below:

- Create an empty 'checkpoints' directory.

- Run:
```save_features
python save_plk.py --dataset [miniImagenet/CUB] 
```
### Or you can directly download the extracted features/pretrained models from the link:
https://drive.google.com/drive/folders/1IjqOYLRH0OwkMZo8Tp4EG02ltDppi61n?usp=sharing


After downloading the extracted features, please adjust your file path according to the code.


## Evaluate our distribution calibration

To evaluate our P3DC-Shot method, run:

```eval
python P3DC.py for 1-shot
python P3DC_5shot.py for 5-shot
```



## Reference

[Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087v3.pdf)

[https://github.com/nupurkmr9/S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot)

[Free Lunch for Few-shot Learning: Distribution Calibration](https://arxiv.org/abs/2101.06395)

[https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration)




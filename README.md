# Face Emotion Recognition in SVM

A simple facial emotion recognition classifier implemented by SVM

## prepare work

### Create an environment for the experiment using the following command

```shell
conda create -n SVM python=3.8
conda activate SVM
pip install -r requirements.txt
```

### Download the corresponding data set from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## try to use

The structure of our model is as follows

```
SVM
	|
	| - log (our experiment result)
	| - data (you should get this directory from kaggle)
	| - model (some models we use to extract feature)
		| - hog.py
		| - resnet.py
		| - ...
	| - main.py
	| - ...
```

We provide detailed parameter descriptions in `main.py`

## train and eval

```python
python -u main.py --kernel rbf --method {your_method} --gpu_id {your_gpu_id} --C 5 --gamma 0.02 > res.log
```
## our experiment result
| kernel  | feature method | other description | score  |
| :-----: | :------------: | :---------------: | :----: |
| sigmoid |       /        |         /         | 24.97  |
|   rbf   |       /        |         /         | 30.90  |
| linear  |       /        |         /         | not converged |
| linear  |      hog       |    patch(8,2)     | 45.31  |
|   rbf   |      hog       |    patch(8,2)     | 45.90  |
|   rbf   |      hog       |    patch(4,4)     | 51.46  |
|   rbf   |    hog+pca     |    patch(4,4)     | 51.52  |
|   rbf   |     align      |  dim: 11664->136  | 44.10  |
|   rbf   | hog |         C=5, gamma=0.02         | 57.65 |
|   rbf   | hog+aug |         C=5, gamma=0.02         | **59.07** |
|    	rbf	|resnet18+aug|C=5, gamma=0.02|**68.04**|


for more details, please see `./log` directory.
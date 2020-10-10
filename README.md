# Adversarial-Profile
### Under Develpment


## What does Adversarial Profile mean:


For a given CNN, adversarial profile of ith class (C_i) is a set of adversarial perturbations <img src="https://render.githubusercontent.com/render/math?math=\{\delta_{i,1},\cdots, \delta_{i,i-1},  \delta_{i,i%2B1},\cdots, \delta_{i,c} \}"> to any clean sample from class $i$ leads the target CNN to misclassify that sample to class j (i.e., if <img src="https://render.githubusercontent.com/render/math?math=x\in c_i , \:\:\: \mathrm{argmax}\:\:F(x+\delta_{i,j})=j"> ) with high probability; and ii) adding <img src="https://render.githubusercontent.com/render/math?math=\delta_{i,j}"> to any clean sample from other classes (except i), would lead the CNN to  misclassify that sample to any other class except $j$ (i.e., if <img src="https://render.githubusercontent.com/render/math?math=x\notin c_i , \:\:\: \mathrm{argmax} \:\:F(x+\delta_{i,j})\neq j"> ). 

<img src="figs/example.png" width=500 align=center> 


We say an adversarial perturbation <img src="https://render.githubusercontent.com/render/math?math=\delta_{i,j}">  is <img src="https://render.githubusercontent.com/render/math?math=p_{i,j}"> -intra-class transferable} if the probability of fooling CNN to the target class j for samples from source class i is <img src="https://render.githubusercontent.com/render/math?math=p_{i,j}">  (i.e., <img src="https://render.githubusercontent.com/render/math?math=Prob(\mathrm{argmax}\:\: F(x+\delta_{i,j})==j|x\in C_i)=p_{i,j}"> ).

Although, the adversarial perturbation <img src="https://render.githubusercontent.com/render/math?math=\delta_{i,j}">  is learned on source class $i$ but there is a possibility that this perturbation also works on samples from other classes (not class i). To measure transferability of an adversarial perturbation for other classes we define  Inter-Class Transferability. 


We call adversarial perturbation <img src="https://render.githubusercontent.com/render/math?math=\delta_{i,j}">  is <img src="https://render.githubusercontent.com/render/math?math=e_{i,j}"> inter-class-transferable if the probability of fooling the  CNN to the target class j for samples from other classes (not equal to i) is <img src="https://render.githubusercontent.com/render/math?math=e_{i,j}">  (i.e., <img src="https://render.githubusercontent.com/render/math?math=Prob(\mathrm{argmax}\:\: F(x+\delta_{i,j})==j|x\notin C_i)=e_{i,j}">.)

| <img src="figs/MNIST_InDist_Transferability.png" width=300> | <img src="figs/MNIST_OutDist_Transferability.png" width=300>
|:--:|:--:| 
| Intra class transferability matrix  |Inter class transferability matrix |

The element at [i,j]  in Inter class transferability matrix represents the value of p_{i,j}. Similarly,  the element at [i,j]  in intra class transferability matrix  represents the value of e_{i,j}. The larger value for p_{i,j}  and lower value for e_{i,j} are preferred.

## How to learn:
Finding an adversarial perturbation  (<img src="https://render.githubusercontent.com/render/math?math=\delta_{i,j}">) that is be able to fool all samples from class i to target class j is hard and computationally expensive. Therefore,  we only use n randomly selected samples from each source class to learn an adversarial perturbation and accept it for use in the adversarial profile if it can fool the CNN for at least p*n of them (0<p<1). 

Extended Carlini Wagnar Attack for Learning Targeted and Untargeted Universal Perturbation
This attack is an extension of https://github.com/rwightman/pytorch-nips2017-attack-example 


## How to run
### Install pythorch in a python virtual env
$python3 -m venv myenv

$source myenv/bin/activate.csh

$pip3 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

### Train the CNN and Adversarial Profiles
$python3 trainmodels.py --dataset mnist

$python3 TrainAdvProfile.py --dataset mnist

Please cite A. Rajabi , R. Bobba, "Adversarial Profile: Detecting Out-distribution Samples and Adversarial Examples for Pre-trained CNNs ", DSN workshop on Dependable and Secure Machine Learning (DSML) 2019 if you used this code [slides](DSML2019.pptx)




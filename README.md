# Adversarial-Profile
### Under Develpment
Extended Carlini Wagnar Attack for Learning Targeted and Untargeted Universal Perturbation
This attack is an extension of https://github.com/rwightman/pytorch-nips2017-attack-example 

| <img src="figs/MNIST_InDist_Transferability.png" width=300> | <img src="figs/MNIST_OutDist_Transferability.png" width=300>
|:--:|:--:| 
| Intra class transferability matrix  |Inter class transferability matrix |

The element at [i,j]  in Inter class transferability matrix represents the value of p_{i,j}. Similarly,  the element at [i,j]  in intra class transferability matrix  represents the value of e_{i,j}. The larger value for p_{i,j} and lower value for e_{i,j} are preferred.

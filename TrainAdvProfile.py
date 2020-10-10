from __future__ import print_function
import argparse
import torch
from torchvision import datasets,transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import gc
import model as AllModels
import os
from importlib import import_module
from UniversalCW import AttackCarliniWagnerL2
import numpy as np
torch.manual_seed(1)


def test(model, test_loader,device):
	model.eval()
	test_loss=0
	accuracy=0
	with torch.no_grad():
		for batch_indx, (data,target) in enumerate(test_loader):
			data,target=data.to(device),target.to(device)
			target_onehot=torch.nn.functional.one_hot(target,num_classes=10)
			output=model(data)
			#softmax_cross_entropy_with_logits 
			output=F.log_softmax(output, -1)
			test_loss+= torch.sum(- target_onehot *output, -1).mean()
			#test_loss+= F.nll_loss(output, target)
			pred=output.argmax(dim=1,keepdim=True)
			accuracy+=pred.eq(target.view_as(pred)).sum().item()
	accuracy/=len(test_loader.dataset)
	test_loss/=len(test_loader.dataset)
	print('Test accuracy {:.4f} and  \tLoss: {:.6f}'.format(accuracy,test_loss.item()))
	return test_loss, accuracy


def main():
	parser=argparse.ArgumentParser(description='PyTorch Learning MNIST')
	parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
	
	
	parser.add_argument('--dataset', type=str, default="mnist", help='dataset name')
	parser.add_argument('--model_type', type=str, default="vgg19", help='CIFAR model type')
	#parser.add_argument('--dataset', type=str, default="mnist", help='dataset name')
	parser.add_argument('--PATH', type=str, default="models/mnist.pt", help='model path')

	args=parser.parse_args()

	
	kwargs={'batch_size':args.batch_size}
	#transforms.ToTensor(),
	transform=transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	if torch.cuda.is_available():
		device=torch.device("cuda")
		kwargs.update({'num_workers':1, 'pin_memory':1,'shuffle':True})
		print('Using CUDA, GPU is available')
	else:
		device=torch.device("cpu")

	if args.dataset=='mnist':
		transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
		model=AllModels.mnist_model().to(device)
		train_data=datasets.MNIST('./data',train=True, download=True,transform=transform)
		test_data=datasets.MNIST('./data',train=False, download=True,transform=transform)
	elif args.dataset=='cifar10':
		m= ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
			'vgg19_bn', 'vgg19', 'ResNet', 'resnet20', 'resnet32', 'resnet44',
			 'resnet56', 'resnet110', 'resnet1202'
]
		if args.model_type in m:
			#model=AllModels.vgg19().to(device)
			
			#module = import_module(model,args.model_type.lower())
			func = getattr(AllModels, args.model_type.lower())
			model=func(10).to(device)
		else:
			print('Unknown Model Type. Select from '+ model_type.join())
			quit()
		train_data=datasets.CIFAR10('./data',train=True, download=True,transform=transform)
		test_data=datasets.CIFAR10('./data',train=False, download=True,transform=transform)
	else:
		print("Dataset Not Found")
		quit()

	
	model.load_state_dict(torch.load(args.PATH))
	model.eval()

	train_loader= torch.utils.data.DataLoader(train_data,**kwargs)
	test_loader= torch.utils.data.DataLoader(test_data,**kwargs)

	CW=AttackCarliniWagnerL2(search_steps=25)
	adv=np.zeros((10,10,1,28,28))
	for source in range(10):
		for targetclass in range(10):
			if source!=targetclass:
				
				source_data=[]
				for batch_indx, (data,target) in enumerate(train_loader):
				#print(target.shape)
					indx=np.where(target==source)[0]
					if len(indx)>1:
						if len(source_data)>0:
							source_data=np.concatenate((source_data, data[indx]), axis=0)
						else:
							source_data=data[indx]
					if len(source_data)>=100:
						break
				tar=(np.ones(len(source_data[:100]))*targetclass).astype(int)
	
				adv[source,targetclass,:,:,:]=CW.run( model, torch.Tensor(source_data[:100]).cuda(), torch.Tensor(tar).to(torch.int64).cuda())
	
		print('Learned Adversarial Profile for Source class'+str(source)+'to target class'+str(targetclass))
	np.save('AdversarialProfile.npy', adv)




if __name__ == '__main__':
	main()




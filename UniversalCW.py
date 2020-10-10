# https://github.com/rwightman/pytorch-nips2017-attack-example/blob/master/attacks/attack_carlini_wagner_l2.py
# Pytorch Universal Targeted Model of CW



import os
import sys
import torch
import numpy as np
from torch import optim
from torch import autograd
from helpers import *


class AttackCarliniWagnerL2:

    def __init__(self, targeted=True, search_steps=None, max_steps=None, cuda=True, debug=False):
        self.debug = debug
        self.targeted = targeted
        self.num_classes = 10
        self.confidence = 20  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        self.initial_const = 0.01  # bumped up from default of .01 in reference code
        self.binary_search_steps = search_steps or 5
        self.repeat = self.binary_search_steps >= 10
        self.max_steps = max_steps or 1000
        self.abort_early = True
        self.clip_min = -1.
        self.clip_max = 1.
        self.cuda = cuda
        self.clamp_fn = 'tanh'  # set to something else perform a simple clamp instead of tanh
        self.init_rand = False  # an experiment, does a random starting point help?

    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            if self.targeted:
                output[target] -= self.confidence
            else:
                output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target
        else:
            return output != target

    def _loss(self, output, target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        #print(target.shape,output.shape)
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)

        loss1 = torch.sum(scale_const * loss1)

        loss2 = dist.sum()
        #print(loss1,loss2,dist.shape)
        loss = loss1 + loss2
        return loss

    def _optimize(self, optimizer, model, input_var, modifier_var, target_var, scale_const_var, input_orig=None):
        # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
        if self.clamp_fn == 'tanh':
            input_adv = tanh_rescale(input_var+modifier_var , self.clip_min, self.clip_max)
        else:
            input_adv = torch.clamp(input_var+ modifier_var, self.clip_min, self.clip_max)

        output = model(input_adv)

        # distance to the original input data
        if input_orig is None:
            dist = l2_dist(input_adv, input_var, keepdim=False)
        else:
            dist = l2_dist(input_adv, input_orig, keepdim=False)

        loss = self._loss(output, target_var, dist, scale_const_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_np = loss.cpu()
        dist_np = dist.data.cpu().numpy()
        output_np = output.data.cpu().numpy()
        adv_noise_np = modifier_var.data.cpu().numpy()  # back to BHWC for numpy consumption
        return loss_np, dist_np, output_np, adv_noise_np

    def run(self, model, input, target,p_value=0.8):
        batch_size = input.size(0)

        # set the lower and upper bounds accordingly
        lower_bound = 0
        scale_const =  self.initial_const
        upper_bound =  1e10

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = 1e10
        
        

        # setup input (image) variable, clamp/scale as necessary
        if self.clamp_fn == 'tanh':
            # convert to tanh-space, input already int -1 to 1 range, does it make sense to do
            # this as per the reference implementation or can we skip the arctanh?
            input_var = autograd.Variable(torch_arctanh(input), requires_grad=False)
            input_orig = tanh_rescale(input_var, self.clip_min, self.clip_max)
        else:
            input_var = autograd.Variable(input, requires_grad=False)
            input_orig = None

        # setup the target variable, we need it to be in one-hot form for the loss function
        # target_onehot = torch.zeros(target.size() + (self.num_classes,))
        target_onehot=torch.nn.functional.one_hot(target,num_classes=self.num_classes)
        if self.cuda:
            target_onehot = target_onehot.cuda()
        #target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = autograd.Variable(target_onehot, requires_grad=False)

        # setup the modifier variable, this is the variable we are optimizing over
        img_shape=input.size()
        modifier = torch.zeros(img_shape[1:]).float()
        if self.init_rand:
            # Experiment with a non-zero starting point...
            modifier = torch.normal(means=modifier, std=0.001)
        if self.cuda:
            modifier = modifier.cuda()

        modifier_var = autograd.Variable(modifier, requires_grad=True)

        optimizer = optim.Adam([modifier_var], lr=0.0005)

        for search_step in range(self.binary_search_steps):
            print(' search step: {0}'.format( search_step))
            if self.debug:
                print('Const:', scale_const)
                
            best_l2 = 1e10
            best_score = [-1] * batch_size
            best_mod=np.zeros(img_shape[1:])
            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor =torch.tensor( scale_const)
            if self.cuda:
                scale_const_tensor = scale_const_tensor.cuda()
            scale_const_var = autograd.Variable(scale_const_tensor, requires_grad=False)

            prev_loss = 1e6
            for step in range(self.max_steps):
                # perform the attack
                loss, dist, output, adv_noise = self._optimize(
                    optimizer,
                    model,
                    input_var,
                    modifier_var,
                    target_var,
                    scale_const_var,
                    input_orig)

                if step % 100 == 0 or step == self.max_steps - 1:
                    print('Step: {0:>4}, loss: {1:6.4f}, dist: {2:8.5f}, modifier mean: {3:.5e}'.format(
                        step, loss, dist.mean(), modifier_var.data.mean()))

                if self.abort_early and step % (self.max_steps // 10) == 0:
                    if loss > prev_loss * 2.9999:
                        print('Aborting early...')
                        break
                    prev_loss = loss

                

                sys.stdout.flush()
                # end inner step loop

            # adjust the constants
            #print(target,np.argmax(output,1))
            batch_failure = 0
            batch_success = 0
            misclass=np.argmax(output,1)
            for i in range(batch_size):
                if self._compare(misclass[i], target[i]):
                    batch_success += 1
                else:
                    batch_failure += 1

            if batch_success>p_value*(batch_success+batch_success):
                    # successful, do binary search and divide const by two
                    upper_bound = min(upper_bound, scale_const)
                    if upper_bound < 1e9:
                        scale_const = (lower_bound + upper_bound) / 2
                    if self.debug:
                        print('{0:>2} successful attack, lowering const to {1:.3f}'.format(batch_success, scale_const))
            else:
                # failure, multiply by 10 if no solution found
                # or do binary search with the known upper bound
                lower_bound = max(lower_bound, scale_const)
                if upper_bound < 1e9:
                    scale_const= (lower_bound + upper_bound) / 2
                else:
                    scale_const *= 10
                if self.debug:
                    print('{0:>2} failed attack, raising const to {1:.3f}'.format(batch_success, scale_const))

            print('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()

            dist=np.linalg.norm(adv_noise)
            
            if  o_best_l2>dist and  batch_success>p_value*(batch_failure+batch_success):
                o_best_l2 = dist
                best_mod=adv_noise
                
            # end outer search loop

        return best_mod





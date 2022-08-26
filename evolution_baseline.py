import torch
import time
import random
from tqdm import tqdm
import numpy as np
import copy

def generate_subnets(sample_num=100, model_len=16, pruned=False, prune_points=None, type='resnet', lessskip=False):
    '''
    suggested:sample_num >= 700
    '''
    # import pdb;pdb.set_trace()
    if lessskip:
        skip_rate = [0.02 + 0.02*i for i in range(10)]
    else:
        skip_rate = [0.05 + 0.05 * i for i in range(15)]
    # skip_rate = [0.8+0.01*i for i in range(21)]
    # skip_rate = [0.3 + 0.035 * i for i in range(21)]
    # skip_rate = [0.1 + 0.035 * i for i in range(21)]
    subnets = []

    for _ in range(8):
        for i in range(len(skip_rate)):
            for distill_next_rate in [1., 0.8, 0.6, 0.5, 0.4, 0.2, 0.]:
                for prunerate in [0.,0.1,  0.2,0.4, 0.6]:
                    blockidx = 0
                    subnet = []
                    while blockidx < model_len:
                        if pruned:
                            choices = [-2,-1,0] if blockidx in prune_points else [0]
                        else:
                            choices = [0]
                        # dealing with resnets please use "choices = [0] if not pruned else [-2, -1, 0]"
                        if type == 'mbv2':
                            if 0 < blockidx < model_len - 1:  # for resnet do not 0 <, for mbv2, its 0<
                                choices.append(1)  # distill next one
                            if 1 < blockidx < model_len - 2:  # for resnet do not 1 <, for mbv2_120d, its 0 <, for mbv2_100 or 140, its 1 <
                                choices.append(2)  # distill next two
                            if 1 < blockidx < model_len - 2:  # for resnet do not 1 <, for mbv2_120d,  for mbv2_100 or 140, its 1 <
                                skipnext_prob = skip_rate[i] * distill_next_rate
                                skip_next_next_prob = skip_rate[i] * (1 - distill_next_rate)
                                if len(choices) == 3:
                                    probs = [(1 - skip_rate[i]), skipnext_prob, skip_next_next_prob]
                                else:
                                    probs = [(1 - skip_rate[i])*prunerate / 2., (1 - skip_rate[i])*prunerate / 2., (1 - skip_rate[i])*(1-prunerate), skipnext_prob, skip_next_next_prob]
                                    # print(probs)
                                choice = np.random.choice(choices, p=probs)
                            else:
                                choice = np.random.choice(choices)

                        elif type == 'mbv2d' or type == 'effi_s':
                            if 0 < blockidx < model_len - 1:  # for resnet do not 0 <, for mbv2, its 0<
                                choices.append(1)  # distill next one
                            if 0 < blockidx < model_len - 2:  # for resnet do not 1 <, for mbv2_120d, its 0 <, for mbv2_100 or 140, its 1 <
                                choices.append(2)  # distill next two
                            if 0 < blockidx < model_len - 2:  # for resnet do not 1 <, for mbv2_120d,  for mbv2_100 or 140, its 1 <
                                skipnext_prob = skip_rate[i] * distill_next_rate
                                skip_next_next_prob = skip_rate[i] * (1 - distill_next_rate)
                                if len(choices) == 3:
                                    probs = [(1 - skip_rate[i]), skipnext_prob, skip_next_next_prob]
                                else:
                                    probs = [skipnext_prob / 2., skip_next_next_prob / 2., 1 - skip_rate[i], skipnext_prob / 2., skip_next_next_prob / 2.]
                                choice = np.random.choice(choices, p=probs)
                            else:
                                choice = np.random.choice(choices)
                        else:
                            if blockidx < model_len - 1:  # for resnet do not 0 <, for mbv2, its 0<
                                choices.append(1)  # distill next one
                            if blockidx < model_len - 2:  # for resnet do not 1 <, for mbv2_120d, its 0 <, for mbv2_100 or 140, its 1 <
                                choices.append(2)  # distill next two
                            if blockidx < model_len - 2:  # for resnet do not 1 <, for mbv2_120d,  for mbv2_100 or 140, its 1 <
                                skipnext_prob = skip_rate[i] * distill_next_rate
                                skip_next_next_prob = skip_rate[i] * (1 - distill_next_rate)
                                if len(choices) == 3:
                                    probs = [(1 - skip_rate[i]), skipnext_prob, skip_next_next_prob]
                                else:
                                    probs = [(1 - skip_rate[i])/3, (1 - skip_rate[i])/3, (1 - skip_rate[i])/3, skipnext_prob, skip_next_next_prob]
                                choice = np.random.choice(choices, p=probs)
                            else:
                                choice = np.random.choice(choices)
                        if choice == 1:
                            subnet += [1, 99]
                            blockidx += 2
                        elif choice == 2:
                            subnet += [2, 99, 99]
                            blockidx += 3
                        else:
                            subnet.append(choice)
                            blockidx += 1
                        # else:  # pruning choice
                        #     subnet.append(choice)
                        #     for i in range(1, layer_lens[blockidx]):
                        #         subnet.append(99)
                        #     blockidx += layer_lens[blockidx]
                    subnets.append(subnet)
                    # print("skip rate:", skip_rate[i], "round:", test_times)\
    if lessskip:
        startidx = 0 if type != 'effi_s' else 1
        if type == 'mbv2':
            startidx = 2
        tmpsubnets = []
        for skipidx in range(startidx, model_len-1):
            subnet = [0 for _ in range(model_len)]
            subnet[skipidx] = 1
            subnet[skipidx+1] = 99
            tmpsubnets.append(subnet)
        for skipidx in range(startidx, model_len-2):
            subnet = [0 for _ in range(model_len)]
            subnet[skipidx] = 2
            subnet[skipidx+1] = 99
            subnet[skipidx+2] = 99
            tmpsubnets.append(subnet)
        for skipidx in range(startidx, model_len-3):
            subnet = [0 for _ in range(model_len)]
            subnet[skipidx] = 1
            subnet[skipidx+1] = 99
            for anotherskipidx in range(skipidx+2, model_len-1):
                subnet[anotherskipidx] = 1
                subnet[anotherskipidx+1] = 99
                tmpsubnet = copy.deepcopy(subnet)
                tmpsubnets.append(tmpsubnet)
                subnet[anotherskipidx] = 0
                subnet[anotherskipidx+1] = 0
        for skipidx in range(startidx, model_len-5):
            subnet = [0 for _ in range(model_len)]
            subnet[skipidx] = 2
            subnet[skipidx+1] = 99
            subnet[skipidx+2] = 99
            for anotherskipidx in range(skipidx+3, model_len-2):
                subnet[anotherskipidx] = 2
                subnet[anotherskipidx+1] = 99
                subnet[anotherskipidx+2] = 99
                tmpsubnet = copy.deepcopy(subnet)
                tmpsubnets.append(tmpsubnet)
                subnet[anotherskipidx] = 0
                subnet[anotherskipidx+1] = 0
                subnet[anotherskipidx+2] = 0
        for skipidx in range(startidx, model_len-4):
            subnet = [0 for _ in range(model_len)]
            subnet[skipidx] = 1
            subnet[skipidx+1] = 99
            for anotherskipidx in range(skipidx+2, model_len-2):
                subnet[anotherskipidx] = 2
                subnet[anotherskipidx+1] = 99
                subnet[anotherskipidx+2] = 99
                tmpsubnet = copy.deepcopy(subnet)
                tmpsubnets.append(tmpsubnet)
                subnet[anotherskipidx] = 0
                subnet[anotherskipidx+1] = 0
                subnet[anotherskipidx+2] = 0
        for skipidx in range(startidx, model_len-4):
            subnet = [0 for _ in range(model_len)]
            subnet[skipidx] = 2
            subnet[skipidx+1] = 99
            subnet[skipidx+2] = 99
            for anotherskipidx in range(skipidx+3, model_len-1):
                subnet[anotherskipidx] = 1
                subnet[anotherskipidx+1] = 99
                tmpsubnet = copy.deepcopy(subnet)
                tmpsubnets.append(tmpsubnet)
                subnet[anotherskipidx] = 0
                subnet[anotherskipidx+1] = 0
        print('newly added:', len(tmpsubnets))
    # print(tmpsubnets)
    new_subnets = []
    for subnetidx in range(len(subnets)):
        if subnets[subnetidx] not in new_subnets:
            new_subnets.append(subnets[subnetidx])
    # print(len(subnets), len(new_subnets))
    np.random.shuffle(new_subnets)
    if lessskip:
        new_subnets = tmpsubnets + new_subnets
        np.random.shuffle(new_subnets)
        newnew_subnets = []
        for subnetidx in range(len(new_subnets)):
            if new_subnets[subnetidx] not in newnew_subnets:
                newnew_subnets.append(new_subnets[subnetidx])
        print(len(new_subnets), len(newnew_subnets))
        np.random.shuffle(newnew_subnets)
        return newnew_subnets[:sample_num]

    return new_subnets[:sample_num]

class EvolutionFinder:
    def __init__(self, mutate_prob=0.1, propulation_size=150, parent_ratio=0.4, mutation_ratio=0.5, searching_times=50,
                 max_trying_times=1000, time_budget=0.01, batch_size=4, branch_choices=None, lats=None, pruned_points=None, modeltype='resnet', pruned=False):  # parent_ratio*2+mutation_ratio=1.0
        # if branch_choices is None:
        #     branch_choices = [0, 1, 2]
        self.mutate_prob = mutate_prob
        self.population_size = propulation_size
        self.parent_ratio = parent_ratio
        self.mutation_ratio = mutation_ratio
        self.searching_times = searching_times
        self.max_trying_times = max_trying_times
        self.time_budget = time_budget
        self.test_input = torch.randn(batch_size, 3, 224, 224)
        self.branch_choices = branch_choices  # [[0, 1, 2], [0, 1]...[0, 1]], len=len(subnetchoices)
        self.population = []
        self.latencys = []
        self.accs = []
        self.lats = lats
        self.prunedpoints = pruned_points
        # self.prune_points = None if self.model_lens is None else self.get_prune_points(model_lens)
        self.modeltype = modeltype
        self.pruned = pruned
        # import pdb;pdb.set_trace()

    def get_prune_points(self, model_lens):  # model_lens:[1, 2, 2, 3, 3, 3, 2, 2, 1...]
        old_model_len = model_lens[0]
        prune_points = []
        for blockidx in range(len(model_lens)):
            if model_lens[blockidx] != old_model_len:
                old_model_len = model_lens[blockidx]
                prune_points.append(blockidx)
        return prune_points

    # def get_subnet_latency(self, originalmodel, subnet):
    #     model = mytools.get_resnet_model_from_subnet(originalmodel, subnet, "output/resnet_weights/pth_for_device/")
    #     start = time.time()
    #     for _ in range(1000):
    #         out = model(self.test_input, validate_subnet=True)
    #     latency = (time.time() - start) / 1000
    #     return latency

    def init_population(self, model):
        # for _ in range(self.population_size):
        #     subnet = model.generate_random_subnet()
        #     self.population.append(subnet)
        example_subnet = model.generate_random_subnet()
        self.population = generate_subnets(sample_num=self.population_size, model_len=len(example_subnet),
                                                   pruned=self.pruned, prune_points=self.prunedpoints, type=self.modeltype, lessskip=False)

    def mutate_sample(self, model, oldsample, get_latency=True):  # [0..len(subnet)]
        # if random.random() < self.mutate_prob:
        sample = copy.deepcopy(oldsample)
        for _ in range(self.max_trying_times):
            # for blockidx in range(len(sample)):
            blockidx = 0  # 1 for mbv2 and 0 for resnets
            if self.modeltype == 'mbv2' or self.modeltype == 'effi_s':
                blockidx = 1
            while blockidx < len(sample):
                if (sample[blockidx] != 99) and (random.random() < self.mutate_prob):
                    sample[blockidx] = random.choice(self.branch_choices[blockidx])
                    if sample[blockidx] == 2:
                        sample[blockidx + 1] = 99
                        sample[blockidx + 2] = 99
                        blockidx += 3

                    elif sample[blockidx] == 1:
                        sample[blockidx + 1] = 99
                        blockidx += 2
                    else:
                        blockidx += 1
                    # else:  # [-2, -1, 0], pruning cases
                    #     for i in range(1, self.model_lens[blockidx]):
                    #         sample[blockidx + i] = 99
                    #     blockidx += self.model_lens[blockidx]
                    # if blockidx <= len(sample) - 1:
                    #     if sample[blockidx] == 99:
                    #         sample[blockidx] = 0
                    #         blockidx += 1
                    #         if blockidx <= len(sample) - 1:
                    #             if sample[blockidx] == 99:
                    #                 sample[blockidx] = 0
                    #                 blockidx += 1
                    while blockidx <= len(sample) - 1:
                        if sample[blockidx] == 99:
                            sample[blockidx] = 0
                            blockidx += 1
                        else:
                            break
                else:
                    blockidx += 1
            if get_latency:
                latency = self.get_subnet_latency(model, sample)
                if latency < self.time_budget:
                    return sample, latency
            else:
                return sample

    def cross_over(self, model, sample1, sample2, get_latency=True):
        for _ in range(self.max_trying_times):
            blockidx = 0
            new_sample = []
            while blockidx < len(sample1):
                if sample1[blockidx] == 99:
                    block_choice = sample2[blockidx]
                elif sample2[blockidx] == 99:
                    block_choice = sample1[blockidx]
                else:
                    block_choice = random.choice([sample1[blockidx], sample2[blockidx]])
                new_sample.append(block_choice)
                if block_choice == 1:
                    new_sample.append(99)
                    blockidx += 2
                elif block_choice == 2:
                    new_sample += [99, 99]
                    blockidx += 3
                else:
                    blockidx += 1
                # else:  # [-2, -1, 0]
                #     for i in range(1, self.model_lens[blockidx]):
                #         # new_sample[blockidx + i] = 99
                #         new_sample.append(99)
                #     blockidx += self.model_lens[blockidx]
            if get_latency:
                latency = self.get_subnet_latency(model, new_sample)
                if latency < self.time_budget:
                    return new_sample, latency
            else:
                return new_sample

    # def set_acc(self, acc):

    def evolution_search(self, model, validate, data_loader, args, loss_fn):  # validate is the function to get the acc and latency of a subnet
        self.init_population(model)
        mutation_number = int(round(self.population_size * self.mutation_ratio))
        parent_size = int(round(self.population_size * self.parent_ratio))

        best_valids = [-100]
        best_info = None

        for subnetidx in range(len(self.population)):
            acc, latency = validate(model, subnet=self.population[subnetidx], loader=data_loader, args=args, loss_fn=loss_fn, lats=self.lats)  # TODO:finish the validate function
            # score = acc - latency / self.time_budget
            score = (acc / 100. - latency / self.time_budget) if latency > self.time_budget else acc
            self.population[subnetidx] = (score, self.population[subnetidx], latency, acc)

        for iter in tqdm(range(self.searching_times), desc='searching for time budget %s' % (self.time_budget)):
            parents = sorted(self.population, key=lambda x: x[0])[::-1][:parent_size]  # reverted sort
            acc = parents[0][3]
            latency = parents[0][2]
            print('iter: {} Acc: {} Latency/TimeBudget: {} subnet:{}'.format(iter, acc, latency / self.time_budget, parents[0][1]))

            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_info = parents[0]

            self.population = parents

            for _ in range(mutation_number):
                subnet = self.population[np.random.randint(parent_size)][1]
                child_subnet = self.mutate_sample(model, subnet, get_latency=False)
                # print(child_subnet)
                acc, latency = validate(model, subnet=child_subnet, loader=data_loader, args=args, loss_fn=loss_fn, lats=self.lats)
                score = (acc / 100. - latency / self.time_budget) if latency > self.time_budget else acc
                self.population.append((score, child_subnet, latency, acc))

            for _ in range(self.population_size - mutation_number):
                father_subnet = self.population[np.random.randint(parent_size)][1]
                mother_subnet = self.population[np.random.randint(parent_size)][1]
                child_subnet = self.cross_over(model, father_subnet, mother_subnet, get_latency=False)
                # print(child_subnet)
                acc, latency = validate(model, subnet=child_subnet, loader=data_loader, args=args, loss_fn=loss_fn, lats=self.lats)

                score = (acc / 100. - latency / self.time_budget) if latency > self.time_budget else acc
                self.population.append((score, child_subnet, latency, acc))

        return best_valids, best_info
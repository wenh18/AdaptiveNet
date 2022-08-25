import torch
import torch_pruning as tp
import copy

### find out whether the selected subnet has blocks to be trained ###
def no_grad_fn(subnet):
    no_grad = True
    for i in subnet:
        if i > 1:
            no_grad = False
            break
    return no_grad

############################################### get pruned blocks ###############################################
def prune_mbv2_block(model, example, prune_rate):
    x = torch.rand(example.shape)
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, x)

    def prune_conv(conv, amount=0.2):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()

    prune_conv(model.conv_pw, prune_rate)
    prune_conv(model.conv_dw, prune_rate)
    return model

def prune_mbv2(model, prune_rate_list):
    '''return:[[block1_prunerate1, block1_prunerate2],...[blockn_prunerate1, blockn_prunerate2]]'''
    model.cpu()
    def prune_layer_blocks(input, layer, prune_rate):
        layer_blocks = []
        for i in range(len(layer)):
            tmp = layer[i](input)
            layer_blocks.append(prune_mbv2_block(layer[i], input, prune_rate))
            input = tmp
        return layer_blocks
    blocks = [[] for _ in range(15)]
    for rate in prune_rate_list:
        cp_model = copy.deepcopy(model)
        tmp_blocks = []
        x = torch.rand(1, 3, 224, 224)
        x = model.conv_stem(x)
        x = model.bn1(x)
        x = model.act1(x)
        x = model.blocks[0](x)
        for layeridx in range(1, 6):
            tmp_blocks += prune_layer_blocks(x, cp_model.blocks[layeridx], rate)
            x = model.blocks[layeridx](x)
        for ii in range(15):
            blocks[ii].append(tmp_blocks[ii])
    return blocks

############################################### load pretrained weights to elasticized supernet before training ###############################################
def load_to_MultiModel(MultiModel, path, stage=0):
    '''
    load efficientnet pretrained weight to elasticized supernet before training.
    return:
        stage=0(distillation): a 2-dimension list, the i-th of wich is [stageidx, blockidx], mapping the i-th multiblock to teachermodel.blocks[stageidx, blockidx]
        stage=1(further training): None
    '''
    if stage == 1:
        state = {}
        model_dict = MultiModel.state_dict()
        pretrained_model = torch.load(path, map_location=torch.device('cpu'))
        for k, v in pretrained_model.items():
            # print(k)
            key = k[7:] if "module" in k else k
            state[key] = v
        model_dict.update(state)
        MultiModel.load_state_dict(model_dict)
        for (name, parameter) in MultiModel.named_parameters():
            # print(name)
            name_list = name.split('.')
            if name_list[0] == "multiblocks" and name_list[2] == "0":
                print(name, 'frozen')
                parameter.requires_grad = False
            if name_list[0] == "conv_stem" or name_list[0] == "bn1" or name_list[0] == "bn2" or name_list[
                0] == "conv_head":
                print(name, 'frozen')
                parameter.requires_grad = False
            if "classifier" in name:
                parameter.requires_grad = False
                print(name, 'frozen')
        return None

    pretrained_model = torch.load(path)
    model_dict = MultiModel.state_dict()
    state = {}
    multimodel_idx = 0
    idx1 = idx2 = 0
    teacher_model_dict = [[0, 0]]
    for k, v in pretrained_model.items():
        if k in model_dict.keys():
            state[k] = v
        else:
            key_list = k.split('.', 3)
            new_idx1 = int(key_list[1])
            new_idx2 = int(key_list[2])
            if idx1 != new_idx1 or idx2 != new_idx2:
                idx1 = new_idx1
                idx2 = new_idx2
                multimodel_idx += 1
                teacher_model_dict.append([new_idx1, new_idx2])
            key = "multiblocks." + str(multimodel_idx) + ".0." + key_list[3]
            state[key] = v

    else:
        model_dict.update(state)
        MultiModel.load_state_dict(model_dict)
        for (name, parameter) in MultiModel.named_parameters():
            if name in state:
                # if ("multiblocks" in name and (name[14] == "0" or name[15] == "0")) or "classifier" in name or "conv_head" in name or "conv_stem" in name:
                print(name, 'frozen')
                # print(parameter.requires_grad)
                parameter.requires_grad = False
    print("loaded")
    teacher_model_dict.append([99, 99])
    return teacher_model_dict

def load_resnet_checkpoint(MultiModel, path, stage=0, args=None):
    if stage == 0:
        pretrained_model = torch.load(path)
        multimodel_dict = MultiModel.state_dict()
        state = {}
        multimodel_idx = 0
        idx1 = 1
        idx2 = 0
        teacher_model_dict = [[1, 0]]
        for k, v in pretrained_model.items():
            if k in multimodel_dict.keys():
                state[k] = v
            else:
                new_idx1 = int(k[5])
                new_idx2 = int(k[7])
                if idx1 != new_idx1 or idx2 != new_idx2:
                    idx1 = new_idx1
                    idx2 = new_idx2
                    multimodel_idx += 1
                    teacher_model_dict.append([new_idx1, new_idx2])
                key = "multiblocks." + str(multimodel_idx) + ".0." + k[9:]
                state[key] = v
        multimodel_dict.update(state)
        MultiModel.load_state_dict(multimodel_dict)
        for (name, parameter) in MultiModel.named_parameters():
            if name in state:
                print(name, 'frozen')
                parameter.requires_grad = False
        print("loaded")

        init_resnet_multiblocks(MultiModel, args)

        return teacher_model_dict

    state = {}
    model_dict = MultiModel.state_dict()
    pretrained_model = torch.load(path, map_location=torch.device('cpu'))
    for k, v in pretrained_model.items():
        state[k] = v
    model_dict.update(state)
    MultiModel.load_state_dict(model_dict)
    for (name, parameter) in MultiModel.named_parameters():
        name_list = name.split('.')
        if "module" in name[0]:
            if name_list[1] == "multiblocks" and name_list[3] == "0":
                print(name, 'frozen')
                parameter.requires_grad = False
            if "conv1" in name_list[1] or name_list[1] == "bn1" or name_list[1] == "bn2" or name_list[1] == "fc":
                print(name, 'frozen')
                parameter.requires_grad = False
        else:
            if name_list[0] == "multiblocks" and name_list[2] == "0":
                print(name, 'frozen')
                parameter.requires_grad = False
            if "conv1" in name_list[0] or name_list[0] == "bn1" or name_list[0] == "bn2" or name_list[0] == "fc":
                print(name, 'frozen')
                parameter.requires_grad = False
    return None


############################################### get teacher feature for distillation ###############################################
# efficientnet / mobilenetv2
def get_teacher_outputv2(x, teacher_model, teachernet_map, feature_idx_outs):
    '''
    feature_idx_outs denotes the student output idxs, ranging [1, len(model.multiblocks))
    teachernet_map is a 2-dimension list, its i-th item is [stage_idx, block_idx], mapping the i-th of elasticized block to the teachermodel.blocks[stage_idx][block_idx]
    '''
    x = teacher_model.conv_stem(x)
    x = teacher_model.bn1(x)
    x = teacher_model.act1(x)
    teacher_features = []
    teacher_feature_idxs = []
    for i in range(len(feature_idx_outs)):
        teacher_feature_idxs.append([teachernet_map[feature_idx_outs[i]][0], teachernet_map[feature_idx_outs[i]][1]])
    for stageidx in range(len(teacher_model.blocks)):
        for blockidx in range(len(teacher_model.blocks[stageidx])):
            if [stageidx, blockidx] in teacher_feature_idxs:
                teacher_features.append(x)
                if len(teacher_features) == len(feature_idx_outs):
                    return teacher_features
            x = teacher_model.blocks[stageidx][blockidx](x)
    teacher_features.append(x)
    return teacher_features

def get_resnet_output(x, teacher_model, teachernet_map, feature_idxs):
    idxs = []
    for i in range(len(feature_idxs)):
        idxs.append(teachernet_map[feature_idxs[i]])

    teacher_features = []
    x = teacher_model.conv1(x)
    x = teacher_model.bn1(x)
    x = teacher_model.act1(x)  # relu(x) for torchvision pretrained model
    x = teacher_model.maxpool(x)
    for idx1 in range(len(teacher_model.layer1)):
        # print(";;;;;", idx1)
        x = teacher_model.layer1[idx1](x)
        if [1, idx1] in idxs:
            # print(";;;", idx1)
            teacher_features.append(x)
            # print(x.shape)
    for idx2 in range(len(teacher_model.layer2)):
        x = teacher_model.layer2[idx2](x)
        if [2, idx2] in idxs:
            teacher_features.append(x)
    for idx3 in range(len(teacher_model.layer3)):
        x = teacher_model.layer3[idx3](x)
        if [3, idx3] in idxs:
            teacher_features.append(x)
    for idx4 in range(len(teacher_model.layer4)):
        x = teacher_model.layer4[idx4](x)
        if [4, idx4] in idxs:
            teacher_features.append(x)
    return teacher_features

################################################ init resnet supernet ###########################################
def init_resnet_multiblocks(model, args):
    if args.distributed:
        for blockchoice in model.module.multiblocks:
            for layerchoice in blockchoice[1:]:
                if layerchoice.bn1.weight.data.shape == blockchoice[0].bn1.weight.data.shape:
                    layerchoice.bn1.weight.data = blockchoice[0].bn1.weight.data.clone().detach()
                    layerchoice.bn1.bias.data = blockchoice[0].bn1.bias.data.clone().detach()
                    layerchoice.bn1.running_mean.data = blockchoice[0].bn1.running_mean.data.clone().detach()
                    layerchoice.bn1.running_var.data = blockchoice[0].bn1.running_var.data.clone().detach()
                    layerchoice.bn1.num_batches_tracked.data = blockchoice[0].bn1.num_batches_tracked.data.clone().detach()
                if layerchoice.bn2.weight.data.shape == blockchoice[0].bn2.weight.data.shape:
                    layerchoice.bn2.weight.data = blockchoice[0].bn2.weight.data.clone().detach()
                    layerchoice.bn2.bias.data = blockchoice[0].bn2.bias.data.clone().detach()
                    layerchoice.bn2.running_mean.data = blockchoice[0].bn2.running_mean.data.clone().detach()
                    layerchoice.bn2.running_var.data = blockchoice[0].bn2.running_var.data.clone().detach()
                    layerchoice.bn2.num_batches_tracked.data = blockchoice[0].bn2.num_batches_tracked.data.clone().detach()
                if layerchoice.bn3.weight.data.shape == blockchoice[0].bn3.weight.data.shape:
                    layerchoice.bn3.weight.data = blockchoice[0].bn3.weight.data.clone().detach()
                    layerchoice.bn3.bias.data = blockchoice[0].bn3.bias.data.clone().detach()
                    layerchoice.bn3.running_mean.data = blockchoice[0].bn3.running_mean.data.clone().detach()
                    layerchoice.bn3.running_var.data = blockchoice[0].bn3.running_var.data.clone().detach()
                    layerchoice.bn3.num_batches_tracked.data = blockchoice[0].bn3.num_batches_tracked.data.clone().detach()
                if layerchoice.conv1.weight.data.shape == blockchoice[0].conv1.weight.data.shape:
                    layerchoice.conv1.weight.data = blockchoice[0].conv1.weight.data.clone().detach()
                if layerchoice.conv2.weight.data.shape == blockchoice[0].conv2.weight.data.shape:
                    layerchoice.conv2.weight.data = blockchoice[0].conv2.weight.data.clone().detach()
                if layerchoice.conv3.weight.data.shape == blockchoice[0].conv3.weight.data.shape:
                    layerchoice.conv3.weight.data = blockchoice[0].conv3.weight.data.clone().detach()
                if hasattr(layerchoice, 'downsample'):
                    if layerchoice.downsample is not None and blockchoice[0].downsample is not None:
                        if layerchoice.downsample[0].weight.data.shape == blockchoice[0].downsample[0].weight.data.shape:
                            layerchoice.downsample[0].weight.data = blockchoice[0].downsample[0].weight.data.clone().detach()
                        if layerchoice.downsample[1].weight.data.shape == blockchoice[0].downsample[1].weight.data.shape:
                            layerchoice.downsample[1].weight.data = blockchoice[0].downsample[1].weight.data.clone().detach()
                            layerchoice.downsample[1].bias.data = blockchoice[0].downsample[1].bias.data.clone().detach()
                            layerchoice.downsample[1].running_mean.data = blockchoice[0].downsample[1].running_mean.data.clone().detach()
                            layerchoice.downsample[1].running_var.data = blockchoice[0].downsample[1].running_var.data.clone().detach()
                            layerchoice.downsample[1].num_batches_tracked.data = blockchoice[0].downsample[1].num_batches_tracked.data.clone().detach()
                # blockchoice[0].bn1.eval()
                # blockchoice[0].bn2.eval()
                # blockchoice[0].bn3.eval()
                # if hasattr(blockchoice[0], 'downsample'):
                #     if blockchoice[0].downsample is not None:
                #         blockchoice[0].downsample[1].eval()
    else:
        for blockchoice in model.multiblocks:
            idx = 0
            for layerchoice in blockchoice[1:]:
                idx += 1
                if layerchoice.bn1.weight.data.shape == blockchoice[0].bn1.weight.data.shape:
                    layerchoice.bn1.weight.data = blockchoice[0].bn1.weight.data.clone().detach()
                    layerchoice.bn1.bias.data = blockchoice[0].bn1.bias.data.clone().detach()
                    layerchoice.bn1.running_mean.data = blockchoice[0].bn1.running_mean.data.clone().detach()
                    layerchoice.bn1.running_var.data = blockchoice[0].bn1.running_var.data.clone().detach()
                    layerchoice.bn1.num_batches_tracked.data = blockchoice[0].bn1.num_batches_tracked.data.clone().detach()
                    print(idx)
                if layerchoice.bn2.weight.data.shape == blockchoice[0].bn2.weight.data.shape:
                    layerchoice.bn2.weight.data = blockchoice[0].bn2.weight.data.clone().detach()
                    layerchoice.bn2.bias.data = blockchoice[0].bn2.bias.data.clone().detach()
                    layerchoice.bn2.running_mean.data = blockchoice[0].bn2.running_mean.data.clone().detach()
                    layerchoice.bn2.running_var.data = blockchoice[0].bn2.running_var.data.clone().detach()
                    layerchoice.bn2.num_batches_tracked.data = blockchoice[0].bn2.num_batches_tracked.data.clone().detach()
                    print(idx)
                if layerchoice.bn3.weight.data.shape == blockchoice[0].bn3.weight.data.shape:
                    layerchoice.bn3.weight.data = blockchoice[0].bn3.weight.data.clone().detach()
                    layerchoice.bn3.bias.data = blockchoice[0].bn3.bias.data.clone().detach()
                    layerchoice.bn3.running_mean.data = blockchoice[0].bn3.running_mean.data.clone().detach()
                    layerchoice.bn3.running_var.data = blockchoice[0].bn3.running_var.data.clone().detach()
                    layerchoice.bn3.num_batches_tracked.data = blockchoice[0].bn3.num_batches_tracked.data.clone().detach()
                    print(idx)
                if layerchoice.conv1.weight.data.shape == blockchoice[0].conv1.weight.data.shape:
                    layerchoice.conv1.weight.data = blockchoice[0].conv1.weight.data.clone().detach()
                    print(idx)
                if layerchoice.conv2.weight.data.shape == blockchoice[0].conv2.weight.data.shape:
                    layerchoice.conv2.weight.data = blockchoice[0].conv2.weight.data.clone().detach()
                    print(idx)
                if layerchoice.conv3.weight.data.shape == blockchoice[0].conv3.weight.data.shape:
                    layerchoice.conv3.weight.data = blockchoice[0].conv3.weight.data.clone().detach()
                    print(idx)
                if hasattr(layerchoice, 'downsample'):
                    # resnet50
                    if layerchoice.downsample is not None and blockchoice[0].downsample is not None:
                        if layerchoice.downsample[0].weight.data.shape == blockchoice[0].downsample[0].weight.data.shape:
                            layerchoice.downsample[0].weight.data = blockchoice[0].downsample[0].weight.data.clone().detach()
                        if layerchoice.downsample[1].weight.data.shape == blockchoice[0].downsample[1].weight.data.shape:
                            layerchoice.downsample[1].weight.data = blockchoice[0].downsample[1].weight.data.clone().detach()
                            layerchoice.downsample[1].bias.data = blockchoice[0].downsample[1].bias.data.clone().detach()
                            layerchoice.downsample[1].running_mean.data = blockchoice[0].downsample[1].running_mean.data.clone().detach()
                            layerchoice.downsample[1].running_var.data = blockchoice[0].downsample[1].running_var.data.clone().detach()
                            layerchoice.downsample[1].num_batches_tracked.data = blockchoice[0].downsample[1].num_batches_tracked.data.clone().detach()
                    # resnet50d
                    # if layerchoice.downsample is not None and blockchoice[0].downsample is not None:
                    #     if layerchoice.downsample[1].weight.data.shape == blockchoice[0].downsample[2].weight.data.shape:
                    #         layerchoice.downsample[1].weight.data = blockchoice[0].downsample[2].weight.data.clone().detach()
                    #     if layerchoice.downsample[2].weight.data.shape == blockchoice[0].downsample[2].weight.data.shape:
                    #         layerchoice.downsample[2].weight.data = blockchoice[0].downsample[2].weight.data.clone().detach()
                    #         layerchoice.downsample[2].bias.data = blockchoice[0].downsample[2].bias.data.clone().detach()
                    #         layerchoice.downsample[2].running_mean.data = blockchoice[0].downsample[2].running_mean.data.clone().detach()
                    #         layerchoice.downsample[2].running_var.data = blockchoice[0].downsample[2].running_var.data.clone().detach()
                    #         layerchoice.downsample[2].num_batches_tracked.data = blockchoice[0].downsample[2].num_batches_tracked.data.clone().detach()
                    print(idx)
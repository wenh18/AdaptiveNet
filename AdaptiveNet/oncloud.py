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
    imp = tp.importance.MagnitudeImportance(p=2)
    # import pdb;pdb.set_trace()
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            ignored_layers.append(m)
    pruner = tp.pruner.GroupNormPruner(model=model, example_inputs=x,
                                       importance=imp, iterative_steps=1,
                                       ch_sparsity=prune_rate, ignored_layers=ignored_layers)
    pruner.step()
    # DG = tp.DependencyGraph().build_dependency(model, x)
    #
    # def prune_conv(conv, amount=0.2):
    #     strategy = tp.strategy.L1Strategy()
    #     pruning_index = strategy(conv.weight, amount=amount)
    #     # pruner = tp.pruner.MagnitudePruner(conv, )
    #     plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
    #     plan.exec()
    #
    # prune_conv(model.conv_pw, prune_rate)
    # prune_conv(model.conv_dw, prune_rate)
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
        # x = model.act1(x)
        x = model.blocks[0](x)
        for layeridx in range(1, 6):
            tmp_blocks += prune_layer_blocks(x, cp_model.blocks[layeridx], rate)
            x = model.blocks[layeridx](x)
        for ii in range(15):
            blocks[ii].append(tmp_blocks[ii])
    return blocks


def prune_resnet(model, prune_rate_list):
    '''return:[[block1_prunerate1, block1_prunerate2],...[blockn_prunerate1, blockn_prunerate2]]'''
    model.cpu()
    def prune_layer_blocks(input, layer, prune_rate):
        layer_blocks = []
        for i in range(len(layer)):
            tmp = layer[i](input)
            layer_blocks.append(prune_mbv2_block(layer[i], input, prune_rate))
            input = tmp
        return layer_blocks
    blocks = [[] for _ in range(16)]
    for rate in prune_rate_list:
        cp_model = copy.deepcopy(model)
        tmp_blocks = []
        x = torch.rand(1, 3, 224, 224)
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.act1(x)
        x = model.maxpool(x)
        # for layeridx in range(1, 6):
        tmp_blocks += prune_layer_blocks(x, cp_model.layer1, rate)
        x = model.layer1(x)

        tmp_blocks += prune_layer_blocks(x, cp_model.layer2, rate)
        x = model.layer2(x)

        tmp_blocks += prune_layer_blocks(x, cp_model.layer3, rate)
        x = model.layer3(x)

        tmp_blocks += prune_layer_blocks(x, cp_model.layer4, rate)
        # x = model.layer4(x)
        # import pdb;pdb.set_trace()
        for ii in range(16):
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

################################### for segmentation ##########################################
def _layeridx2multmodelidx(layeridx, block_idx, block_choice_idx, layer_len_list):  # layeridx = 1, 2, 3, 4
    multiblockidx = [0, block_choice_idx]
    for i in range(1, layeridx):
        multiblockidx[0] += layer_len_list[i - 1]
    multiblockidx[0] += block_idx
    multiblockidxstr = "multiblocks." + str(multiblockidx[0]) + "." + str(multiblockidx[1]) + "."
    return multiblockidxstr


def _headlayer2multiResDetLayer(layeridx, block_idx):
    newlayeridx = 2 if layeridx == 1 else layeridx
    newblockidx = block_idx if layeridx != 2 else block_idx + 3
    new_key = "model.backbone." + "layer" + str(newlayeridx) + "." + str(newblockidx) + ".0."
    return new_key


def no_new_subnet(subnet):
    no_new = True
    for i in range(len(subnet)):
        for j in range(len(subnet[i])):
            if subnet[i][j] != 0:
                no_new = False
    return no_new

def load_to_multimodel(path, multimodel, part="head", freeze_head=True):
    pretrained_model_dict = torch.load(path, map_location=torch.device('cpu'))
    multimodel_dict = multimodel.state_dict()
    state = {}
    if part == "head":  # load the head and the original main subnet, freeze this subnet and the head
        key_list = ["decoder", "segmentation_head"]
        for k, v in pretrained_model_dict.items():
            # head_flag = False
            for _k in key_list:
                if _k in k:
                    state[k] = v
                    # print(k)
        multimodel_dict.update(state)
        multimodel.load_state_dict(multimodel_dict)
        if freeze_head:
            for (name, parameter) in multimodel.named_parameters():
                if name in state:
                    parameter.requires_grad = False
                    print(';;;', name)
        return multimodel, state
    else:
        # multimodel_dict.items():model.backbone.conv1.weight, model.backbone.layer2.0.0.conv1.weight...
        # pretrained_model_dict.items():conv1.weight,multiblocks.0.0.conv1.weight...
        for k, v in multimodel_dict.items():
            k_list = k.split('.')
            if k_list[1] == "conv1" or k_list[1] == "bn1":
                state[k] = pretrained_model_dict[k[8:]]
                # print(k[8:])
            else:
                k_list = k.split('.', 4)
                if k_list[0] != "decoder" and k_list[0] != "segmentation_head":
                    multiblockidxstr = _layeridx2multmodelidx(layeridx=int(k_list[1][-1]), block_idx=int(k_list[2]),
                                                              block_choice_idx=int(k_list[3]), layer_len_list=[3, 4, 6, 3])
                    multiblockkey = multiblockidxstr + k_list[4]
                    # print(multiblockkey)
                    state[k] = pretrained_model_dict[multiblockkey]
        # for k, v in state.items():
        #     print("multimodelkeys:", k)
        multimodel_dict.update(state)
        multimodel.load_state_dict(multimodel_dict)
        for (name, parameter) in multimodel.named_parameters():
            namek = name.split('.')
            if name in state and (len(namek) > 3 and namek[3]=='0'):
                parameter.requires_grad = False
                print(';;;', name)
            if len(namek) == 3:
                parameter.requires_grad = False
                print(';;;', name)
        return multimodel


def freeze_main_subnet(multimodel, state):
    for (k, v) in multimodel.named_parameters():
        # print(k)
        if k in state:
            # print(";;;", k)
            v.requires_grad = False
    multimodel.model.backbone.conv1.eval()
    multimodel.model.backbone.bn1.eval()
    for blockidx in range(len(multimodel.model.backbone.layer2)):
        multimodel.model.backbone.layer2[blockidx][0].eval()
    for blockidx in range(len(multimodel.model.backbone.layer3)):
        multimodel.model.backbone.layer3[blockidx][0].eval()
    for blockidx in range(len(multimodel.model.backbone.layer4)):
        multimodel.model.backbone.layer4[blockidx][0].eval()
    multimodel.model.fpn.eval()
    multimodel.model.class_net.eval()
    multimodel.model.box_net.eval()


#################################### for object detection #######################################
def load_to_detectionmodel(path, multimodel, part="head", freeze_head=False):
    pretrained_model_dict = torch.load(path, map_location=torch.device('cpu'))
    multimodel_dict = multimodel.state_dict()
    state = {}
    if part == "head":  # load the head and the original main subnet, freeze this subnet and the head
        key_list = ["fpn", "class_net", "box"]
        for k, v in pretrained_model_dict.items():
            head_flag = False
            for _k in key_list:
                if _k in k:
                    key = "model." + k
                    state[key] = v
                    head_flag = True
            if not head_flag:
                k_list = k.split('.')
                if len(k_list) <= 3:
                    key = "model." + k
                else:
                    k_list = k.split('.', 3)
                    key = _headlayer2multiResDetLayer(int(k_list[1][-1]), int(k_list[2]))
                    key += k_list[3]
                state[key] = v
        # for k, v in state.items():
        #     print("headkeys:", k)
        multimodel_dict.update(state)
        multimodel.load_state_dict(multimodel_dict)
        if freeze_head:
            for (name, parameter) in multimodel.named_parameters():
                if name in state:
                    parameter.requires_grad = False
        return multimodel, state
    else:
        # multimodel_dict.items():model.backbone.conv1.weight, model.backbone.layer2.0.0.conv1.weight...
        # pretrained_model_dict.items():conv1.weight,multiblocks.0.0.conv1.weight...
        for k, v in multimodel_dict.items():
            if k == "anchors.boxes":
                continue
            k_list = k.split('.')
            if k_list[2] == "conv1" or k_list[2] == "bn1":
                state[k] = pretrained_model_dict[k[15:]]
            else:
                k_list = k.split('.', 5)
                if k_list[1] != "fpn" and k_list[1] != "class_net" and k_list[1] != "box_net":
                    multiblockidxstr = _layeridx2multmodelidx(layeridx=int(k_list[2][-1]), block_idx=int(k_list[3]),
                                                              block_choice_idx=int(k_list[4]), layer_len_list=[7, 6, 3])
                    multiblockkey = multiblockidxstr + k_list[5]
                    state[k] = pretrained_model_dict[multiblockkey]
        # for k, v in state.items():
        #     print("multimodelkeys:", k)
        multimodel_dict.update(state)
        multimodel.load_state_dict(multimodel_dict)
        return multimodel

def freeze_detection_main_subnet(multimodel, state):
    for (k, v) in multimodel.named_parameters():
        # print(k)
        if k in state:
            # print(";;;", k)
            v.requires_grad = False
    multimodel.model.backbone.conv1.eval()
    multimodel.model.backbone.bn1.eval()
    for blockidx in range(len(multimodel.model.backbone.layer2)):
        multimodel.model.backbone.layer2[blockidx][0].eval()
    for blockidx in range(len(multimodel.model.backbone.layer3)):
        multimodel.model.backbone.layer3[blockidx][0].eval()
    for blockidx in range(len(multimodel.model.backbone.layer4)):
        multimodel.model.backbone.layer4[blockidx][0].eval()
    multimodel.model.fpn.eval()
    multimodel.model.class_net.eval()
    multimodel.model.box_net.eval()

def freeze_detection_bn(multimodel):
    # for layer in multimodel.model.modules():
    #     if isinstance(layer, torch.nn.BatchNorm2d):
    #         layer.eval()
    multimodel.model.backbone.bn1.eval()
    for blockidx in range(len(multimodel.model.backbone.layer2)):
        multimodel.model.backbone.layer2[blockidx][0].bn1.eval()
        multimodel.model.backbone.layer2[blockidx][0].bn2.eval()
        multimodel.model.backbone.layer2[blockidx][0].bn3.eval()
        if multimodel.model.backbone.layer2[blockidx][0].downsample is not None:
            multimodel.model.backbone.layer2[blockidx][0].downsample[1].eval()
    for blockidx in range(len(multimodel.model.backbone.layer3)):
        multimodel.model.backbone.layer3[blockidx][0].bn1.eval()
        multimodel.model.backbone.layer3[blockidx][0].bn2.eval()
        multimodel.model.backbone.layer3[blockidx][0].bn3.eval()
        if multimodel.model.backbone.layer3[blockidx][0].downsample is not None:
            multimodel.model.backbone.layer3[blockidx][0].downsample[1].eval()
    for blockidx in range(len(multimodel.model.backbone.layer4)):
        multimodel.model.backbone.layer4[blockidx][0].bn1.eval()
        multimodel.model.backbone.layer4[blockidx][0].bn2.eval()
        multimodel.model.backbone.layer4[blockidx][0].bn3.eval()
        if multimodel.model.backbone.layer4[blockidx][0].downsample is not None:
            multimodel.model.backbone.layer4[blockidx][0].downsample[1].eval()
    for layer in multimodel.model.fpn.modules():
        # if isinstance(layer, torch.nn.BatchNorm2d):
        layer.eval()
    for layer in multimodel.model.class_net.modules():
        # if isinstance(layer, torch.nn.BatchNorm2d):
        layer.eval()
    for layer in multimodel.model.box_net.modules():
        # print(layer)
        # if isinstance(layer, torch.nn.BatchNorm2d):
        layer.eval()

def no_new_detection_subnet(subnet):
    no_new = True
    for i in range(len(subnet)):
        for j in range(len(subnet[i])):
            if subnet[i][j] != 0:
                no_new = False
    return no_new
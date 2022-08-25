import copy

import torch
from . import initialization as init
import time
import numpy as np
def get_latency(input, model, display=False):
    # inputcp = copy.deepcopy(input)
    # import pdb;pdb.set_trace()
    lats = []
    for i in range(1000):
        if display:
            print(i)
            import pdb;pdb.set_trace()
        t1 = time.time()
        _=model(input)
        torch.cuda.synchronize()
        if i > 100:
            lats.append(time.time() - t1)
    return np.mean(lats)

def get_layer_lats(x, layer):
    # xcp = copy.deepcopy(x)
    lats = []
    for blockidx in range(len(layer)):
        lats.append([])
        for blockchoice in range(len(layer[blockidx])):
            lats[-1].append(get_latency(x, layer[blockidx][blockchoice]))
        x = layer[blockidx][0](x)
    return lats, x

class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x, subnet):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # latencys = []
        #
        # latency = get_latency(x, self.check_input_shape)
        # latencys.append(latency)
        self.check_input_shape(x)

        # formerlayers = torch.nn.Sequential(*[self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.encoder.maxpool]).cuda()
        # xforlat = copy.deepcopy(x)
        # latencys.append(get_latency(xforlat, formerlayers))
        #
        # xforlat = formerlayers(xforlat)
        # lats, xforlat = get_layer_lats(xforlat, self.encoder.layer1)
        # latencys.append(lats)
        # lats, xforlat = get_layer_lats(xforlat, self.encoder.layer2)
        # latencys.append(lats)
        # lats, xforlat = get_layer_lats(xforlat, self.encoder.layer3)
        # latencys.append(lats)
        # lats, xforlat = get_layer_lats(xforlat, self.encoder.layer4)
        # latencys.append(lats)

        features = self.encoder(x, subnet)
        decoder_output = self.decoder(*features)
        # decoderlats=[]
        # for _ in range(500):
        #     t1 = time.time()
        #     decoder_output = self.decoder(*features)
        #     if _ > 200:
        #         torch.cuda.synchronize()
        #         decoderlats.append(time.time() - t1)
        # latencys.append(np.mean(decoderlats))
        # latencys.append(get_latency(decoder_output, self.segmentation_head))
        # print(latencys)
        # exit(0)
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x

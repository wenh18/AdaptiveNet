import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y, subnet=None):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, main=True, stage=1, subnet=None, getlat=False):
        print(subnet)
        self.on_epoch_start()
        if stage == 2 or stage == 0:
            freeze_main(self.model)
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                if subnet is None:
                    if main:
                        subnet = self.model.encoder.generate_main_subnet()
                    else:
                        subnet = self.model.encoder.generate_random_subnet()
                if stage == 0:
                    self.optimizer.zero_grad()
                    loss = self.model.encoder.forward(x, subnet, distill=True)
                    # loss = self.loss(prediction, y)
                    loss.backward()
                    self.optimizer.step()
                else:
                    loss, y_pred = self.batch_update(x, y, subnet)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                if stage != 0:
                    for metric_fn in self.metrics:
                        metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)

                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)
        if getlat:
            LATS=[0.0006080291692884401, [[0.0011172477607599752, 0.0011172082453336282, 0.0011180653853198976], [0.0009189702247220762, 0.0009168536299725661], [0.000919947767416813]], [[0.0012086358033244946, 0.0012221365537208497, 0.0012101474672854278], [0.0006532860014409457, 0.000653071186035971, 0.0006527282770006224], [0.0006541030955924606, 0.000655250899385955], [0.0006550634000139587]], [[0.0009606255837887094, 0.0009626443181870644, 0.0009633457833057781], [0.0005442982123081093, 0.0005451882376156341, 0.0005425887060112364], [0.0005405304561865343, 0.00054124650753645, 0.0005429536269954898], [0.0005395237410293935, 0.0005463898778624741, 0.0005407993732490582], [0.0005424754638162683, 0.0005428430368425584], [0.0005426817926867254]], [[0.0008723258441759561, 0.0008712366505114732, 0.0008710348301124785], [0.0005364667321736609, 0.0005389310096341856], [0.0005399265862147721]], 0.004941137360646437]
            lat = LATS[0]+LATS[-1]
            for i in range(len(subnet)):
                for j in range(len(subnet[i])):
                    if subnet[i][j] != 99:
                        lat += LATS[i+1][j][subnet[i][j]]
            return logs, lat
        return logs

def freeze_layer(layer):
    for blockidx in range(len(layer)):
        # import pdb;pdb.set_trace()
        layer[blockidx][0].eval()
        layer[blockidx][0].conv1.eval()
        layer[blockidx][0].conv2.eval()
        layer[blockidx][0].conv3.eval()
        layer[blockidx][0].bn1.eval()
        layer[blockidx][0].bn2.eval()
        layer[blockidx][0].bn3.eval()
        if hasattr(layer[blockidx][0], 'downsample') and layer[blockidx][0].downsample is not None:
            layer[blockidx][0].downsample[1].eval()

def freeze_main(model):
    print('freezing main')
    model.encoder.conv1.eval()
    model.encoder.bn1.eval()
    model.encoder.relu.eval()
    model.encoder.maxpool.eval()
    freeze_layer(model.encoder.layer1)
    freeze_layer(model.encoder.layer2)
    freeze_layer(model.encoder.layer3)
    freeze_layer(model.encoder.layer4)
    model.decoder.eval()
    model.segmentation_head.eval()

class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, subnet=None):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x, subnet)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, subnet=None):
        with torch.no_grad():
            prediction = self.model.forward(x, subnet)
            loss = self.loss(prediction, y)
        return loss, prediction

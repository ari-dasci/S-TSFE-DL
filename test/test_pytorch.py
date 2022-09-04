import sys
sys.path.append("..")
sys.path.append(".")
import os

import unittest
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from TSFEDL.data import MIT_BIH
from torch.utils.data.sampler import SubsetRandomSampler
from TSFEDL.models_pytorch import *
import inspect


def acc_from_logits(y_hat, y):
    y_hat = F.softmax(y_hat, dim=1)
    preds = torch.argmax(y_hat, dim=1)
    acc = (preds == y).sum().item() / len(y)
    return acc


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pl.seed_everything(42)
        if not os.path.isdir("physionet.org/files/mitdb/1.0.0/"):
            os.system("wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/")

    def trainModel(self, model, data_length, epochs, hot_coded=False, evaluate_test=True):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    
        mit_bih = MIT_BIH(path="physionet.org/files/mitdb/1.0.0/", return_hot_coded=hot_coded)
        mit_bih.x = mit_bih.x[:10]
        mit_bih.y = mit_bih.y[:10]
        tra_size = int(len(mit_bih) * 0.8)
        tst_size = len(mit_bih) - tra_size
        train, test = torch.utils.data.random_split(mit_bih, [tra_size, tst_size])
        train_loader = torch.utils.data.DataLoader(train, batch_size=1, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test, batch_size=1, num_workers=0)
        # train_sampler = SubsetRandomSampler(range(len(mit_bih)))
        # train_loader = torch.utils.data.DataLoader(mit_bih, batch_size=256, sampler=train_sampler, num_workers=8)


        # Train
        trainer = None
        if torch.cuda.is_available():
            trainer = pl.Trainer(gpus=1, max_epochs=epochs)
        else:
            trainer = pl.Trainer(max_epochs=epochs)
        trainer.fit(model, train_loader)
        if evaluate_test:
            test_results = None
            if "model" in inspect.getfullargspec(trainer.test).args:
                test_results = trainer.test(model, test_loader)
            else:
                test_results = trainer.test(test_loader)
            return test_results[0]['test_acc_epoch']

    #@unittest.skip
    def test_LiOhShu(self):
        model = LihOhShu(in_features=1,
                         loss=nn.CrossEntropyLoss(),
                         metrics={'acc': acc_from_logits},
                         optimizer=torch.optim.Adam,
                         lr=0.001)

        acc = self.trainModel(model, 2000, 1)
        assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_OhShuLi(self):
        model = OhShuLih(in_features=1,
                         loss=nn.CrossEntropyLoss(),
                         metrics={"acc": acc_from_logits},
                         optimizer=torch.optim.Adam,
                         lr=0.001)

        acc = self.trainModel(model, 1000, 1)
        assert 1.0 >= acc >= 0


    # def test_YiboGao(self):  # TODO: Test more carefully
    #     model = YiboGao(in_features=1,
    #                      metrics={"acc": acc_from_logits},
    #                      optimizer=torch.optim.Adam,
    #                      lr=0.001)
    #
    #     acc = self.trainModel(model, 1000, 10, True)
    #     assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_YaoQihang(self):
        model = YaoQihang(in_features=1,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 250, 1)
        assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_HtetMyetLynn(self):
        model = HtetMyetLynn(in_features=1,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 750, 1)
        assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_YildirimOzal(self):
        model = YildirimOzal(
            input_shape=(1,1000),
            top_module=nn.Linear(32, 5),
            loss=nn.MSELoss(),
            metrics={"acc": acc_from_logits},
            optimizer=torch.optim.Adam,
            lr=0.001
        )
        # First, train the autoencoder
        self.trainModel(model, 260, 1, False, False)

        # Next, train the LSTM and the classifier with the encoded features.
        model.train_autoencoder = False
        model.encoder.requires_grad_(False)
        model.decoder.requires_grad_(False)
        model.loss = nn.CrossEntropyLoss()
        acc = self.trainModel(model, 1000, 1, False, True)

        assert 1.0 >= acc >= 0

    #####################################################

    #@unittest.skip
    def test_KhanZulfiqar(self):
        model = KhanZulfiqar(in_features=1,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 1000, 1)
        assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_ZhengZhenyu(self):
        model = ZhengZhenyu(in_features=1,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 128, 1)
        assert 1.0 >= acc >= 0

    # # No funciona
     ###@unittest.skip
    # def test_HouBoroui(self):
    #     model = HouBoroui(
    #         in_features=1,
    #         top_module=nn.Linear(1, 5),
    #         loss=nn.MSELoss(),
    #         metrics={"acc": acc_from_logits},
    #         optimizer=torch.optim.Adam,
    #         lr=0.001
    #     )
    #     # First, train the autoencoder
    #     self.trainModel(model, 1000, 30, False, False)
    #
    #     # Next, train the LSTM and the classifier with the encoded features.
    #     model.train_autoencoder = False
    #     model.encoder.requires_grad_(False)
    #     model.decoder.requires_grad_(False)
    #     model.loss = nn.CrossEntropyLoss()
    #     acc = self.trainModel(model, 1000, 30, False, True)
    #
    #     assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_WangKejun(self):
        model = WangKejun(in_features=1,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 100, 1)
        assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_ChenChen(self):
        model = ChenChen(in_features=1,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 3600, 1)
        assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_KimTaeYoung(self):
        model = KimTaeYoung(in_features=1,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 100, 1)
        assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_GenMinxing(self):
        model = GenMinxing(in_features=1000,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 100, 1)
        assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_FuJiangmeng(self):
        model = FuJiangmeng(in_features=1,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 100, 1)
        assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_ShiHaotian(self):
        model = ShiHaotian(in_features=1,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 100, 1)
        assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_HuangMeiLing(self):
        model = HuangMeiLing(in_features=1,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 100, 1)
        assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_SharPar(self):
        model = SharPar(in_features=1,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 100, 1)
        assert 1.0 >= acc >= 0

    #@unittest.skip
    def test_DaiXiLi(self):
        model = DaiXiLi(in_features=1,
                          loss=nn.CrossEntropyLoss(),
                          metrics={"acc": acc_from_logits},
                          optimizer=torch.optim.Adam,
                          lr=0.001
                          )
        acc = self.trainModel(model, 100, 1)
        assert 1.0 >= acc >= 0


if __name__ == '__main__':
    unittest.main()

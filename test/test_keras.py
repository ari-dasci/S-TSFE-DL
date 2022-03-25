import sys
sys.path.append("..")
sys.path.append(".")
import os
import random as rn
import unittest

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow import keras

import TSFEDL.models_keras as TSFEDL
from TSFEDL.data import read_mit_bih


class TestMethods(unittest.TestCase):

    def setUp(self) -> None:
        # https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
        os.environ['PYTHONHASHSEED'] = '0'

        # The below is necessary for starting Numpy generated random numbers
        # in a well-defined initial state.
        np.random.seed(42)

        # The below is necessary for starting core Python generated random numbers
        # in a well-defined state.
        rn.seed(12345)

        # The below tf.set_random_seed() will make random number generation
        # in the TensorFlow backend have a well-defined initial state.
        # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
        tf.random.set_seed(1234)
        if not os.path.isdir("physionet.org/files/mitdb/1.0.0/"):
            os.system("wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/")

    def getData(self, size=1000, test_size=0.2):
        """Gets the segments with the specified size from the MIT-BIH arrythmia dataset.
            Returns a partition of the data as follow:
            (X_tra, y_tra, y_hotcoded_tra) -> the instances and labels of the training set
            (X_tst, y_tst, y_hotcoded_tst) -> the instances and labels of the test set.
        """
        print("Load data...")
        dir = "physionet.org/files/mitdb/1.0.0/"
        X, y = read_mit_bih(dir, fixed_length=size)
        y_hot_encoded = np.zeros((y.size, y.max() + 1))
        y_hot_encoded[np.arange(y.size), y] = 1
        X_tra, X_tst, y_tra, y_tst, y_HC_tra, y_HC_tst = train_test_split(X, y, y_hot_encoded,
                                                                          test_size=test_size,
                                                                          random_state=1234,
                                                                          stratify=y)
        return (X_tra[:1], y_tra[:1], y_HC_tra[:1]), (X_tst[:1], y_tst[:1], y_HC_tst[:1])

    def trainModel(self, model, X_tra, y_tra, y_hot_encoded_tra,
                   X_tst, y_tst, y_hot_encoded_tst,
                   loss=keras.losses.categorical_crossentropy,
                   metrics=['accuracy'],
                   batch_size=256,
                   epochs=6):
        model.compile(optimizer='Adam', loss=loss, metrics=metrics)
        model.fit(X_tra, y_hot_encoded_tra, batch_size=batch_size, epochs=epochs, verbose=1)

        # evaluate
        pred = np.argmax(model.predict(X_tst), axis=1)
        acc = accuracy_score(y_tst, pred)
        print("Accuracy: " + str(acc))
        return acc, pred

    #@unittest.skip
    def test_YiboGao(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=500)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model, loss = TSFEDL.YiboGao(input_tensor=input, include_top=True, return_loss=True)
        model.summary()

        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           loss=loss,
                                           epochs=1)

        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('YiboGao_test.h5')

        # re-load de model
        new_model = TSFEDL.YiboGao(input_tensor=input, weights='YiboGao_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('YiboGao_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_YaoQihang(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=250)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))

        # The number units, filters, kernel_sizes, etc. in this model are not specified in the original paper.
        model = TSFEDL.YaoQihang(input_tensor=input, include_top=True)
        model.summary()

        # Train the model
        # If the predictions are ok, the sum of predictions should be 1: test it
        model.compile(optimizer='Adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        model.fit(X_tra, y_hc_tra, batch_size=256, epochs=1, verbose=0)
        p = model.predict(X_tst)
        np.testing.assert_almost_equal(np.sum(p, axis=1), np.ones((X_tst.shape[0],)), decimal=6)

        # Once check, train the model as usual
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('YaoQihang_test.h5')

        # re-load de model
        new_model = TSFEDL.YaoQihang(input_tensor=input, weights='YaoQihang_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('YaoQihang_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_ZhangJin(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=250)
        input = keras.Input((15360, 12))
        model = TSFEDL.ZhangJin(input_tensor=input, include_top=True)

        # The output shapes matches the description of the paper at Section 5.2
        model.summary()

        # Now, train the model with our data.
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.ZhangJin(input_tensor=input, include_top=True)

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('ZhangJin_test.h5')

        # re-load de model
        new_model = TSFEDL.ZhangJin(input_tensor=input, weights='ZhangJin_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('ZhangJin_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_HtetMyetLynn(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=750)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.HtetMyetLynn(input_tensor=input, include_top=True)
        model.summary()

        # assert that output shapes are the same that in Figure 5 of the paper.
        model_output_shapes = [l.output_shape for l in model.layers
                               if not isinstance(l, keras.layers.Dropout)
                               if not isinstance(l, keras.layers.Conv1D)]
        desired_output_shape = [[(None, 750, 1)],
                                (None, 375, 30),
                                (None, 187, 30),
                                (None, 93, 60),
                                (None, 46, 60),
                                (None, 80),
                                (None, 5)]
        self.assertTrue(model_output_shapes == desired_output_shape,
                        msg="Output shapes do not match. Model shapes: " + str(model_output_shapes))

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('HtetMyetLynn_test.h5')

        # re-load de model
        new_model = TSFEDL.HtetMyetLynn(input_tensor=input, weights='HtetMyetLynn_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('HtetMyetLynn_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_CaiWenjuann(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((2160, 12))  # The shape of the paper's data.

        # The number units, filters, kernel_sizes, etc. in this model are not specified in the original paper.
        model = TSFEDL.CaiWenjuan(input_tensor=input, include_top=True, classes=2)
        model.summary()
        # In the paper, for 12-variables ECG data, only the number of paramwters of the final model are given.
        # The number of params in this case are 69087. Therefore, we assert that the number of params is almost the same
        assert 70000 >= model.count_params() >= 68500

        # Once checked that model's parameters are almost the same as in the original paper. We train the model with
        # our data.
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.CaiWenjuan(input_tensor=input, include_top=True)

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('CaiWenjuan_test.h5')

        # re-load de model
        new_model = TSFEDL.CaiWenjuan(input_tensor=input, weights='CaiWenjuan_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('CaiWenjuan_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_KimMinGu(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))

        # The number units, filters, kernel_sizes, etc. in this model are not specified in the original paper.
        models = TSFEDL.KimMinGu(input_tensor=input, include_top=True)
        preds = []
        for i, model in enumerate(models):
            model.summary()

            # Train the model
            acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                               batch_size=256, epochs=1)
            assert 1.0 >= acc >= 0
            preds.append(predictions)

            # save weights
            model.save_weights("KimMinGu_test_" + str(i) + ".h5")

        # re-load model
        w = ["KimMinGu_test_" + str(i) + ".h5" for i in range(6)]
        models = TSFEDL.KimMinGu(input_tensor=input, weights=w)
        for i, model in enumerate(models):
            new_predictions = np.argmax(model.predict(X_tst), axis=1)
            np.testing.assert_allclose(preds[i], new_predictions, rtol=1e-6, atol=1e-6)
            os.remove(w[i])

        return True

    #@unittest.skip
    def test_YildirimOzal(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=260)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        ae, enc, classifier = TSFEDL.YildirimOzal(input_tensor=input, include_top=True)
        ae.summary()
        enc.summary()
        classifier.summary()

        # assert that output shapes are the same that in Table 3 of the paper.
        model_output_shapes = [l.output_shape for l in ae.layers
                               if not isinstance(l, keras.layers.Dropout)]
        desired_output_shape = [[(None, 260, 1)],
                                (None, 260, 16),
                                (None, 130, 16),
                                (None, 130, 64),
                                (None, 130, 64),
                                (None, 65, 64),
                                (None, 65, 32),
                                (None, 65, 1),
                                (None, 32, 1),
                                (None, 32, 1),
                                (None, 32, 32),
                                (None, 64, 32),
                                (None, 64, 64),
                                (None, 128, 64),
                                (None, 128, 16),
                                (None, 2048),
                                (None, 260)]
        self.assertTrue(model_output_shapes == desired_output_shape,
                        msg="Output shapes do not match. Model shapes: " + str(model_output_shapes))

        # Train the autoencoder
        ae.compile(optimizer='Adam', loss=keras.losses.MSE, metrics=['mse'])
        ae.fit(X_tra, X_tra, batch_size=256, epochs=1, verbose=0)
        preds = ae.predict(X_tst)
        # Compute average MSE
        avg_mse_per_instance = [mean_squared_error(true, pred) for true, pred in zip(X_tst, preds)]
        avg_mse = np.mean(avg_mse_per_instance)
        print("Average MSE: ", avg_mse)


        # Train the LSTM (mark the encoder part as non-trainable)
        enc.trainable = False
        ae.trainable = False
        acc, class_preds = self.trainModel(classifier, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst, epochs=1)
        # classifier.compile(optimizer='Adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        # classifier.fit(X_tra, y_hc_tra, batch_size=256, epochs=30, verbose=0)
        # class_preds = np.argmax(classifier.predict(X_tst), axis=1)
        # acc = accuracy_score(y_tst, class_preds)
        # print("Accuracy: ", acc)

        # Assert test accuracy above 90%
        assert 1.0 >= acc >= 0

        # save weights
        ae.save_weights('YildirimOzal_test.h5')
        classifier.save_weights('YildirimOzal_classifier_test.h5')

        # Assert testing prediction
        # re-load de models
        ae, enc, classifier = TSFEDL.YildirimOzal(input_tensor=input,
                                                      autoencoder_weights='YildirimOzal_test.h5',
                                                      lstm_weights='YildirimOzal_classifier_test.h5')
        new_predictions = ae.predict(X_tst)
        np.testing.assert_allclose(preds, new_predictions, rtol=1e-6, atol=1e-6)

        new_predictions = np.argmax(classifier.predict(X_tst), axis=1)
        os.remove('YildirimOzal_test.h5')
        os.remove('YildirimOzal_classifier_test.h5')
        return np.testing.assert_allclose(class_preds, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_KongZhengmin(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))

        # The number units, filters, kernel_sizes, etc. in this model are not specified in the original paper.
        model = TSFEDL.KongZhengmin(input_tensor=input, include_top=True)
        model.summary()

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('KongZhengmin_test.h5')

        # re-load de model
        new_model = TSFEDL.KongZhengmin(input_tensor=input, weights='KongZhengmin_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('KongZhengmin_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_WeiXiaoyan(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.WeiXiaoyan(input_tensor=input, include_top=True)
        model.summary()  # Description of layers available at table 2 of the paper. No output shapes nor nº of params available

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('WeiXiaoyan_test.h5')

        # re-load de model
        new_model = TSFEDL.WeiXiaoyan(input_tensor=input, weights='WeiXiaoyan_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('WeiXiaoyan_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_GaoJunLi(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.GaoJunLi(input_tensor=input, include_top=True)
        model.summary()  # No parameters, layer descriptions or output shapes in the paper...

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('GaoJunLi_test.h5')

        # re-load de model
        new_model = TSFEDL.GaoJunLi(input_tensor=input, weights='GaoJunLi_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('GaoJunLi_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_LihOhShu(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=2000)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.LihOhShu(input_tensor=input, include_top=True)
        model.summary()  # No parameters, layer descriptions or output shapes in the paper...

        # assert that output shapes are the same that in Table 3 of the paper.
        model_output_shapes = [l.output_shape for l in model.layers
                               if not isinstance(l, keras.layers.Dropout)]
        desired_output_shape = [[(None, 2000, 1)],
                                (None, 1981, 3),
                                (None, 990, 3),
                                (None, 981, 6),
                                (None, 490, 6),
                                (None, 486, 6),
                                (None, 243, 6),
                                (None, 239, 6),
                                (None, 119, 6),
                                (None, 110, 6),
                                (None, 55, 6),
                                (None, 10),
                                (None, 8),
                                (None, 8),
                                (None, 5)]
        self.assertTrue(model_output_shapes == desired_output_shape,
                        msg="Output shapes do not match. Model shapes: " + str(model_output_shapes))

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('LihOhShu_test.h5')

        # re-load de model
        new_model = TSFEDL.LihOhShu(input_tensor=input, weights='LihOhShu_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('LihOhShu_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_HuangMeiLing(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.HuangMeiLing(input_tensor=input, include_top=True)
        model.summary()  # No parameters, layer descriptions or output shapes in the paper...

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('HuangMeiLing_test.h5')

        # re-load de model
        new_model = TSFEDL.HuangMeiLing(input_tensor=input, weights='HuangMeiLing_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('HuangMeiLing_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_ShiHaotian(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.ShiHaotian(input_tensor=input, include_top=True)
        model.summary()  # No parameters, layer descriptions or output shapes in the paper...

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('ShiHaotian_test.h5')

        # re-load de model
        new_model = TSFEDL.ShiHaotian(input_tensor=input, weights='ShiHaotian_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('ShiHaotian_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_FuJiangmeng(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.FuJiangmeng(input_tensor=input, include_top=True)
        model.summary()  # No parameters, layer descriptions or output shapes in the paper...

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('FuJiangmeng_test.h5')

        # re-load de model
        new_model = TSFEDL.FuJiangmeng(input_tensor=input, weights='FuJiangmeng_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('FuJiangmeng_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_GenMinxing(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.GenMinxing(input_tensor=input, include_top=True)
        model.summary()  # No parameters, layer descriptions or output shapes in the paper...

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('GenMinxing_test.h5')

        # re-load de model
        new_model = TSFEDL.GenMinxing(input_tensor=input, weights='GenMinxing_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('GenMinxing_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_ChenChen(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=3600)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.ChenChen(input_tensor=input, include_top=True)
        model.summary()

        # assert that output shapes are the same that in Table 2 of the paper.
        model_output_shapes = [l.output_shape for l in model.layers
                               if not isinstance(l, keras.layers.Flatten)]
        desired_output_shape = [[(None, 3600, 1)],
                                (None, 3350, 5),
                                (None, 1675, 5),
                                (None, 1526, 5),
                                (None, 763, 5),
                                (None, 664, 10),
                                (None, 332, 10),
                                (None, 252, 20),
                                (None, 126, 20),
                                (None, 66, 20),
                                (None, 33, 20),
                                (None, 20, 10),
                                (None, 10, 10),
                                (None, 10, 32),
                                (None, 10, 64),
                                (None, 5)]
        self.assertTrue(model_output_shapes == desired_output_shape,
                        msg="Output shapes do not match. Model shapes: " + str(model_output_shapes))

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        print(acc)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('ChenChen_test.h5')

        # re-load de model
        new_model = TSFEDL.ChenChen(input_tensor=input, weights='ChenChen_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('ChenChen_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_KimTaeYoung(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.KimTaeYoung(input_tensor=input, include_top=True)
        model.summary()

        # Check if params matches the description in the paper at table 2
        # Note: In the original paper, the time-series is windowed. Therefore, a TimeDistributed layer is employed
        # to traverse all the generated windows. Here, we do not implement this as it is problem-specific.
        # If you need this layer, you should manually add it.
        params = [p.count_params() for p in model.layers]
        desired_params = [0, 192, 0, 8256, 0, 33024, 2080, 165]
        self.assertTrue(params == desired_params,
                        msg="Number of parameters do not match. Model params: " + str(params))
        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst,
                                           batch_size=256, epochs=1)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('KimTaeYoung_test.h5')

        # re-load de model
        new_model = TSFEDL.KimTaeYoung(input_tensor=input, weights='KimTaeYoung_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('KimTaeYoung_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_WangKejun(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.WangKejun(input_tensor=input, include_top=True)
        model.summary()

        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra,
                                           X_tst, y_tst, y_hc_tst, batch_size=256, epochs=1)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('WangKejun_test.h5')

        # re-load de model
        new_model = TSFEDL.WangKejun(input_tensor=input, weights='WangKejun_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('WangKejun_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_HouBoroui(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model, encoder = TSFEDL.HouBoroui(input_tensor=input)
        model.summary()  # Layer shapes not available in the paper.

        # Train the model
        model.compile(optimizer='Adam', loss='mse', metrics=['mse'])
        model.fit(X_tra, X_tra, batch_size=256, epochs=1, verbose=0)
        predictions = model.predict(X_tst)

        # Compute average MSE
        avg_mse_per_instance = [mean_squared_error(true, pred) for true, pred in zip(X_tst, predictions)]
        avg_mse = np.mean(avg_mse_per_instance)
        print("Average MSE: ", avg_mse)
        assert 1 >= avg_mse

        # save weights
        model.save_weights('HouBoroui_test.h5')

        # re-load de model
        new_model, encoder = TSFEDL.HouBoroui(input_tensor=input, weights='HouBoroui_test.h5')
        new_predictions = new_model.predict(X_tst)
        os.remove('HouBoroui_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_ZhengZhenyu(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=128)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.ZhengZhenyu(input_tensor=input, include_top=True)
        model.summary()

        # assert that output shapes are the same that in Table 2 of the paper.
        model_output_shapes = [l.output_shape for l in model.layers
                               if not isinstance(l, keras.layers.BatchNormalization)
                               if not isinstance(l, keras.layers.Dropout)]
        desired_output_shape = [[(None, 128, 1)],
                                (None, 128, 64),  (None, 128, 64), (None, 64, 64),
                                (None, 64, 128), (None, 64, 128), (None, 32, 128),
                                (None, 32, 256), (None, 32, 256), (None, 16, 256),
                                (None, 256),
                                (None, 2048),
                                (None, 2048),
                                (None, 5)]
        self.assertTrue(model_output_shapes == desired_output_shape,
                        msg="Output shapes do not match. Model shapes: " + str(model_output_shapes))
        # Train the model
        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra,
                                           X_tst, y_tst, y_hc_tst, batch_size=256, epochs=1)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('ZhengZhenyu_test.h5')

        # re-load de model
        new_model = TSFEDL.ZhengZhenyu(input_tensor=input, weights='ZhengZhenyu_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('ZhengZhenyu_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_KhanZulfiqar(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=100)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.KhanZulfiqar(input_tensor=input, include_top=True)
        model.summary()

        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra,
                                           X_tst, y_tst, y_hc_tst, batch_size=256, epochs=1)
        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('KhanZulfiqar_test.h5')

        # re-load de model
        new_model = TSFEDL.KhanZulfiqar(input_tensor=input, weights='KhanZulfiqar_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('KhanZulfiqar_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

    #@unittest.skip
    def test_OhShuLih(self):
        (X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = self.getData(size=1000)
        input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
        model = TSFEDL.OhShuLih(input_tensor=input, include_top=True)

        # assert that output shapes are the same that in Table 2 of the paper.
        model_output_shapes = [l.output_shape for l in model.layers
                               if not isinstance(l, keras.layers.ZeroPadding1D)]
        desired_output_shape = [[(None, 1000, 1)],
                                (None, 1019, 3),
                                (None, 509, 3),
                                (None, 518, 6),
                                (None, 259, 6),
                                (None, 263, 6),
                                (None, 131, 6),
                                (None, 20),
                                (None, 20),
                                (None, 20),
                                (None, 10),
                                (None, 5)]
        self.assertTrue(model_output_shapes == desired_output_shape,
                        msg="Output shapes do not match. Model shapes: " + str(model_output_shapes))

        acc, predictions = self.trainModel(model, X_tra, y_tra, y_hc_tra,
                                           X_tst, y_tst, y_hc_tst, batch_size=256, epochs=1)

        assert 1.0 >= acc >= 0

        # save weights
        model.save_weights('OhShuLi_test.h5')

        # re-load de model
        new_model = TSFEDL.OhShuLih(input_tensor=input, weights='OhShuLi_test.h5')
        new_predictions = np.argmax(new_model.predict(X_tst), axis=1)
        os.remove('OhShuLi_test.h5')
        return np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

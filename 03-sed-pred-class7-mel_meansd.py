"""サウンドイベント検出のための推論とメトリクス計算

このスクリプトは学習済みのViTモデルを使用してサウンドイベント検出の推論を行い、
性能評価のためのメトリクスを計算します。

前提条件:
1. feature_testdataset.ipynbでテストデータセットを準備済みであること

必要な環境:
- Python 3.8
- CUDA 11.0
- cuDNN 8.0.5.39

必要なパッケージ:
- tensorflow-gpu==2.4.1
- tensorflow-addons (最新版)
- numpy==1.16.4
- h5py==2.10.0
- librosa
- matplotlib
- pandas
- scikit-learn

Author:
Date: 2025
"""

from __future__ import print_function
from tensorflow.python.client import device_lib

###################
# 標準ライブラリ
###################
import os
import sys
import time

###################
# 科学計算/機械学習
###################
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix

###################
# 音声処理/可視化
###################
import matplotlib.pyplot as plt

###################
# 自作モジュール
###################
import utils
import metrics

###################
# システム設定
###################

# 再帰制限の緩和
sys.setrecursionlimit(10000)

# GPUの設定と確認
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Available devices:", device_lib.list_local_devices())

# TensorFlow/Keras設定
K.set_image_data_format('channels_last')  # チャンネルの順序を設定

# 描画バックエンド設定
plt.switch_backend('agg')  # GUIなしの環境でも動作するように設定


def preprocess_data(_X_test, _Y_test, _seq_len, _nb_ch):
    """テストデータの前処理を行う

    Args:
        _X_test: テスト用の特徴量データ
        _Y_test: テスト用のラベルデータ
        _seq_len: シーケンス長
        _nb_ch: チャンネル数

    Returns:
        tuple: 処理済みの (_X_test, _Y_test)
    """
    # シーケンスに分割
    _X_test = utils.split_in_seqs(_X_test, _seq_len)
    _Y_test = utils.split_in_seqs(_Y_test, _seq_len)

    # マルチチャンネル処理
    _X_test = utils.split_multi_channels(_X_test, _nb_ch)

    return _X_test, _Y_test


def time_output(pred_thresh_, model_filename_, eval_f_, fold_):
    """クラスごとの開始時間と終了時間を抽出する

    Args:
        pred_thresh_: 予測閾値後の結果
        model_filename_: モデルのファイル名
        eval_f_: 評価用ファイル名

    Returns:
        None (結果をファイルに保存)
    """
    # shape: [全フレーム, クラス]に整形
    print("pred_thresh.shape", pred_thresh_.shape)
    for n, stack in enumerate(pred_thresh_):
        if n == 0:
            pred_vstack = stack
        if n != 0:
            pred_vstack = np.vstack([pred_vstack, stack])
    pred_vstack.shape

    # True, Falseの変化点から開始・終了インデックスを取得し、時間(秒)を算出

    data = pd.DataFrame(index=[], columns=['start', 'end', 'label'])

    start = 0
    end = 0
    buff = np.zeros(3)
    flag_s = 0
    flag_e = 0

    for m in range(pred_vstack.shape[1]):  # クラス
        for n in range(pred_vstack.shape[0]):  # frame
            if n == 0:  # 最初からTrueを開始時間に格納
                if pred_vstack[n][m]:  # 開始時間
                    idx_s = n
                    start = idx_s / (sr/(nfft/2.0))
                    buff[0] = np.round(start, decimals=7)

                    flag_s = 1  # startカラムに代入したことのフラグ

            elif n > 0:
                if (not pred_vstack[n-1][m]) and pred_vstack[n][m]:  # 開始時間
                    idx_s = n
                    start = idx_s / (sr/(nfft/2.0))
                    buff[0] = np.round(start, decimals=7)

                    flag_s = 1  # startカラムに代入したことのフラグ

                elif pred_vstack[n-1][m] and (not pred_vstack[n][m]):  # 終了時間
                    idx_e = n
                    end = idx_e / (sr/(nfft/2.0))

                    buff[1] = np.round(end, decimals=7)
                    buff[2] = m

                    flag_e = 1  # end, labelカラムに代入したことのフラグ

                else:
                    pass

            if (flag_s == 1) and (flag_e == 1):
                d3 = pd.Series(buff, index=data.columns)
                data = pd.concat([data, d3.to_frame().T], ignore_index=True)
                flag_s = 0
                flag_e = 0

    # ラベル列を文字列に置換
    __class_labels = {
        0: 'OT',
        1: 'APPLIX',
        2: 'TE',
        3: 'THOPAZ',
        4: 'SCD',
        5: 'PVM-high',
        6: 'C1-mid'
    }

    # __class_labels = {
    #     0:'OT',
    #     1:'CUF',
    #     2:'TE',
    #     3:'SAV'
    # }

    data['label'] = data['label'].replace(__class_labels)
    #         print(pred_thresh[0][n][m], Y_test[0][n][m])

    # TSV形式で出力
    filename = f'./result_fold/result_{model_filename_}_{eval_f_}_fold{fold_}.txt'
    data.to_csv(filename, sep='\t', header=False,
                index=False, index_label=False)
    print("Save Result: ")

    # data

#######################################################################################
# MAIN SCRIPT STARTS HERE
#######################################################################################


is_mono = True  # True: mono-channel input, False: binaural input

feat_folder = 'feat/'
__fig_name = '{}_{}'.format(
    'mon' if is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))


nb_ch = 1 if is_mono else 2
batch_size = 64    # Decrease this if you want to run on smaller GPU's
seq_len = 256       # Frame sequence length. Input to the CRNN.
nb_epoch = 500      # Training epochs
patience = int(0.25 * nb_epoch)  # Patience for early stopping

# Number of frames in 1 second, required to calculate F and ER for 1 sec segments.
# Make sure the nfft and sr are the same as in feature.py
sr = 48000
nfft = 1024
frames_1_sec = int(sr/(nfft/2.0))

print('\n\nUNIQUE ID: {}'.format(__fig_name))
print('TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
    nb_ch, seq_len, batch_size, nb_epoch, frames_1_sec))

# Folder for saving model and training curves
__models_dir = 'models/'
utils.create_folder(__models_dir)

# CRNN model definition
cnn_nb_filt = 128            # CNN filter size
# Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
cnn_pool_size = [5, 2, 2]
# Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
rnn_nb = [32, 32]
# Number of FC nodes.  Length of fc_nb =  number of FC layers
fc_nb = [32]
dropout_rate = 0.5          # Dropout after each layer
print('MODEL PARAMETERS:\n cnn_nb_filt: {}, cnn_pool_size: {}, rnn_nb: {}, fc_nb: {}, dropout_rate: {}'.format(
    cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate))


eval_file = [
    "XY_test_sn30",
    "XY_test_sn20",
    "XY_test_sn10",
    "XY_test_sn0"

]

model_filename_list = [

    "mbf40_cnn-f128-[5, 2, 2]_SimpleRNN[32, 32]_fc[32]_spec10_e150p37_batch480_2022_05_02_00_15_07",
    "mbf40_cnn-f128-[5, 2, 2]_bigru[32, 32]_fc[32]_spec10_e150p37_batch480_2022_05_01_08_38_46",
    "mbf40_cnn-f128-[2, 2, 2, 2]_bigru[32, 32]_fc[32]_spec10_e150p37_batch480_2022_04_30_21_48_01",
    "mbf40_cnn-f128-s1x2-[2, 2, 2, 2]_bigru[32, 32]_fc[32]_spec10_e150p37_batch480_2022_05_01_12_53_46",

]

posterior_thresh = 0.5

for model_filename in model_filename_list:
    for eval_f in eval_file:
        print(eval_f)
        avg_er = list()
        avg_f1 = list()

        feat_file_fold = os.path.join(
            feat_folder, eval_f + '_{}.npz'.format('mon' if is_mono else 'bin'))
        dmp = np.load(feat_file_fold)
        X_test, Y_test = dmp['arr_0'],  dmp['arr_1']

    #     X_test, Y_test = load_data(feat_folder, is_mono)
        print("load.data_X_test:", X_test.shape)
        X_test, Y_test = preprocess_data(X_test, Y_test, seq_len, nb_ch)
        X_test = X_test.transpose(0, 2, 3, 1)  # (time, mel, ch)

        for fold in [1, 2, 3, 4, 5]:
            print('\n----------------------------------------------')
            print('FOLD: {}'.format(fold))
            print('----------------------------------------------\n')

            K.clear_session()  # モデルを初期化してメモリーリセット

            model = keras.models.load_model(
                os.path.join(
                    __models_dir, f'{model_filename}_fold{fold}_model.h5')
            )

            tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list = [
                0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch, [0] * nb_epoch
            # Calculate the predictions on test data, in order to calculate ER and F scores
            pred = model.predict(X_test)

            # 각 fold의 결과를 파일로 출력
            pred_thresh_fold = pred > posterior_thresh
            time_output(utils.reshape_3Dto2D(
                pred_thresh_fold), model_filename, eval_f, fold)

            if fold == 1:
                pred_all = np.array([pred])
            else:
                pred_all = np.concatenate([pred_all, [pred]], 0)
            print("pred_all", pred_all.shape)

        pred_mean = np.mean(pred_all, axis=0)
        print("pred_mean", pred_mean.shape)

        pred_thresh = pred_mean > posterior_thresh
        print(pred_thresh.shape)
        score_list = metrics.compute_scores(
            pred_thresh, Y_test, frames_in_1_sec=frames_1_sec)
        print("score_list:", score_list)

        avg_er.append(score_list['er_overall_1sec'])
        avg_f1.append(score_list['f1_overall_1sec'])

        # Calculate confusion matrix
        test_pred_cnt = np.sum(pred_thresh, 2)
    #     print("test_pred_cnt:", test_pred_cnt.shape, "\n", test_pred_cnt)
        Y_test_cnt = np.sum(Y_test, 2)
    #     print("Y_test_cnt", Y_test_cnt.shape, "\n", Y_test_cnt)
        conf_mat = confusion_matrix(
            Y_test_cnt.reshape(-1), test_pred_cnt.reshape(-1))
        print("conf_mat\n", conf_mat)
    #     conf_mat = conf_mat / (utils.eps + np.sum(conf_mat, 1)[:, None].astype('float'))
    #     print("norm conf_mat\n",conf_mat)

        print(f'\n\nEvaluation dataset: {eval_f}')
    #     print('METRICS FOR ALL 5 FOLDS: avg_er: {}, avg_f1: {}'.format(avg_er, avg_f1))
        print('MODEL AVERAGE OVER 5 FOLDS: avg_er: {}, avg_f1: {}'.format(
            np.mean(avg_er), np.mean(avg_f1)))

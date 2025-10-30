"""
Sound Event Detection Metrics

Code blocks taken from Toni Heittola's repository: http://tut-arg.github.io/sed_eval/

Implementation of the Metrics in the following paper:
Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, 'Metrics for polyphonic sound event detection',
Applied Sciences, 6(6):162, 2016
"""

import numpy as np
import utils


def f1_overall_framewise(O, T):
    """Calculate frame-wise F1 score"""
    if len(O.shape) == 3:
        O, T = utils.reshape_3Dto2D(O), utils.reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum()
    Nref, Nsys = T.sum(), O.sum()

    prec = float(TP) / float(Nsys + utils.eps)
    recall = float(TP) / float(Nref + utils.eps)
    f1_score = 2 * prec * recall / (prec + recall + utils.eps)
    return f1_score


def er_overall_framewise(O, T):
    """Calculate frame-wise Error Rate"""
    if len(O.shape) == 3:
        O, T = utils.reshape_3Dto2D(O), utils.reshape_3Dto2D(T)
    FP = np.logical_and(T == 0, O == 1).sum(1)
    FN = np.logical_and(T == 1, O == 0).sum(1)

    S = np.minimum(FP, FN).sum()
    D = np.maximum(0, FN-FP).sum()
    I = np.maximum(0, FP-FN).sum()

    Nref = T.sum()
    ER = (S+D+I) / (Nref + 0.0)
    return ER


def f1_overall_1sec(O, T, block_size):
    """Calculate 1-second segment-based F1 score"""
    if len(O.shape) == 3:
        O, T = utils.reshape_3Dto2D(O), utils.reshape_3Dto2D(T)
    new_size = int(np.ceil(O.shape[0] / block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        start_idx = int(i * block_size)
        end_idx = int(i * block_size + block_size - 1)
        O_block[i, :] = np.max(O[start_idx:end_idx, ], axis=0)
        T_block[i, :] = np.max(T[start_idx:end_idx, ], axis=0)
    return f1_overall_framewise(O_block, T_block)


def er_overall_1sec(O, T, block_size):
    """Calculate 1-second segment-based Error Rate"""
    if len(O.shape) == 3:
        O, T = utils.reshape_3Dto2D(O), utils.reshape_3Dto2D(T)
    new_size = int(O.shape[0] / block_size)
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        start_idx = int(i * block_size)
        end_idx = int(i * block_size + block_size - 1)
        O_block[i, :] = np.max(O[start_idx:end_idx, ], axis=0)
        T_block[i, :] = np.max(T[start_idx:end_idx, ], axis=0)
    return er_overall_framewise(O_block, T_block)


def compute_scores(pred, y, frames_in_1_sec=50):
    """Compute F1 and Error Rate scores"""
    scores = dict()
    scores['f1_overall_1sec'] = f1_overall_1sec(pred, y, frames_in_1_sec)
    scores['er_overall_1sec'] = er_overall_1sec(pred, y, frames_in_1_sec)
    return scores

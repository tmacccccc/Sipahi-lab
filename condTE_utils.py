import numpy as np
import pandas as pd
import os
import sys
from math import frexp, ldexp
import math
import time


class DimensionError(Exception):
    def __init__(self, msg):
        super(DimensionError, self).__init__(msg)

        
class CondTEUtils(object):
    
    @classmethod
    def cal_prob(cls, ndarr:np.array) -> dict:
 
        sample_space = cls.__get_sample_space(ndarr) 
        str_arr = np.array([''.join(row.astype(str)) for row in ndarr])
        prob = {sample:(str_arr == sample).sum() / ndarr.shape[0] for sample in sample_space}
        return prob

    @classmethod
    def cal_entropy(cls, prob:dict, epsilon: int=0.0001) -> int:
        h = -np.sum([np.log2(v) * v if v > epsilon else 0 for v in prob.values()])
        return h
    
    @classmethod
    def cal_mutual_information(
        cls, 
        xtarget:np.ndarray,
        xsource:np.ndarray
    ) -> int:
        T = len(xsource)
        # epsilon = 10^(-MantissaExponent[T^4][[2]]) in mathematica
        epsilon = 10 ** (-(np.log10(T**4) + 1))
        
        # Xtarget situation
        Pxtarget = cls.cal_prob(xtarget[1:])
        hxtarget = cls.cal_entropy(Pxtarget, epsilon)
        
        # Xsource situation
        Pxsource = cls.cal_prob(xsource[:-1])
        hxsource = cls.cal_entropy(Pxsource, epsilon)
        
        # XX situation
        XXt = np.stack([xtarget[1:], xsource[:-1]], axis=1)
        Pxx = cls.cal_prob(XXt)
        hxx = cls.cal_entropy(Pxx, epsilon)
                      
        return hxtarget + hxsource - hxx               
        

    @classmethod
    def cal_transfer_entropy(
        cls, 
        xtarget:np.ndarray,
        xsource:np.ndarray,
        xcond:np.ndarray
    ) -> int:
        T = len(xsource)
        epsilon = 10 ** (-(np.log10(T**4) + 1))

        # XX situation
        XXt = np.stack([xtarget, xcond], axis=1)[:-1,:]
        Pxx = cls.cal_prob(XXt)
        hxx = cls.cal_entropy(Pxx, epsilon)

        # XXX situation
        XXXt = np.stack([xtarget, xsource, xcond], axis=1)[:-1,:]
        Pxxx = cls.cal_prob(XXXt)
        hxxx = cls.cal_entropy(Pxxx, epsilon)

        # XpXX situation
        XpXXt = np.stack([xtarget[1:], xtarget[:-1], xcond[:-1]], axis=1)
        Pxpxx = cls.cal_prob(XpXXt)
        hxpxx = cls.cal_entropy(Pxpxx, epsilon)

        # XpXXX situation
        XpXXXt = np.stack([xtarget[1:], xtarget[:-1], xsource[:-1], xcond[:-1]], axis=1)
        Pxpxxx = cls.cal_prob(XpXXXt)
        hxpxxx = cls.cal_entropy(Pxpxxx, epsilon)

        return hxpxx + hxxx - hxx - hxpxxx
    
    
    @classmethod
    def shuffle(cls, xsource: np.ndarray, others: list, rds: np.random.RandomState) -> tuple:
        
        ndarr = np.stack(others, axis=1)
        sample_space = cls.__get_sample_space(ndarr)
        str_arr = np.array([''.join(row.astype(str)) for row in ndarr])
        positions = [str_arr == sample for sample in sample_space]

        xsource_shuffle = np.zeros(xsource.shape)

        for p in positions:
            xsource_shuffle[p] = rds.choice(xsource[p], p.sum())
        xsource_shuffle = xsource_shuffle.astype(int)

        return xsource_shuffle, others
    
    @classmethod
    def surrogate_cond_transfer_entropy(
        cls, 
        xtarget: np.ndarray,
        xsource: np.ndarray,
        xcond: np.ndarray, 
        ns: int, 
        seed: int = 233
    ) -> np.ndarray:
        """
        input:
        ns: number of surrogates
        """
        rds = np.random.RandomState(seed)
        aux = np.zeros(ns)
        for i in range(ns):
            # xt_shuffle = cls.shuffle((xtarget, xsource, xcond), rds)
            xsource_shuffle, others = cls.shuffle(xsource, [xtarget, xcond], rds)
            # aux[i] = cls.cal_transfer_entropy(*xt_shuffle)
            aux[i] = cls.cal_transfer_entropy(xtarget, xsource_shuffle, xcond)
        return aux
    
    @classmethod
    def surrogate_mutual_information(
        cls, 
        xtarget: np.ndarray,
        xsource: np.ndarray,
        ns: int, 
        seed: int = 233
    ) -> np.ndarray:
        """
        input:
        ns: number of surrogates
        """
        rds = np.random.RandomState(seed)
        aux = np.zeros(ns)
        for i in range(ns):
            xsource_shuffle, others = cls.shuffle(xsource, [xtarget], rds)
            aux[i] = cls.cal_mutual_information(xtarget, xsource_shuffle)
        return aux
    
    @classmethod
    def __get_sample_space(cls, ndarr:np.array) -> list:
        if len(ndarr.shape) == 2:
            row_num, col_num = ndarr.shape
        elif len(ndarr.shape) == 1:
            row_num, col_num = ndarr.shape[0], 1
        else:
            msg = "Only 1-d or 2-d array is supported, {}-d array is given.".format(len(ndarr.shape))
            raise DimensionError(msg)
        sample_space = [('{:0' + str(col_num) + 'b}').format(i) for i in range(2**col_num)] 
        return sample_space

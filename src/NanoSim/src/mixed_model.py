#!/usr/bin/env python

"""
Created on May 17th, 2018 by Chen Yang
This script defines Poisson-Geometric distribution and Weibull-Geometric distribution
"""

import numpy as np
from math import ceil, exp
from scipy.stats import rv_discrete, poisson, geom, lognorm


# Scipy geometric starts with x = 1

class poisgeom_gen(rv_discrete):
    # Poisson-Geometric distribution
    def _pmf(self, x, l, p, w):
        return w * poisson.pmf(x, l) + (1 - w) * geom.pmf(x, p, loc=-1)


class weigeom_gen(rv_discrete):
    # Weibull-Geometric distribution, Geometric start from 0
    def _cdf(self, x, l, k, p, w):
        wei_cdf = 1 - np.exp(-1 * np.power(x / l, k))
        return w * wei_cdf + (1 - w) * geom.cdf(x, p, loc=-1)

    def _pmf(self, x, l, k, p, w):
        return self.cdf(x, l, k, p, w) - self.cdf(x-1, l, k, p, w)


class weigeom2_gen(rv_discrete):
    # Weibull-Geometric distribution, Geometric start from 1
    def _cdf(self, x, l, k, p, w):
        wei_cdf = 1 - np.exp(-1 * np.power(x / l, k))
        return w * wei_cdf + (1 - w) * geom.cdf(x, p)

    def _pmf(self, x, l, k, p, w):
        return self.cdf(x, l, k, p, w) - self.cdf(x-1, l, k, p, w)


class trunc_lognorm_gen(rv_discrete):
    def _cdf(self, x, s, m):
        lncdf_x = lognorm.cdf(x, s, scale=m)
        lncdf_a = lognorm.cdf(self.a, s, scale=m)
        lncdf_b = lognorm.cdf(self.b, s, scale=m)
        return (lncdf_x - lncdf_a) / (lncdf_b - lncdf_a)

    def _ppf(self, q, s, m):
        lncdf_a = lognorm.cdf(self.a, s, scale=m)
        lncdf_b = lognorm.cdf(self.b, s, scale=m)
        ln_q = q * (lncdf_b - lncdf_a) + lncdf_a
        return lognorm.ppf(ln_q, s, scale=m)


def pois_geom(lam, prob, weight):
    # Draw a random number from Poisson-Geometric distribution
    # Faster to use numpy random than using Scipy rvs
    tmp_rand = np.random.random()
    if tmp_rand < weight:
        value = np.random.poisson(lam) + 1
    else:
        value = np.random.geometric(prob)
    return value


def wei_geom(lam, k, prob, weight):
    # Draw a random number from Weibull-Geometric distribution
    tmp_rand = np.random.random()
    if tmp_rand < weight:
        value = int(round(ceil(lam * np.random.weibull(k))))
    else:
        value = np.random.geometric(prob) - 1

    if value == 0:
        value = 1

    return value


PHRED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

ONT_R9_PHRED_PROBABILITIES = {
    "match": [],
    "mis": [],
    "isn": [],
    "ht": []
}

ONT_R10_PHRED_PROBABILITIES = {
    "match": [0.0, 0.0, 0.002100630189056717, 0.008302490747224167, 0.012403721116334901, 0.01470441132339702, 0.015104531359407824, 0.014404321296388918, 0.013303991197359208, 0.012403721116334901, 0.011803541062318696, 0.011503451035310594, 0.011403421026307894, 0.011603481044313295, 0.011903571071321398, 0.012403721116334901, 0.013103931179353807, 0.013904171251375413, 0.01470441132339702, 0.015504651395418626, 0.016204861458437532, 0.016905071521456438, 0.017605281584475344, 0.01840552165649695, 0.01930579173752126, 0.020306091827548264, 0.021606481944583377, 0.02320696208862659, 0.025007502250675207, 0.02720816244873462, 0.029608882664799444, 0.03230969290787237, 0.03551065319595879, 0.03901170351105332, 0.04311293388016405, 0.04741422426728019, 0.05101530459137742, 0.052315694708412526, 0.050615184555366614, 0.04551365409622887, 0.0377113133940182, 0.030109032709812945, 0.022906872061618486, 0.01700510153045914, 0.012403721116334901, 0.00850255076522957, 0.006201860558167451, 0.005401620486145844, 0.004001200360108033, 0.0030009002700810247],
    "mis": [0.0, 0.0, 0.0384345911320188, 0.11120008007206483, 0.13622260034030623, 0.13121809628665795, 0.11079971974777297, 0.08757882093884492, 0.06625963367030324, 0.05014513061755578, 0.03833450105094584, 0.029926934240816726, 0.023721349214292854, 0.018917025322790507, 0.015313782404163742, 0.01261135021519367, 0.010509458512661392, 0.008807927134420976, 0.007606846161545389, 0.006605945350815732, 0.005805224702232006, 0.005204684215794213, 0.004604143729356419, 0.004303873486137522, 0.004003603242918626, 0.0038034230807726943, 0.0037033329996997285, 0.0036032429186267628, 0.0036032429186267628, 0.0037033329996997285, 0.0038034230807726943, 0.0039035131618456597, 0.004003603242918626, 0.0042037834050645565, 0.004303873486137522, 0.004504053648283453, 0.004604143729356419, 0.004704233810429385, 0.004403963567210488, 0.0038034230807726943, 0.003002702432188969, 0.0023020718646782094, 0.0017015313782404158, 0.0013011710539485531, 0.0009008107296566907, 0.0006005404864377938, 0.0005004504053648282, 0.00040036032429186256, 0.0003002702432188969, 0.00020018016214593128],
    "isn": [0.0, 0.0039023414048429063, 0.027616569941965183, 0.07024214528717232, 0.08675205123073845, 0.08755253151891136, 0.07834700820492296, 0.06613968381028619, 0.05463277966780069, 0.04532719631779068, 0.03812287372423455, 0.0324194516710026, 0.028016810086051638, 0.02441464878927357, 0.021612967780668405, 0.019011406844106467, 0.01701020612367421, 0.0154092455473284, 0.014208525115069044, 0.013307984790874526, 0.012507504502701624, 0.011807084250550331, 0.011206724034420653, 0.010806483890334203, 0.010506303782269364, 0.010306183710226137, 0.010106063638182911, 0.0100060036021613, 0.009905943566139686, 0.0100060036021613, 0.010106063638182911, 0.010306183710226137, 0.010606363818290976, 0.010906543926355814, 0.01140684410646388, 0.012007204322593558, 0.012607564538723236, 0.012707624574744848, 0.01210726435861517, 0.010906543926355814, 0.009005403241945167, 0.007204322593556135, 0.005503301981188714, 0.004002401440864519, 0.0029017410446267764, 0.0021012607564538724, 0.0015009005403241948, 0.0013007804682809688, 0.0009005403241945168, 0.0007004202521512908],
    "ht": [0.0, 0.0046990601879624075, 0.016396720655868828, 0.03349330133973206, 0.03779244151169766, 0.03649270145970806, 0.03249350129974005, 0.027994401119776045, 0.02389522095580884, 0.020895820835832832, 0.01889622075584883, 0.01759648070385923, 0.016696660667866427, 0.016096780643871225, 0.015896820635872826, 0.015796840631873626, 0.015996800639872025, 0.016196760647870425, 0.016496700659868028, 0.01699660067986403, 0.017396520695860826, 0.01789642071585683, 0.01839632073585283, 0.01889622075584883, 0.019596080783843232, 0.02029594081183763, 0.021195760847830435, 0.022295540891821636, 0.023395320935812838, 0.02469506098780244, 0.025994801039792043, 0.027594481103779243, 0.029194161167766446, 0.03089382123575285, 0.03259348130373925, 0.03399320135972806, 0.035092981403719255, 0.034893021395720855, 0.032793441311737656, 0.028494301139772048, 0.022495500899820036, 0.017296540691861626, 0.012897420515896822, 0.009398120375924815, 0.006798640271945611, 0.004599080183963208, 0.0033993201359728054, 0.0028994201159768043, 0.0021995600879824036, 0.0015996800639872027]
}


def trunc_lognorm_rvs(error_type, read_type, basecaller, n):
    if basecaller == "albacore":
        if error_type not in ONT_R9_PHRED_PROBABILITIES:
            probs = ONT_R9_PHRED_PROBABILITIES["ht"]
        else:
            probs = ONT_R9_PHRED_PROBABILITIES[error_type]
    else:
        if error_type not in ONT_R10_PHRED_PROBABILITIES:
            probs = ONT_R10_PHRED_PROBABILITIES["ht"]
        else:
            probs = ONT_R10_PHRED_PROBABILITIES[error_type]

    return list(np.random.choice(PHRED, size=n, p=probs))


def trunc_lognorm_rvs_old(error_type, read_type, basecaller, n):
    if basecaller == "albacore":
        a = 1
        b = 28
        if read_type == "DNA":
            if error_type == "match":
                mean = 2.7418286
                sd = 0.7578693
            elif error_type == "mis":
                mean = 1.6597215
                sd = 0.6814804
            elif error_type == "ins":
                mean = 1.9016147
                sd = 0.6842999
            elif error_type == "ht":
                mean = 2.3739153
                sd = 0.9635895
            else:  # unaligned
                mean = 2.5484921
                sd = 0.7742894
        elif read_type == "dRNA":
            if error_type == "match":
                mean = 2.236641
                sd = 0.434045
            elif error_type == "mis":
                mean = 1.8138169
                sd = 0.4535039
            elif error_type == "ins":
                mean = 1.9322685
                sd = 0.4668444
            elif error_type == "ht":
                mean = 2.0166876
                sd = 0.5714308
            else:  # unaligned
                mean = 2.1371272
                sd = 0.4763441
        else:  # cDNA
            if error_type == "match":
                mean = 2.6003978
                sd = 0.7181057
            elif error_type == "mis":
                mean = 1.6380338
                sd = 0.6695235
            elif error_type == "ins":
                mean = 1.8462438
                sd = 0.6661691
            elif error_type == "ht":
                mean = 2.510699
                sd = 1.082626
            else:  # unaligned
                mean = 2.6004634
                sd = 0.8526468
    elif basecaller == "guppy":
        a = 1
        b = 31
        if read_type == "DNA":
            if error_type == "match":
                mean = 2.9863022
                sd = 0.9493498
            elif error_type == "mis":
                mean = 1.6184245
                sd = 0.7585733
            elif error_type == "ins":
                mean = 1.8852560
                sd = 0.7623103
            elif error_type == "ht":
                mean = 1.995397
                sd = 1.008650
            else:  # unaligned
                mean = 1.2626728
                sd = 0.9012829
        elif read_type == "dRNA":
            if error_type == "match":
                mean = 2.236641
                sd = 0.434045
            elif error_type == "mis":
                mean = 1.8138169
                sd = 0.4535039
            elif error_type == "ins":
                mean = 1.9322685
                sd = 0.4668444
            elif error_type == "ht":
                mean = 2.0166876
                sd = 0.5714308
            else:  # unaligned
                mean = 2.1371272
                sd = 0.4763441
        else:  # cDNA
            if error_type == "match":
                mean = 2.7500148
                sd = 0.9195383
            elif error_type == "mis":
                mean = 1.5543628
                sd = 0.7601223
            elif error_type == "ins":
                mean = 1.765634
                sd = 0.777587
            elif error_type == "ht":
                mean = 2.001173
                sd = 1.008647
            else:  # unaligned
                mean = 1.2635415
                sd = 0.9008419

    m = exp(mean)
    truncln_gen = trunc_lognorm_gen(name="truncln", a=a, b=b, shapes="s, m")
    return truncln_gen.rvs(s=sd, m=m, size=n)

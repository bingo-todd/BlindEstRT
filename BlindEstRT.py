import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy.signal
import scipy.optimize
import librosa


from BasicTools.easy_parallel import easy_parallel

_EPS = 1e-20


def add_log(logger, txt):
    if logger is not None:
        logger.write(txt)
        logger.write('\n')
        logger.flush()


def lowpass_filter(x, cutfreq, order=4):
    b, a = scipy.signal.butter(order, cutfreq)
    x_filtered = scipy.signal.lfilter(b, a, x)
    return x_filtered


def bandpass_filter(x, bassband, order=5):
    """ butterworth filter
    """
    b, a = scipy.signal.butter(order, bassband, btype='bandpass')
    x_filtered = scipy.signal.lfilter(b, a, x)
    return x_filtered


def get_envelope(x, cutfreq):
    """
    Args:
        x:
        cutfreq: normalized cut-off frequency
    """
    # first hilbert transform
    # dsp.hilbert return analytic signal
    envelope = np.abs(scipy.signal.hilbert(x))
    envelope_filtered = lowpass_filter(envelope, cutfreq)
    envelope_final = np.abs(envelope_filtered)
    return envelope_final


def Choose_signal(env, fs, n_seg_required, max_rt, min_rt,
                  frame_len, frame_shift, is_plot=False):
    """find exponential damping continuous segments
    """
    # transform envelope into log(e as base) scale
    env = np.log(env)
    env_len = env.shape[0]
    # split envelope into frames
    n_frame = np.int(np.floor(env_len-frame_len)/frame_shift) + 1
    a_all = np.zeros(n_frame, dtype=np.float32)

    # constants
    poly_x = np.arange(frame_len)
    for frame_i in range(n_frame):
        frame_start = frame_i * frame_shift
        frame_end = frame_start + frame_len
        frame = env[frame_start:frame_end]

        # 1-order poly fit
        coefs = np.polyfit(poly_x, frame, 1)
        a_all[frame_i] = coefs[0]

    max_a = -6.91/max_rt/fs
    min_a = -6.91/min_rt/fs
    # print(f'max_a: {max_a}  min_a: {min_a}')
    # find all continuous frames with a in the range [min_a, max_a]
    seg_pos_all = []
    seg_start = 0
    while seg_start < n_frame:
        if a_all[seg_start] >= min_a and a_all[seg_start] <= max_a:
            seg_end = seg_start
            while seg_end < n_frame:
                if a_all[seg_end] >= min_a and a_all[seg_end] <= max_a:
                    seg_end = seg_end + 1
                else:
                    seg_end = seg_end - 1  # go back 1 frame
                    break
            if seg_end == n_frame:
                seg_end = n_frame-1
            seg_pos_all.append([seg_start, seg_end])
            # prepare for the next segment
            seg_start = seg_end + 1
        else:
            seg_start = seg_start + 1

    seg_pos_all = np.asarray(seg_pos_all)

    if seg_pos_all.shape[0] < 1:
        return None

    # sort based on segment length
    sort_index = np.argsort(seg_pos_all[:, 1]-seg_pos_all[:, 0])
    n_seg = np.min((len(sort_index), n_seg_required))
    seg_pos_all = seg_pos_all[:n_seg]
    # transform seg_pos from frame-unit to sample-unit
    # star of segments
    seg_pos_all[:, 0] = seg_pos_all[:, 0]*frame_shift
    # end of segments
    seg_pos_all[:, 1] = seg_pos_all[:, 1]*frame_shift+frame_len

    if is_plot:
        fig, ax = plt.subplots(1, 1)
        ax.plot(env)
        for seg_i in range(n_seg):
            ax.plot(np.arange(seg_pos_all[seg_i, 0], seg_pos_all[seg_i, 1]),
                    env[seg_pos_all[seg_i, 0]: seg_pos_all[seg_i, 1]],
                    color='red')
        fig.savefig('images/env_choose_signal.png')
        plt.close(fig)
        # raise Exception()
    return seg_pos_all


def cal_lh(alpha, a1, a2, x):
    x_len = x.shape[0]
    n = np.arange(x_len)
    sigma_square = np.sum(x**2/(alpha*a1**n+(1-alpha)*a2**n+_EPS))
    env_est = alpha*a1**n + (1-alpha)*a2**2
    env_est[env_est <= 1e-20] = 1e-20

    # np.seterr(all='raise')
    with np.errstate(invalid='ignore'):
        lh = (np.sum((x**2)/(2*sigma_square*(env_est**2)))
              + x_len/2*np.log(2*np.pi*sigma_square)
              + np.sum(np.log(env_est)))
    return lh


def cal_lh_for_alpha(alpha, a1, a2, x):
    return cal_lh(alpha, a1, a2, x)


def cal_lh_for_allparams(param, x):
    alpha, a1, a2 = param
    return cal_lh(alpha, a1, a2, x)


def search_env_param(x, fs):
    """find a1, a2, alpha by maximize the likelihood function
    the sound decay is modeled as
    h[n] = env[n]g[n]
    where
    g[n] is gaussian noise
    env[n] = alpha*a1^n + (1-alpha)*a2^2  # two decay region, more realistic
    h[n] is the input, and parameters are alpha, a1, a2, since g[n] is a
    Gaussian noise, so g[n] follows the Gaussian distribution, as a result,
    parameters can be optimized by maxmize the likelihood

    """
    # coarse search
    # draw a grid over the possible range of al and a2, for each point on the
    # grid, the optimal alpha is estimated by mixmizing the likelihood function
    # function. The parameter set with the highest likelihood with the highest
    # likelihood is token as the final result of coarse search the range of a

    # the range of a is limited under sample frequency of 3kHz
    # TODO differ from matlab version
    a_range_3kHz = np.asarray([0.9, 0.9999])
    # corresponding rt range [0.02 - 230]
    rt_range = (-6.91/np.log(a_range_3kHz))/3000
    # then calculate the corrresponding range of a as sample rate if fs
    a_range = np.exp(-6.91/(fs*rt_range))
    n_step = 5  # 5 points are equally sampled from the a_range
    best_params = [np.infty, np.infty, np.infty]  # [a1, a2, alpha]
    min_lh = np.infty
    for a1 in np.linspace(a_range[0], a_range[1], n_step):
        for a2 in np.linspace(a_range[0], a_range[1], n_step):
            alpha_init = 0.5
            res = scipy.optimize.minimize(fun=cal_lh_for_alpha, x0=alpha_init,
                                          args=(a1, a2, x), bounds=((0, 1),))
            alpha = res.x
            lh = cal_lh(alpha, a1, a2, x)
            if lh < min_lh:
                best_params = [alpha, a1, a2]

    # fine search
    # chose the params in coarse searching with the largest lh as the start
    # point of fine tuning
    res = scipy.optimize.minimize(fun=cal_lh_for_allparams, x0=best_params,
                                  args=(x), bounds=(a_range, a_range, [0, 1]))
    best_params = res.x
    return best_params


def optimize_env_est(env_params, fs):
    """set the duration of decay phase to 4s and divides it into frames
    (half overlaoped). In each frames, calculated corresponding envelope of
    parameter set, chose the envelope with smallest energy as the final
    estimation. Finally, envelopes of each frame are windowed and added to get
    a optimized envelope
    """
    env_len_t = 4
    # frame_len_t=0.1 frame_shift_t=0.05 in original code, these two values
    # are decreased for finer result
    frame_len_t = 0.01
    frame_shift_t = 0.005

    env_len = int(env_len_t*fs)
    n = np.arange(env_len)
    frame_len = int(frame_len_t*fs)
    frame_shift = int(frame_shift_t*fs)
    n_frame = np.int(np.floor((env_len-frame_len)/frame_shift))+1

    env = []
    for param in env_params:
        alpha, a1, a2 = param
        env_param = alpha*a1**n+(1-alpha)*a2**n
        env_param = env_param/np.max(np.abs(env_param))
        inv_cumsum = np.flip(np.cumsum(np.flip(env_param**2)))
        inv_cumsum = 10*np.log10(inv_cumsum/np.max(inv_cumsum))
        if inv_cumsum[-1] > -25:
            # too small dynamic range
            continue
        else:
            env.append(env_param)
    env = np.asarray(env, dtype=np.float32)

    # find envelep with smallest energy for each frame
    env_final = np.zeros(env_len, dtype=np.float32)
    window_frame = np.hanning(frame_len)  # window function for the new frame
    # window function for the save result
    window_result = np.hanning(frame_len)
    window_result[frame_shift:] = 1
    window_result = np.abs(window_result-1)
    for frame_i in range(n_frame):
        frame_start = frame_i*frame_shift
        frame_end = frame_start+frame_len
        param_i = np.argmin(
            np.sum(env[:, frame_start:frame_end]**2,
                   axis=1))
        if frame_i == 0:
            env_final[frame_start:frame_end] = \
                env[param_i, frame_start:frame_end]
        else:
            env_final[frame_start:frame_end] = \
                ((env_final[frame_start:frame_end]*window_result)
                 + (env[param_i, frame_start:frame_end]*window_frame))

    with open('tmp.txt', 'w') as file_obj:
        file_obj.write(' '.join([str(item) for item in env_final]))

    return env_final


def MLE_decay_estimate(x, fs, logger):

    logger.write('MLE_decay_estimate\n')

    # get envelope of x, the envelope is low-pass filtered
    f_nyquist = fs/2
    cutfreq = 80/f_nyquist
    env = get_envelope(x, cutfreq)

    n_seg_required = 100000
    max_rt = 100
    min_rt = 0.005
    frame_len = np.int(0.5*fs)
    frame_shift = np.int(0.5*fs)
    seg_pos_all = Choose_signal(env, fs, n_seg_required, max_rt, min_rt,
                                frame_len, frame_shift)

    # fine tuning of the start and end of segments
    if seg_pos_all is None:
        return None

    n_seg = seg_pos_all.shape[0]
    # estimation result of all segments
    env_params = np.zeros((n_seg, 3), dtype=np.float32)
    half_frame_len = int(frame_len/2)
    for seg_i in range(n_seg):
        seg_start, seg_end = seg_pos_all[seg_i]
        # fine tuning the start and end position of segment
        # start pos
        tmp = np.argmax(env[seg_start:seg_start+half_frame_len])
        seg_start = seg_start + tmp
        # end pos
        tmp = np.argmin(env[seg_end-half_frame_len:seg_end])
        seg_end = seg_end - half_frame_len + tmp

        # seg_env = env[seg_start:seg_end]
        seg_x = x[seg_start: seg_end]
        seg_pos_all[seg_i] = [seg_start, seg_end]
        env_params[seg_i] = search_env_param(seg_x, fs)
        add_log(logger, f'seg {seg_i} from {seg_start} to {seg_end}')
        add_log(logger, f'{env_params[seg_i]}')
    return env_params


def cal_rt(env, fs):
    env = env/np.max(np.abs(env))
    inv_cumsum = np.flip(np.cumsum(np.flip(env**2)))
    inv_cumsum = 10*np.log10(inv_cumsum/np.max(inv_cumsum)+1e-20)
    # find the -5dB point
    _5dB_pos = np.argmin(np.abs(inv_cumsum+5))
    _35dB_pos = np.argmin(np.abs(inv_cumsum+35))
    coefs = np.polyfit(np.arange(_5dB_pos, _35dB_pos),
                       inv_cumsum[_5dB_pos:_35dB_pos], 1)
    rt = -60/coefs[0]
    if np.isnan(rt):
        inv_cumsum_polyfit = np.arange(_5dB_pos, _35dB_pos)*coefs[0]+coefs[1]
        fig, ax = plt.subplots(1, 1)
        ax.plot(inv_cumsum)
        ax.plot(np.arange(_5dB_pos, _35dB_pos), inv_cumsum_polyfit)
        fig.savefig('images/exception_cal_rt.png')
        plt.close(fig)
        raise Exception()
    rt = rt/fs
    return rt


def est_rt_seg(seg, fs, log_path):

    if log_path is None:
        logger = None
    else:
        logger = open(log_path, 'a')

    env_params = MLE_decay_estimate(seg, fs, logger)
    if env_params is None:
        rt = None
    else:
        # further optimize the envelope by merging env_params
        env = optimize_env_est(env_params, fs)
        rt = cal_rt(env, fs)
    add_log(logger, f'RT after envelope optimization: {rt}')
    return rt


def BlindEstRT(wav_path, n_worker, is_log=True):

    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)

    # constants
    cfs = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    fs_band_all = [3000, 3000, 3000, 3000, 3000, 6000, 12000, 24000]
    sqrt_2 = np.sqrt(2)

    # only accept 1-channel audio file
    wav, fs = librosa.load(wav_path, sr=None)
    if len(wav.shape) > 1:
        wav = wav[:, 0]

    # split wav into segments
    n_seg = 20  # TODO maybe fix the length of segments
    rt_est_all = []
    for band_i, cf in enumerate(cfs):
        #
        fs_band = fs_band_all[band_i]
        f_nyquist = fs_band/2
        cf_norm = cf/f_nyquist

        # resample to decrease caculation load
        wav_resampled = librosa.resample(wav, fs, fs_band)

        # band filter
        wav_filtered = bandpass_filter(wav_resampled,
                                       [cf_norm/sqrt_2, cf_norm*sqrt_2])
        del wav_resampled
        # np.save(f'{band_i}.npy', wav_filtered)
        # continue
        # wav_filtered = np.load(f'{band_i}.npy')

        seg_len = np.int(np.round(wav_filtered.shape[0]/n_seg))

        if True:
            tasks = []
            for seg_i in range(n_seg):
                seg_start = np.int(seg_i*seg_len)
                seg_end = seg_start+seg_len
                seg = wav_filtered[seg_start:seg_end]
                if is_log:
                    log_path = f'{log_dir}/log_{band_i}_{seg_i}'
                    with open(log_path, 'w') as logger:
                        add_log(logger, '{seg_start} - {seg_end}')
                else:
                    log_path = None
                tasks.append([seg, fs_band, log_path])
            del wav_filtered

            n_worker = np.min((len(tasks), n_worker))
            rt_est = easy_parallel(est_rt_seg, tasks, n_worker)
        else:
            rt_est = []
            for seg_i in range(n_seg):
                seg_start = np.int(seg_i*seg_len)
                seg_end = seg_start+seg_len
                seg = wav_filtered[seg_start:seg_end]
                if is_log:
                    log_path = f'{log_dir}/log_{band_i}_{seg_i}'
                    with open(log_path, 'w') as logger:
                        add_log(logger, '{seg_start} - {seg_end}')
                else:
                    log_path = None
                rt_tmp = est_rt_seg(seg, fs_band, log_path)
                rt_est.append(rt_tmp)

        # exclude None
        rt_est = [rt for rt in rt_est if rt is not None]
        rt_est_all.append(rt_est)
    rt_mean = [np.mean(item) for item in rt_est_all]
    rt_std = [np.std(item) for item in rt_est_all]

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(cfs, rt_mean, yerr=rt_std)
    fig.savefig('images/estimation_result.png')
    return rt_mean, rt_std


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--wav', dest='wav_path', required=True,
                        type=str, help='path of the input file')
    parser.add_argument('--n-worker', dest='n_worker', required=True,
                        type=int, help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    rt_mean, rt_std = BlindEstRT(args.wav_path, args.n_worker)
    print(rt_mean, rt_std)


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
from lmfit.models import ExpressionModel, Model
import matplotlib
from matplotlib import rc
from scipy import special
import json
from json import JSONEncoder
import matplotlib as mpl
from scipy import integrate
import os
import asyncio
import random
from joblib import Parallel, delayed

# mpl.use('Qt5Agg')

matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 18
# matplotlib.rcParams['legend.fontsize'] = 16
rc('font', **{'size': 24})

#### SET PROJECT PATH #####
gamma = 2.67513e2  # in [rad/(ms*mT)]

#### LOAD FITAID RESULTS ####
df = pd.read_excel("../data/dMRS_aging_fitResults.xlsx", sheet_name='Sheet3')
vol_names = df['Volunteers'].unique()
tissues = ['CEREBELLUM', 'PCC']
metabs = ["tNAA", "tCho", "tCr"]
df_keys = df.keys()
fit_results = \
    {
        'CEREBELLUM':
            {
                'tNAA': {},
                'tCho': {},
                'tCr': {}
            },
        'PCC':
            {
                'tNAA': {},
                'tCho': {},
                'tCr': {}
            }
    }

b_values = np.array([11, 1016, 4031.25, 9057.25, 16093.5, 25140]) * 1e-3

delta = 26.4  # in ms
Delta = 62.5  # in ms

gradients = np.sqrt(b_values / ((Delta - delta / 3) * gamma ** 2 * delta ** 2))

minimizer_method = 'least_squares'
mc_steps = 250
use_emcee = False


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


def astrosticks(b, D_f):
    return np.sqrt(np.pi) / 2 * special.erf(np.sqrt(b * D_f)) / np.sqrt(b * D_f)


def modified_astrosticks(x, b, D_f, K_intra):
    return np.exp(-b * D_f * x ** 2 + K_intra * (b * D_f) ** 2 * x ** 4)


def integrate_mAS(b, D_f, K_intra):
    return integrate.quad(modified_astrosticks, 0, 1,
                          args=(b, D_f, K_intra), epsabs=1.49e-14, epsrel=1.49e-14)[0]


def stick_kurtosis_model(b, D_f, K_intra):
    vect = np.vectorize(integrate_mAS)
    return vect(b, D_f, K_intra)


def m_astrosticks_model_init_fit(init_vals, data, b, fit_method):
    stick_kurt_model = Model(stick_kurtosis_model, independent_vars=['b'], name='Modifed Astro-Sticks')
    stick_kurt_params = stick_kurt_model.make_params()

    stick_kurt_params["D_f"].set(value=init_vals[0], min=0, max=1.5, vary=True)
    stick_kurt_params["K_intra"].set(value=init_vals[1], min=0.0, max=0.1, vary=True)
    fit_result = stick_kurt_model.fit(data=data,
                                      params=stick_kurt_params,
                                      b=b,
                                      # weights=1. / std_area,
                                      method=fit_method)
    return fit_result


def m_astrosticks_model_fit(n_trials, data, b, fit_method, n_cpus=16):
    init_Dfs = np.random.uniform(low=0.0, high=1.5, size=n_trials)
    init_Kapps = np.random.uniform(low=0.0, high=0.1, size=n_trials)

    temp_Dfs = []
    temp_Kapps = []
    chisqrs = []

    temp_results = Parallel(n_jobs=n_cpus)(
        delayed(m_astrosticks_model_init_fit)([random.choice(init_Dfs), random.choice(init_Kapps)],
                                          data,
                                          b,
                                          fit_method) for i in range(n_trials))

    for res in temp_results:
        chisqrs.append(res.chisqr)
        temp_Dfs.append(res.best_values['D_f'])
        temp_Kapps.append(res.best_values['K_intra'])

    chisqrs = np.array(chisqrs)
    temp_Dfs = np.array(temp_Dfs)
    temp_Kapps = np.array(temp_Kapps)

    temp_sort = np.argsort(chisqrs)
    chisqrs = chisqrs[temp_sort]
    temp_Dfs = temp_Dfs[temp_sort]
    temp_Kapps = temp_Kapps[temp_sort]

    init_Df = np.mean(temp_Dfs[:int(n_trials / 10)])
    init_Kapp = np.mean(temp_Kapps[:int(n_trials / 10)])

    final_fit = m_astrosticks_model_init_fit([init_Df, init_Kapp],
                                         data, b, fit_method)

    return final_fit


fitting_data = {}
adc_data = {}

global mono_exp_results
global bi_exp_results
global kurtosis_results
global roc_model

monoexp_model = ExpressionModel("exp(-x * adc)", name="Monoexponential")
monoexp_params = monoexp_model.make_params()

kurtosis_model = ExpressionModel("exp(-x * adc + (K * x ** 2 * adc ** 2) / 6 )", name="Kurtosis")
kurtosis_params = kurtosis_model.make_params()

stick_model = Model(astrosticks, independent_vars=['b'], name='Astro-Sticks')
stick_params = stick_model.make_params()

monoexp_params["adc"].set(value=0.1, min=0.0, vary=True)

kurtosis_params["adc"].set(value=0.1, min=0.0, vary=True)
kurtosis_params["K"].set(value=1, min=0.0, max=2.5, vary=True)

stick_params["D_f"].set(value=0.4, min=0, max=1.0, vary=True)


for i in range(1, len(df_keys)):
    norm_areas = np.exp(df[df_keys[i]].values.reshape(len(vol_names), len(b_values)))

    for j in range(len(vol_names)):
        if vol_names[j].__contains__('DEV_458-73y_onlyPCC') and df_keys[i].__contains__('CEREB'):
            continue

        me_adc = me_std_adc = me_chisqr = me_fit = 0
        k_adc = k_K = k_std_adc = k_std_K = k_chisqr = k_fit = 0
        s_Df = s_err_S0 = s_err_Df = s_chisqr = s_fit = 0
        sk_Df = sk_Deff = sk_K_intra = sk_err_Df = sk_err_Deff = sk_err_K_intra = sk_chisqr = sk_fit = 0
        print(f"+++++++++++{vol_names[j]}_{df_keys[i]}+++++++++++")
        try:
            mono_exp_results = monoexp_model.fit(data=norm_areas[j][0:3],
                                                 params=monoexp_params,
                                                 x=b_values[0:3],
                                                 method=minimizer_method)

            # print("----------MONO-EXPONENTIAL FIT----------")
            # print(mono_exp_results.fit_report())
            me_adc = mono_exp_results.best_values['adc']

            try:
                me_std_adc = np.sqrt(np.diag(mono_exp_results.covar))
            except AttributeError:
                pass

            me_chisqr = mono_exp_results.chisqr
            me_fit = mono_exp_results.best_fit.reshape(1, len(norm_areas[j][:3]))

        except ValueError:
            mono_exp_results = 0
            me_fit = np.zeros(len(norm_areas[j][:3])).reshape(1, len(norm_areas[j][:3]))
            pass
        try:
            kurtosis_results = kurtosis_model.fit(data=norm_areas[j][0:4],
                                                  params=kurtosis_params,
                                                  x=b_values[0:4],
                                                  method=minimizer_method)

            # print("----------KURTOSIS FIT----------")
            # print(kurtosis_results.fit_report())
            k_adc = kurtosis_results.best_values['adc']
            k_K = kurtosis_results.best_values['K']

            try:
                k_std_adc, k_std_K = np.sqrt(np.diag(kurtosis_results.covar))
            except AttributeError:
                pass

            k_chisqr = kurtosis_results.chisqr
            k_fit = kurtosis_results.best_fit.reshape(1, len(norm_areas[j][:4]))

        except ValueError:
            kurtosis_results = 0
            k_fit = np.zeros(len(norm_areas[j][:4])).reshape(1, len(norm_areas[j][:4]))
            pass

        try:
            stick_results = stick_model.fit(data=norm_areas[j],
                                            params=stick_params,
                                            b=b_values,
                                            # weights=1. / std_area,
                                            method=minimizer_method)

            print("---------- ASTRO-STICKS MODEL FIT----------")
            # print(stick_results.fit_report())
            s_Df = stick_results.best_values['D_f']

            try:
                s_err_Df = np.sqrt(stick_results.covar)
            except AttributeError:
                pass

            s_chisqr = stick_results.chisqr
            s_fit = stick_results.best_fit.reshape(1, len(norm_areas[j]))

        except ValueError:
            stick_results = 0
            s_fit = np.zeros(len(norm_areas[j])).reshape(1, len(norm_areas[j]))
            pass

        try:
            # stick_kurt_results = stick_kurt_model.fit(data=norm_areas[j],
            #                                             params=stick_kurt_params,
            #                                             b=b_values,
            #                                             method=minimizer_method)
            stick_kurt_results = m_astrosticks_model_fit(mc_steps, norm_areas[j], b_values, minimizer_method)
            print("---------- MODIFIED ASTRO-STICKS MODEL FIT----------")
            print(stick_kurt_results.fit_report())
            sk_Df = stick_kurt_results.best_values['D_f']
            sk_K_intra = stick_kurt_results.best_values['K_intra']

            sk_Deff = sk_Df * (1 - sk_K_intra * b_values * sk_Df)

            try:
                sk_err_Df, sk_err_K_intra = np.sqrt(np.diag(stick_kurt_results.covar))
            except AttributeError:
                pass
            sk_err_Deff = np.sqrt((1 - 2 * sk_K_intra * sk_Df) ** 2 * b_values * sk_err_Df ** 2 + (
                    b_values * sk_Df) ** 2 * sk_err_K_intra ** 2)
            sk_chisqr = stick_kurt_results.chisqr
            sk_fit = stick_kurt_results.best_fit.reshape(1, len(norm_areas[j]))

        except ValueError:
            stick_kurt_results = 0
            sk_fit = np.zeros(len(norm_areas[j])).reshape(1, len(norm_areas[j]))
            pass


        print("+++++++++++FINISHED!+++++++++++")
  
        if j == 0:
            fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]].update(
                {
                    'monoexp':
                        {
                            # 'S0': np.array([me_S0]),
                            'ADC': np.array([me_adc]),
                            # 'err_S0': np.array([me_std_S0]),
                            'err_ADC': np.array([me_std_adc]),
                            'chisqr': np.array([me_chisqr]),
                            'fit': me_fit,
                        },
                    'kurtosis':
                        {
                            # 'S0': np.array([k_S0]),
                            'ADC': np.array([k_adc]),
                            'K': np.array([k_K]),
                            # 'err_S0': np.array([k_std_S0]),
                            'err_ADC': np.array([k_std_adc]),
                            'err_K': np.array([k_std_K]),
                            'chisqr': np.array([k_chisqr]),
                            'fit': k_fit,
                        },
                    'astrosticks':
                        {
                            # 'S0': s_S0,
                            'D_f': np.array([s_Df]),
                            'err_S0': np.array([s_err_S0]),
                            'err_Df': np.array([s_err_Df]),
                            'chisqr': np.array([s_chisqr]),
                            'fit': s_fit,
                        },
                    'm_astrosticks':
                        {
                            # 'S0': ck_S0,
                            'D_f': np.array([sk_Df]),
                            'D_eff': np.array([sk_Deff]),
                            'K_intra': np.array([sk_K_intra]),
                            # 'err_S0': np.array([ck_err_S0]),
                            'err_Df': np.array([sk_err_Df]),
                            'err_Deff': np.array([sk_err_Deff]),
                            'err_K_intra': np.array([sk_err_K_intra]),
                            'chisqr': np.array([sk_chisqr]),
                            'fit': sk_fit,
                        },
                    'data':
                        {
                            'Area': norm_areas[j].reshape(1, len(norm_areas[j]))
                        }
                }
            )
        else:
            fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]].update(
                {
                    'monoexp':
                        {
                            # 'S0': np.append(
                            #     fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["monoexp"]["S0"],
                            #     me_S0),
                            'ADC': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["monoexp"]["ADC"],
                                me_adc),
                            # 'err_S0': np.append(
                            #     fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["monoexp"]["err_S0"],
                            #     me_std_S0),
                            'err_ADC': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["monoexp"]["err_ADC"],
                                me_std_adc),
                            'chisqr': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["monoexp"]["chisqr"],
                                me_chisqr),
                            'fit': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["monoexp"]["fit"],
                                me_fit, axis=0),
                        },
                    'kurtosis':
                        {
                            # 'S0': np.append(
                            #     fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["kurtosis"]["S0"],
                            #     k_S0),
                            'ADC': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["kurtosis"]["ADC"],
                                k_adc),
                            'K': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["kurtosis"]["K"],
                                k_K),
                            # 'err_S0': np.append(
                            #     fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["kurtosis"]["err_S0"],
                            #     k_std_S0),
                            'err_ADC': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["kurtosis"]["err_ADC"],
                                k_std_adc),
                            'err_K': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["kurtosis"]["err_K"],
                                k_std_K),
                            'chisqr': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["kurtosis"]["chisqr"],
                                k_chisqr),
                            'fit': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["kurtosis"]["fit"],
                                k_fit, axis=0),
                        },
                    'astrosticks':
                        {
                            # 'S0': s_S0,
                            'D_f': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["astrosticks"]["D_f"], s_Df),
                            # 'err_S0': np.append(s_err_S0),
                            'err_Df': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["astrosticks"]["err_Df"],
                                s_err_Df),
                            'chisqr': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["astrosticks"]["chisqr"],
                                s_chisqr),
                            'fit': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["astrosticks"]["fit"],
                                s_fit, axis=0),
                        },

                    'm_astrosticks':
                        {
                            # 'S0': ck_S0,
                            'D_f': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["m_astrosticks"]["D_f"],
                                sk_Df),
                            'D_eff': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["m_astrosticks"]["D_eff"],
                                sk_Deff),
                            'K_intra': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["m_astrosticks"]["K_intra"],
                                sk_K_intra),
                            # 'err_S0': np.append(ck_err_S0),
                            'err_Df': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["m_astrosticks"]["err_Df"],
                                sk_err_Df),
                            'err_Deff': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["m_astrosticks"]["err_Deff"],
                                sk_err_Deff),
                            'err_K_intra': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["m_astrosticks"]["err_K_intra"],
                                sk_err_K_intra),
                            'chisqr': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["m_astrosticks"]["chisqr"],
                                sk_chisqr),
                            'fit': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["m_astrosticks"]["fit"],
                                sk_fit, axis=0),
                        },

                    'data':
                        {
                            'Area': np.append(
                                fit_results[tissues[(i - 1) % 2]][metabs[int((i - 1) / 2)]]["data"]["Area"],
                                norm_areas[j].reshape(1, len(norm_areas[j])), axis=0),
                        }
                }
            )

##### STORE ADC DATA AS JSON #####
with open('../data/dMRS_aging_diffAnalysis_' + minimizer_method + '.json', 'w') as fp:  
    json.dump(fit_results, fp, cls=NumpyArrayEncoder)

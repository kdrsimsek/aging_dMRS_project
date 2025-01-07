import json
import pandas as pd
import numpy as np
import glob, os

import matplotlib
import matplotlib.patches as mpatches
from scipy import stats, integrate, special
from lmfit.models import ExpressionModel, Model
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
import matplotlib as mpl
import statsmodels.api as sm
import random
from joblib import Parallel, delayed
from matplotlib.ticker import FormatStrFormatter
from statsmodels.stats.outliers_influence import summary_table

mpl.use('Qt5Agg')
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 24
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['legend.fontsize'] = 24

# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width / len(orig_handle.colors) * i, 0],
                                         width / len(orig_handle.colors),
                                         height,
                                         facecolor=c,
                                         edgecolor='black',
                                         alpha=0.2))

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


gamma = 2.67513e2  # in [1/(ms*mT)]
delta = 26.4  # in msec
Delta = 62.5  # in msec

def astrosticks(b, D_f):
    return np.sqrt(np.pi) / (2 * np.sqrt(b * D_f)) * special.erf(np.sqrt(b * D_f))

def modified_astrosticks(x, b, D_f, K_intra):
    return np.exp(-b * D_f * x ** 2 + K_intra * (b * D_f) ** 2 * x ** 4)


def integrate_mAS(b, D_f, K_intra):
    return integrate.quad(modified_astrosticks, 0, 1,
                          args=(b, D_f, K_intra), epsabs=1.49e-14, epsrel=1.49e-14)[0]


def stick_kurtosis_model(b, D_f, K_intra):
    vect = np.vectorize(integrate_mAS)
    return vect(b, D_f, K_intra)


def set_box_color(bp, color):
    plt.setp(bp['boxes'], edgecolor="black", facecolor=color)
    plt.setp(bp['whiskers'], color="black", linewidth=1.5)
    plt.setp(bp['caps'], color="black", linewidth=1.5)


def m_astrosticks_model_init_fit(init_vals, data, b, fit_method):
    stick_kurt_model = Model(stick_kurtosis_model, independent_vars=['b'], name='Modifed Astro-Sticks')
    stick_kurt_params = stick_kurt_model.make_params()

    stick_kurt_params["D_f"].set(value=init_vals[0], min=0., max=1.5, vary=True)
    stick_kurt_params["K_intra"].set(value=init_vals[1], min=0., max=0.1, vary=True)
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


def multivar_stat_analysis(df, ci_p, linecolor, text_pos, align, ax=None):
    """
    Return an axes of confidence bands using a simple approach.
    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb
    """

    y = df['param']
    x = df['Age']

    X = sm.add_constant(df[['Age', 'rGMWM']])
    # X = df[['Age', 'fGM']]
    model = sm.OLS(y, X, formula='param ~ Age + rGMWM')
    res_sm = model.fit()
    # print(res.summary(alpha=0.05))

    pred = res_sm.get_prediction()
    pred_sum = pred.summary_frame(alpha=1 - ci_p)

    pVal_age = res_sm.pvalues['Age']
    pVal_fGM = res_sm.pvalues['rGMWM']
    f_test_p = res_sm.f_pvalue

    res = stats.linregress(x, y)
    fit = res.intercept + res.slope * x
    p, cov = np.polyfit(x, y, 1, cov=True)  # parameters and covariance from of the fit of 1-D polynom.
    y_model = np.polyval(p, x)  # model using the fit parameters; NOTE: parameters here are coefficients
    x2 = np.linspace(np.min(x), np.max(x), 100)
    y2 = np.polyval(p, x2)

    # Confidence Interval (select one)

    # plot_ci_bootstrap(x, y, resid, ax=ax)
    # Statistics
    n = y.size  # number of observations
    m = p.size  # number of parameters
    dof = n - m  # degrees of freedom
    t = stats.t.ppf(ci_p, n - m)  # t-statistic; used for CI and PI bands

    # Estimates of Error in Data/Model
    resid = y - y_model  # residuals; diff. actual data from predicted values
    chi2 = np.sum((resid / y_model) ** 2)  # chi-squared; estimates error in data
    chi2_red = chi2 / dof  # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid ** 2) / dof)  # standard deviation of the error
    print(len(y[x < 50]), len(y[x >= 50]))
    t_test = stats.ttest_ind(y[x < 50], y[x >= 50])
    anova = stats.f_oneway(y[x < 50], y[x >= 50])

    print("T-Test -> p-value = " + str(t_test.pvalue))
    print("Linear Regression -> p-value = " + str(res.pvalue))
    print(f"ANOVA -> p-value is significant: {anova.pvalue < 0.05 / 6}")


    ttest_text_age = r'$Age\rightarrow\mathrm{p}=%.3f$' % (np.round(pVal_age, 3))

    ttest_text_fgm = r'$fGM/fWM\rightarrow\mathrm{p}=%.3f$' % (np.round(pVal_fGM, 3))

    fTest_text = r'$\mathrm{p}=%.3f$' % (np.round(f_test_p, 3))

    if t_test.pvalue < 5e-2 / 6:
        ttest_text = r'$\mathrm{p}^*=%.3f$' % (np.round(t_test.pvalue, 3))
    else:
        ttest_text = r'$\mathrm{p}=%.3f$' % (np.round(t_test.pvalue, 3))

    res_p_text = r'$\mathrm{p}=%.3f$' % (np.round(res.pvalue, 3))

    if anova.pvalue < 5e-2 / 6:
        anova_text = r'$\mathrm{p}^*=%.3f$' % (np.round(anova.pvalue, 3))
    else:
        anova_text = r'$\mathrm{p}=%.3f$' % (np.round(anova.pvalue, 3))

    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))

    ax.scatter(x, y, c=linecolor, s=96, marker='x')
    ax.plot(x, y_model, lw=3, c=linecolor, label=r"$Fit$")

    ax.fill_between(x2, y2 + ci, y2 - ci, color=linecolor, alpha=0.2)

    pi = t * s_err * np.sqrt(1 + 1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
    ax.plot(x2, y2 - pi, "--", color=linecolor, alpha=0.5, label="$95\%$ $Prediction$ $Limits$")
    ax.plot(x2, y2 + pi, "--", color=linecolor, alpha=0.5)

    if res.slope <1e-3:
        textstr = '\n'.join((
            r'$\mathbf{Linear\,Regression}$',
            r'$\mathrm{m}=%.1f \times 10^{-4}$' % (res.slope * 1e4),
            res_p_text,
            r'$\mathrm{R^2}=%.3f$' % (np.round(res.rvalue ** 2, 3)),
            r'${\mathbf{Multivariate\,Regression}}$',
            r'$\mathrm{R^2}=%.3f$' % (np.round(res_sm.rsquared, 3)),
            ttest_text_age,
            ttest_text_fgm,
            r'$\mathbf{T-Test}$',
            ttest_text))

    else:
        textstr = '\n'.join((
            r'$\mathbf{Linear\,Regression}$',
            r'$\mathrm{m}=%.3f$' % (res.slope),
            res_p_text,
            r'$\mathrm{R^2}=%.3f$' % (np.round(res.rvalue ** 2, 3)),
            r'$\mathbf{Multivariate\,Regression}$',
            r'$\mathrm{R^2}=%.3f$' % (np.round(res_sm.rsquared, 3)),
            ttest_text_age,
            ttest_text_fgm,
            r'$\mathbf{T-Test}$',
            ttest_text))


    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor=linecolor, alpha=0.25)

    # place a text box in upper left in axes coords
    ax.text(text_pos[0], text_pos[1], textstr, transform=ax.transAxes, fontsize=14,
            horizontalalignment="left", verticalalignment=align, bbox=props)

    return ax

def plot_ci_manual(x, y, ci_p, linecolor, text_pos, align, slope_norm, ax=None):
    """
    Return an axes of confidence bands using a simple approach.
    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb
    """
    res = stats.linregress(x, y)
    fit = res.intercept + res.slope * x
    p, cov = np.polyfit(x, y, 1, cov=True)  # parameters and covariance from of the fit of 1-D polynom.
    y_model = np.polyval(p, x)  # model using the fit parameters; NOTE: parameters here are coefficients
    x2 = np.linspace(np.min(x), np.max(x), 100)
    y2 = np.polyval(p, x2)

    # Confidence Interval (select one)

    # plot_ci_bootstrap(x, y, resid, ax=ax)
    # Statistics
    n = y.size  # number of observations
    m = p.size  # number of parameters
    dof = n - m  # degrees of freedom
    t = stats.t.ppf(ci_p, n - m)  # t-statistic; used for CI and PI bands

    # Estimates of Error in Data/Model
    resid = y - y_model  # residuals; diff. actual data from predicted values
    chi2 = np.sum((resid / y_model) ** 2)  # chi-squared; estimates error in data
    chi2_red = chi2 / dof  # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid ** 2) / dof)  # standard deviation of the error
    print(len(y[x < 50]), len(y[x >= 50]))
    t_test = stats.ttest_ind(y[x < 50], y[x >= 50])
    anova = stats.f_oneway(y[x < 50], y[x >= 50])

    print("T-Test -> p-value = " + str(t_test.pvalue))
    print("Linear Regression -> p-value = " + str(res.pvalue))
    print(f"ANOVA -> p-value is significant: {anova.pvalue < 0.05 / 6}")

    if t_test.pvalue < 5e-2 / 6:
        if t_test.pvalue < 1e-3:
            ttest_text = r'$\mathrm{p}^*<10^{-3}$'
        else:
            ttest_text = r'$\mathrm{p}^*=%.3f$' % (np.round(t_test.pvalue, 3))
    else:
        ttest_text = r'$\mathrm{p}=%.3f$' % (np.round(t_test.pvalue, 3))

    if res.pvalue < 5e-2:
        if res.pvalue < 1e-3:
            res_p_text = r'$\mathrm{p}<10^{-3}$'
        else:
            res_p_text = r'$\mathrm{p}=%.3f$' % (np.round(res.pvalue, 3))
    else:
        res_p_text = r'$\mathrm{p}=%.3f$' % (np.round(res.pvalue, 3))

    if anova.pvalue < 5e-2:
        anova_text = r'$\mathrm{p}^a=%.3f$' % (np.round(anova.pvalue, 3))
    else:
        anova_text = r'$\mathrm{p}=%.3f$' % (np.round(anova.pvalue, 3))

    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))

    ax.scatter(x, y, c=linecolor, s=96, marker='x')
    ax.plot(x, y_model, lw=3, c=linecolor, label=r"$Fit$")

    ax.fill_between(x2, y2 + ci, y2 - ci, color=linecolor, alpha=0.2)

    pi = t * s_err * np.sqrt(1 + 1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
    ax.plot(x2, y2 - pi, "--", color=linecolor, alpha=0.5, label="$95\%$ $Prediction$ $Limits$")
    ax.plot(x2, y2 + pi, "--", color=linecolor, alpha=0.5)

    if slope_norm:
        textstr = '\n'.join((
            r'$\mathbf{Linear\,Regression}$',
            r'$\mathrm{m}=%.1f \times 10^{-4}$' % (res.slope * 1e4),
            res_p_text,
            r'$\mathrm{R^2}=%.3f$' % (np.round(res.rvalue ** 2, 3)),
            r'${\mathbf{T-test}}$',
            ttest_text))

    else:
        textstr = '\n'.join((
            r'$\mathbf{Linear\,Regression}$',
            r'$\mathrm{m}=%.3f$' % (res.slope),
            res_p_text,
            r'$\mathrm{R^2}=%.3f$' % (np.round(res.rvalue ** 2, 3)),
            r'$\mathbf{T-Test}$',
            ttest_text))


    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor=linecolor, alpha=0.25)

    # place a text box in upper left in axes coords
    ax.text(text_pos[0], text_pos[1], textstr, transform=ax.transAxes, fontsize=14,
            horizontalalignment="left", verticalalignment=align, bbox=props)

    return ax


def plot_ci_bootstrap(x, y, ci_p, linecolor, text_pos, h_align, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """

    res = stats.linregress(x, y)
    fit = res.intercept + res.slope * x

    p, cov = np.polyfit(x, y, 1, cov=True)  # parameters and covariance from of the fit of 1-D polynom.
    y_model = np.polyval(p, x)  # model using the fit parameters; NOTE: parameters here are coefficients
    x2 = np.linspace(np.min(x), np.max(x), 100)
    y2 = np.polyval(p, x2)

    # Confidence Interval (select one)

    # plot_ci_bootstrap(x, y, resid, ax=ax)
    # Statistics
    n = y.size  # number of observations
    m = p.size  # number of parameters
    dof = n - m  # degrees of freedom
    t = stats.t.ppf(ci_p, n - m)  # t-statistic; used for CI and PI bands

    # Estimates of Error in Data/Model
    resid = y - y_model  # residuals; diff. actual data from predicted values
    chi2 = np.sum((resid / y_model) ** 2)  # chi-squared; estimates error in data
    chi2_red = chi2 / dof  # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid ** 2) / dof)  # standard deviation of the error
    if ax is None:
        ax = plt.gca()

    bootindex = np.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = np.polyfit(x, y + resamp_resid, 1)
        # Plot bootstrap cluster
        ax.plot(x, np.polyval(pc, y), ls="-", color=linecolor, linewidth=2, alpha=5.0 / float(nboot))

    pi = t * s_err * np.sqrt(1 + 1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))

    ax.scatter(x, y, c=colors[i], s=96, marker='x')
    ax.plot(x, y_model, lw=3, c=colors[i], label=r"$Fit$")

    ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
    ax.plot(x2, y2 - pi, "--", color=linecolor, alpha=0.5, label=r"$95\%$ $Prediction$ $Limits$")
    ax.plot(x2, y2 + pi, "--", color=linecolor, alpha=0.5)

    textstr = '\n'.join((
        r'$\mathrm{slope}=%.3f$' % (res.slope),
        r'$\mathrm{p}=%.3f$' % (res.pvalue,),
        r'$\mathrm{R^2}=%.3f$' % (res.rvalue ** 2,),))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor=linecolor, alpha=0.25)

    # place a text box in upper left in axes coords
    ax.text(text_pos[0], text_pos[1], textstr, transform=ax.transAxes, fontsize=14,
            horizontalalignment=h_align, verticalalignment="bottom", bbox=props)

    return ax


proj_path = "../"

file = open("../data/dMRS_aging_diffAnalysis_least_squares.json")
vol_info = pd.read_excel("../data/dMRS_aging_fitResults.xlsx", sheet_name='info subj')
vol_ages = vol_info.Age[:-2]

fit_results = json.load(file)
file.close()


b_values = np.array([11, 1016, 4031.25, 9057.25, 16093.5, 25140]) * 1e-3


gradients = np.sqrt(b_values / ((Delta - delta / 3) * gamma ** 2 * delta ** 2))
metabs = ['tNAA', 'tCho', 'tCr']
tissues = ['CEREBELLUM', 'PCC']

##### Load groundtruth values #####

meanpointprops = dict(marker='D', markeredgecolor="black", markerfacecolor='white', markersize=10)
flierprops = dict(markersize=10)
medianprops = dict(linewidth=3.0, color='black')
boxprops = dict(linewidth=1.5)
outprops = dict(linewidth=1.5)
spacing = 5

colors = ["#0047ab", "#cf1020"]
h_align = ["left", "right"]
diff_models = [r"$E\,[a.u.]$", r"$E\,[a.u.]$"]
params = [r'$ADC$', r'$D_{intra}$', r"$ADC$", r"$K$", r"$D_{intra}$", r"$K_{intra}$"]

##### SAMPLE SPECTRA WITH PROJECTION #####
fig, axs = plt.subplots(3, 2, figsize=(24, 24))


figMS, axsMS = plt.subplots(3, 2, figsize=(24, 24))
figY, axsY = plt.subplots(3, 2, figsize=(24, 24))
fig2, axs2 = plt.subplots(3, 2, figsize=(24, 24))
figX, axsX = plt.subplots(3, 3, figsize=(24, 24))

v_align = ['top', 'bottom', 'left', 'right']

fGM_CEREB = vol_info.fGM_CEREB[:-2]
fGM_PCC = vol_info.fGM_PCC[:-2]
rGMWM_CEREB = vol_info.spectro_gray_cereb[:-2] / vol_info.spectro_white_cereb[:-2]
rGMWM_PCC = vol_info.spectro_gray_pcc[:-2] / vol_info.spectro_white_pcc[:-2]

for i, tissue in enumerate(tissues):
    # print(tissue)

    for j, metab in enumerate(metabs):
        loc = j * spacing + 2 * i + 1  # (change spacing to 10)
        # loc = j * spacing + 2 * i #(change spacing to 10)

        s_Dfs = np.array(fit_results[tissue][metab]['astrosticks']['D_f'])  # .reshape(5, len(vol_ages)).transpose()
        df_map = s_Dfs > 0.0

        m_ADCs = np.array(fit_results[tissue][metab]['monoexp']['ADC'])  # .reshape(5, len(vol_ages)).transpose()
        k_ADCs = np.array(fit_results[tissue][metab]['kurtosis']['ADC'])  # .reshape(5, len(vol_ages)).transpose()

        Ks = np.array(fit_results[tissue][metab]['kurtosis']['K'])  # .reshape(5, len(vol_ages)).transpose()
        adc_map = (k_ADCs > 0.0) * (k_ADCs <= 0.25) * (Ks > 0)

        sk_Dfs = np.array(fit_results[tissue][metab]['m_astrosticks']['D_f'])  # .reshape(5, len(vol_ages)).transpose()
        sk_K_intras = np.array(fit_results[tissue][metab]['m_astrosticks']['K_intra'])  # .reshape(5, len(vol_ages)).transpose()
        sk_map = (0.01 < sk_K_intras) * (sk_K_intras < 0.2)


        text_pos = [1.05, 1.02 - i * 1.04]

        if tissue.__contains__('CEREBELLUM'):
            ##### STICK KURTOSIS #####
            sk_ages, sk_Dfz, sk_K_intraz, sk_rGMWM_CEREB = zip(*sorted(zip(vol_ages[sk_map],
                                                                        sk_Dfs[sk_map],
                                                                        sk_K_intras[sk_map],
                                                                        rGMWM_CEREB[sk_map])))

            multivar_stat_analysis(df=pd.DataFrame(data={'Age': sk_ages, 'param': sk_Dfz, 'rGMWM': sk_rGMWM_CEREB}),
                                   ci_p=0.95, linecolor=colors[i], text_pos=text_pos,
                                   align=v_align[i], ax=axs2[j, 0])
            multivar_stat_analysis(df=pd.DataFrame(data={'Age': sk_ages, 'param': sk_K_intraz, 'rGMWM': sk_rGMWM_CEREB}),
                                   ci_p=0.95, linecolor=colors[i], text_pos=text_pos,
                                   align=v_align[i], ax=axs2[j, 1])


            ##### KURTOSIS ADC #####
            k_ages, k_ADCz, Kz, k_rGMWM_CEREB = zip(*sorted(zip(vol_ages[adc_map],
                                                                k_ADCs[adc_map],
                                                                Ks[adc_map],
                                                                rGMWM_CEREB[adc_map])))

            multivar_stat_analysis(df=pd.DataFrame(data={'Age': k_ages, 'param': k_ADCz, 'rGMWM': k_rGMWM_CEREB}),
                                   ci_p=0.95, linecolor=colors[i], text_pos=text_pos,
                                   align=v_align[i], ax=axsY[j, 0])
            multivar_stat_analysis(df=pd.DataFrame(data={'Age': k_ages, 'param': Kz, 'rGMWM': k_rGMWM_CEREB}),
                                   ci_p=0.95, linecolor=colors[i], text_pos=text_pos,
                                   align=v_align[i], ax=axsY[j, 1])

            ##### MONOEXP #####
            m_ages, m_ADCz, m_rGMWM_CEREB = zip(*sorted(zip(vol_ages[df_map], m_ADCs[df_map], rGMWM_CEREB[df_map])))

            multivar_stat_analysis(df=pd.DataFrame(data={'Age': m_ages, 'param': m_ADCz, 'rGMWM': m_rGMWM_CEREB}),
                                   ci_p=0.95, linecolor=colors[i], text_pos=text_pos,
                                   align=v_align[i], ax=axsMS[j, 0])

            ##### STICK #####
            s_ages, s_Dfz, s_rGMWM_CEREB = zip(*sorted(zip(vol_ages[df_map], s_Dfs[df_map], rGMWM_CEREB[df_map])))

            multivar_stat_analysis(df=pd.DataFrame(data={'Age': s_ages, 'param': s_Dfz, 'rGMWM': s_rGMWM_CEREB}),
                                   ci_p=0.95, linecolor=colors[i], text_pos=text_pos,
                                   align=v_align[i], ax=axsMS[j, 1])

        else:
            ##### STICK KURTOSIS #####
            sk_ages, sk_Dfz, sk_K_intraz, sk_rGMWM_PCC = zip(*sorted(zip(vol_ages[sk_map],
                                                                      sk_Dfs[sk_map],
                                                                      sk_K_intras[sk_map],
                                                                      rGMWM_PCC[sk_map])))

            multivar_stat_analysis(df=pd.DataFrame(data={'Age': sk_ages, 'param': sk_Dfz, 'rGMWM': sk_rGMWM_PCC}),
                                   ci_p=0.95, linecolor=colors[i], text_pos=text_pos,
                                   align=v_align[i], ax=axs2[j, 0])
            multivar_stat_analysis(df=pd.DataFrame(data={'Age': sk_ages, 'param': sk_K_intraz, 'rGMWM': sk_rGMWM_PCC}),
                                   ci_p=0.95, linecolor=colors[i], text_pos=text_pos,
                                   align=v_align[i], ax=axs2[j, 1])


            #### KURTOSIS ADC #####
            k_ages, k_ADCz, Kz, k_rGMWM_PCC = zip(*sorted(zip(vol_ages[adc_map],
                                                              k_ADCs[adc_map],
                                                              Ks[adc_map],
                                                              rGMWM_PCC[adc_map])))

            multivar_stat_analysis(df=pd.DataFrame(data={'Age': k_ages, 'param': k_ADCz, 'rGMWM': k_rGMWM_PCC}),
                                   ci_p=0.95, linecolor=colors[i], text_pos=text_pos,
                                   align=v_align[i], ax=axsY[j, 0])
            multivar_stat_analysis(df=pd.DataFrame(data={'Age': k_ages, 'param': Kz, 'rGMWM': k_rGMWM_PCC}),
                                   ci_p=0.95, linecolor=colors[i], text_pos=text_pos,
                                   align=v_align[i], ax=axsY[j, 1])

            ##### MONOEXP #####
            m_ages, m_ADCz, m_rGMWM_PCC = zip(*sorted(zip(vol_ages[df_map], m_ADCs[df_map], rGMWM_PCC[df_map])))

            multivar_stat_analysis(df=pd.DataFrame(data={'Age': m_ages, 'param': m_ADCz, 'rGMWM': m_rGMWM_PCC}),
                                   ci_p=0.95, linecolor=colors[i], text_pos=text_pos,
                                   align=v_align[i],  ax=axsMS[j, 0])

            ##### STICK #####
            s_ages, s_Dfz, s_rGMWM_PCC = zip(*sorted(zip(vol_ages[df_map], s_Dfs[df_map], rGMWM_PCC[df_map])))

            multivar_stat_analysis(df=pd.DataFrame(data={'Age': s_ages, 'param': s_Dfz, 'rGMWM': s_rGMWM_PCC}),
                                   ci_p=0.95, linecolor=colors[i], text_pos=text_pos,
                                   align=v_align[i], ax=axsMS[j, 1])

        ##### MONOEXP PLOTS #####
        bp_a1 = axs.flatten()[0].boxplot(m_ADCs, meanprops=meanpointprops,
                               medianprops=medianprops, boxprops=boxprops, patch_artist=True, flierprops=flierprops,
                               showfliers=True, showmeans=True, positions=[loc], widths=1)

        set_box_color(bp_a1, colors[i])

        ##### STICKS PLOTS #####
        bp_b1 = axs.flatten()[1].boxplot(s_Dfs, meanprops=meanpointprops,
                               medianprops=medianprops, boxprops=boxprops, patch_artist=True, flierprops=flierprops,
                               showfliers=True, showmeans=True, positions=[loc], widths=1)
        
        set_box_color(bp_b1, colors[i])

        ##### KURTOSIS PLOTS #####
        bp_c1 = axs.flatten()[2].boxplot(k_ADCs, meanprops=meanpointprops,
                               medianprops=medianprops, boxprops=boxprops, patch_artist=True, flierprops=flierprops,
                               showfliers=True, showmeans=True, positions=[loc], widths=1)
        bp_c2 = axs.flatten()[3].boxplot(Ks, meanprops=meanpointprops,
                               medianprops=medianprops, boxprops=boxprops, patch_artist=True, flierprops=flierprops,
                               showfliers=True, showmeans=True, positions=[loc], widths=1)

        set_box_color(bp_c1, colors[i])
        set_box_color(bp_c2, colors[i])

        ##### STICK-KURTOSIS PLOTS #####
        bp_d1 = axs.flatten()[4].boxplot(sk_Dfs, meanprops=meanpointprops,
                               medianprops=medianprops, boxprops=boxprops, patch_artist=True, flierprops=flierprops,
                               showfliers=True, showmeans=True, positions=[loc], widths=1)
        bp_d2 = axs.flatten()[5].boxplot(sk_K_intras, meanprops=meanpointprops,
                               medianprops=medianprops, boxprops=boxprops, patch_artist=True, flierprops=flierprops,
                               showfliers=True, showmeans=True, positions=[loc], widths=1)

        set_box_color(bp_d1, colors[i])
        set_box_color(bp_d2, colors[i])

        print(tissue, metab)

for ax in axs2:
    ax[0].set_ylim(0, 0.75)
    ax[1].set_ylim(0, 0.15)
    ax[0].set_xlim(20, 85)
    ax[1].set_xlim(20, 85)

for ax in axsX:
    ax[0].set_ylim(0, 0.75)
    ax[2].set_ylim(0, 0.15)
    ax[0].set_xlim(20, 85)
    ax[1].set_xlim(20, 85)

for ax in axsY:
    ax[0].set_ylim(0, 0.25)
    ax[1].set_ylim(0, 3)
    ax[0].set_xlim(20, 85)
    ax[1].set_xlim(20, 85)

for ax in axsMS:
    ax[0].set_ylim(0, 0.25)
    ax[1].set_ylim(0, 1)
    ax[0].set_xlim(20, 85)
    ax[1].set_xlim(20, 85)

pa3 = mpatches.Patch(facecolor=colors[0], edgecolor="black", linewidth=2.0)
pa4 = mpatches.Patch(facecolor=colors[1], edgecolor="black", linewidth=2.0)
leg_labels = [r"$CEREBELLUM$", r"$PCC$", "$Mean$", "$Median$", "$Outliers$"]
leg_lines = [pa3, pa4, bp_a1["means"][0], bp_a1["medians"][0], bp_a1["fliers"][0]]

leg = fig.legend(leg_lines, leg_labels, handler_map={MulticolorPatch: MulticolorPatchHandler()},
                 loc='upper center', frameon=True, ncol=5, handlelength=2, mode="expand",
                 borderaxespad=0.5, facecolor='none')

leg_lines2 = leg_lines[:2] + [plt.Line2D([0], [0], marker='x', markeredgecolor='black', lw=0,
                                         markeredgewidth=2, markersize=12, label=r"$Model$ $Parameters$", )]
leg_labels2 = leg_labels[:2] + [r"$Model$ $Parameters$"]
leg2 = fig2.legend(leg_lines2, leg_labels2, handler_map={MulticolorPatch: MulticolorPatchHandler()},
                   loc='upper center', frameon=True, ncol=3,
                   borderaxespad=0.5, mode="expand", facecolor='none')

legX = figX.legend(leg_lines2, leg_labels2, handler_map={MulticolorPatch: MulticolorPatchHandler()},
                   loc='upper center', frameon=True, ncol=3,
                   borderaxespad=0.5, mode="expand", facecolor='none')

legY = figY.legend(leg_lines2, leg_labels2, handler_map={MulticolorPatch: MulticolorPatchHandler()},
                   loc='upper center', frameon=True, ncol=3,
                   borderaxespad=0.5, mode="expand", facecolor='none')

legMS = figMS.legend(leg_lines2, leg_labels2, handler_map={MulticolorPatch: MulticolorPatchHandler()},
                     loc='upper center', frameon=True, ncol=3,
                     borderaxespad=0.5, mode="expand", facecolor='none')

legend_elements = [plt.Line2D([0], [0], color='black', lw=3, label=r"$Fit$"),
                   plt.Line2D([0], [0], linestyle='--', color='black', lw=3, label=r"$95\%$ $Prediction$ $Limits$",
                              alpha=0.5),
                   mpatches.Patch(facecolor='black', alpha=0.5, label=r"$95\%$ $Confidence$ $Limits$")]
display = (0, 1)
anyArtist = plt.Line2D((0, 1), (1, 1), color="#b9cfe7")  # create custom artists
legend = fig2.legend(handles=legend_elements, loc="lower center", ncol=3, borderaxespad=0.5, mode="expand")
frame = legend.get_frame().set_edgecolor("0.5")

legendY = figY.legend(handles=legend_elements, loc="lower center", ncol=3, borderaxespad=0.5, mode="expand")
frameY = legendY.get_frame().set_edgecolor("0.5")

legendX = figX.legend(handles=legend_elements, loc="lower center", ncol=3, borderaxespad=0.5, mode="expand")
frameX = legendX.get_frame().set_edgecolor("0.5")

legendMS = figMS.legend(handles=legend_elements, loc="lower center", ncol=3, borderaxespad=0.5, mode="expand")
frameMS = legendMS.get_frame().set_edgecolor("0.5")


m = 0
x_ticks = [5 * (i + 1) - 3 for i in range(len(metabs))]
for ax in axs.flatten():
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(metabs)
    ax.set_xlim([x_ticks[0] - 5, x_ticks[-1] + 5])
    ax.set_ylabel(params[m])
    m += 1


for m, ax in enumerate(axsMS.flatten()):
    ax.set_xlabel(r"$Age$")
    ax.set_ylabel(params[0:2][m % 2])

for m, ax in enumerate(axs2.flatten()):
    ax.set_xlabel(r"$Age$")
    ax.set_ylabel(params[4:6][m % 2])

for m, ax in enumerate(axsY.flatten()):
    ax.set_xlabel(r"$Age$")
    ax.set_ylabel(params[2:4][m % 2])

for m, ax in enumerate(axsX.flatten()):
    ax.set_xlabel(r"$Age$")
    ax.set_ylabel(params[6:][m % 3])

for m, ax in enumerate(axsY):
    ax[1].text(-0.25, 1.075, f'${metabs[m]}$', horizontalalignment='center', verticalalignment="center",
               transform=ax[1].transAxes, fontdict={'fontsize': 28})

for m, ax in enumerate(axs2):
    ax[1].text(-0.25, 1.075, f'${metabs[m]}$', horizontalalignment='center', verticalalignment="center",
               transform=ax[1].transAxes, fontdict={'fontsize': 28})

for m, ax in enumerate(axsMS):
    ax[1].text(-0.25, 1.075, f'${metabs[m]}$', horizontalalignment='center', verticalalignment="center",
               transform=ax[1].transAxes, fontdict={'fontsize': 28})

for m, ax in enumerate(axsX):
    ax[1].set_title(f'${metabs[m]}$', fontdict={'fontsize': 28})


text_xy = [1.03, 0.99]
text_xy2 = [1.03, 0.01]
slope_norm = False
n = 0
brain = ["GM", "WM", "CSF", "GM/(GM+WM)", "WM/(GM+WM)"]

####### Relative GM Volume Fraction ########
text_xy = [1.03, 0.99]
text_xy2 = [1.03, 0.01]
figG, axsG = plt.subplots(1, 1, figsize=(16, 16))

plot_ci_manual(x=np.array(vol_ages[1:]), y=np.array(rGMWM_CEREB[1:]), ci_p=0.95,
               linecolor=colors[0], text_pos=text_xy, align=v_align[0], slope_norm=slope_norm, ax=axsG)
plot_ci_manual(x=np.array(vol_ages), y=np.array(rGMWM_PCC), ci_p=0.95,
               linecolor=colors[1], text_pos=text_xy2, align=v_align[1], slope_norm=slope_norm, ax=axsG)

axsG.set_xlabel(r"$Age$")
axsG.set_ylabel(r"$fGM/fWM$ $[a.u.]$")

leg_lines4 = leg_lines[:2] + [plt.Line2D([0], [0], marker='x', markeredgecolor='black', lw=0,
                                         markeredgewidth=2, markersize=12, label=r"$Volume$ $Fraction$", )]
leg_labels4 = leg_labels[:2] + [r"$Volume$ $Fraction$"]
legG = figG.legend(leg_lines4, leg_labels4, handler_map={MulticolorPatch: MulticolorPatchHandler()},
                   loc='upper center', frameon=True, ncol=3,
                   borderaxespad=0.5, mode="expand", facecolor='none')

# handles, labels = axs2[0].get_legend_handles_labels()
legend_elements = [plt.Line2D([0], [0], color='black', lw=3, label=r"$Fit$"),
                   plt.Line2D([0], [0], linestyle='--', color='black', lw=3, label=r"$95\%$ $Prediction$ $Limits$",
                              alpha=0.5),
                   mpatches.Patch(facecolor='black', alpha=0.5, label=r"$95\%$ $Confidence$ $Limits$")]
display = (0, 1)
anyArtist = plt.Line2D((0, 1), (1, 1), color="#b9cfe7")  # create custom artists
legend = figG.legend(handles=legend_elements, loc="lower center", ncol=3, borderaxespad=0.5, mode="expand")
frame = legend.get_frame().set_edgecolor("0.5")

######## COHORT AVERAGE ANALYSIS ############
fig5, axs5 = plt.subplots(2, 3, sharex='row', sharey='row')

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

D_apps = []
K_apps = []

for i, tissue in enumerate(tissues):
    for j, metab in enumerate(metabs):
        sk_Df = sk_Deff = sk_K_intra = sk_err_Df = sk_err_Deff = sk_err_K_intra = sk_chisqr = sk_fit = 0
        me_adc = me_std_adc = me_chisqr = me_fit = 0
        k_adc = k_K = k_std_adc = k_std_K = k_chisqr = k_fit = 0
        s_Df = s_err_S0 = s_err_Df = s_chisqr = s_fit = 0

        difSig = np.array(fit_results[tissue][metab]['data']['Area'],
                          dtype=float)  

        meanSig = np.nanmean(difSig, axis=0)
        stdSig = np.nanstd(difSig, axis=0)

        print(f"----------{tissue}-{metab}----------")
        try:
            mono_exp_results = monoexp_model.fit(data=meanSig[0:3],
                                                 params=monoexp_params,
                                                 x=b_values[0:3],
                                                 method="least_squares")

            print("----------MONOEXPONENTIAL MODEL FIT----------")
            print(mono_exp_results.fit_report())
            me_adc = mono_exp_results.best_values['adc']

            try:
                me_std_adc = np.sqrt(np.diag(mono_exp_results.covar))
            except AttributeError:
                pass

            me_chisqr = mono_exp_results.chisqr
            me_fit = mono_exp_results.best_fit

        except ValueError:
            mono_exp_results = 0
            pass
        try:
            kurtosis_results = kurtosis_model.fit(data=meanSig[0:4],
                                                  params=kurtosis_params,
                                                  x=b_values[0:4],
                                                  method="least_squares")

            print("----------KURTOSIS FIT----------")
            print(kurtosis_results.fit_report())
            k_adc = kurtosis_results.best_values['adc']
            k_K = kurtosis_results.best_values['K']

            try:
                k_std_adc, k_std_K = np.sqrt(np.diag(kurtosis_results.covar))
            except AttributeError:
                pass

            k_chisqr = kurtosis_results.chisqr
            k_fit = kurtosis_results.best_fit

        except ValueError:
            kurtosis_results = 0
            pass

        try:
            stick_results = stick_model.fit(data=meanSig,
                                            params=stick_params,
                                            b=b_values,
                                            # weights=1. / std_area,
                                            method="least_squares")

            print("----------ASTRO-STICKS MODEL FIT----------")
            print(stick_results.fit_report())
            # s_S0 = stick_results.best_values['s0']
            s_Df = stick_results.best_values['D_f']

            try:
                s_err_Df = np.sqrt(stick_results.covar)
            except AttributeError:
                pass

            s_chisqr = stick_results.chisqr
            s_fit = stick_results.best_fit

        except ValueError:
            stick_results = 0
            pass

        try:
            stick_kurt_results = m_astrosticks_model_fit(n_trials=250,
                                                         data=meanSig,
                                                         b=b_values,
                                                         fit_method='least_squares')

            print("----------MODIFIED ASTRO-STICKS MODEL FIT----------")
            print(stick_kurt_results.fit_report())
            # ck_S0 = cyl_kurt_results.best_values['s0']
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
            sk_fit = stick_kurt_results.best_fit

        except ValueError:
            stick_kurt_results = 0
            pass

        D_apps.append(sk_Df)
        K_apps.append(sk_K_intra)

        lin_fit = np.polyfit(np.log(b_values[-3:]), np.log(meanSig[-3:]), 1)
        slope, intercept = lin_fit
        print(f"----------{metab}----------")
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")

        for sig in difSig:
            axs5.flatten()[j].plot(b_values, sig, ls="-", color=colors[i], linewidth=3, alpha=0.2, label=r'$Subject$')


        axs5.flatten()[j].plot(b_values, meanSig, ls="-", color=colors[i], linewidth=3, alpha=1, label=r'$Subject$')
        axs5.flatten()[j+3].loglog(b_values, meanSig, ls='-', color=colors[i], linewidth=3, alpha=1, label=r'$Cohort$')
        axs5.flatten()[j+3].loglog(b_values, b_values**(-0.5)*np.exp(0.25), ls="--", color='black', linewidth=3, alpha=1,
                               label=r'$Cohort$')

        axs5.flatten()[j+3].fill_between(b_values, meanSig + stdSig, meanSig - stdSig, color=colors[i], alpha=0.2)

        axs5.flatten()[j+3].text(np.exp(0.05), np.exp(-1.4+i*0.1), f"slope: {np.round(slope,2)}", fontsize=16, color=colors[i], horizontalalignment="left")

        axs5.flatten()[j].set_title(r'$' + metab + r'$')
        axs5.flatten()[j].set_xlabel(r'$b\, [ms/\mu m^2]$')
        axs5.flatten()[j+3].set_xlabel(r'$b\, [ms/\mu m^2]$')


for ax in axs5[1]:
    ax.set_ylim([np.exp(-1.5), np.exp(0)])
    ax.set_xlim([np.exp(-0.1), np.exp(3.5)])
    ax.text(np.exp(1.05), np.exp(-.15), r"$S(b)\sim b^{-\frac{1}{2}}$", fontsize=16, color='black',
                               horizontalalignment="left")


axs5[1][0].yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

for ax in axs5[0]:
    ax.set_yscale('log')
    ax.set_ylim([1e-1, 1e0])
    ax.minorticks_off()
    ax.set_yticks([1e-1, 1e0])
    ax.set_xlim([0, 26])

axs5.flatten()[0].set_ylabel(r'$S/S_0$')
axs5.flatten()[3].set_ylabel(r'$S/S_0$')
patch = mpatches.Patch(facecolor="#0047ab", edgecolor='black')
patch2 = mpatches.Patch(facecolor="#cf1020", edgecolor='black')
mpatch = MulticolorPatch(colors)
leg5 = fig5.legend([patch, patch2, mpatch], [r'$CEREBELLUM$', r'$PCC$', r'Subject'],
                   handler_map={MulticolorPatch: MulticolorPatchHandler()},
                   loc='upper center', frameon=True, ncol=3,
                   borderaxespad=0.5, mode="expand", facecolor='none')

plt.show()

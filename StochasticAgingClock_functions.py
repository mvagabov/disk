import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import ttest_ind, pearsonr, spearmanr
from fitter import Fitter, get_common_distributions, get_distributions
import pickle
import matplotlib.ticker as plticker
from sklearn import linear_model
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet, ElasticNetCV
import pingouin as pg
import statsmodels
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

input_path = ''
plot_path = ''
#if not os.path.exists(f'{plot_path}'):
#    os.mkdir(f'{plot_path}')
filetype = '.pdf'
fontsize=7
height=2.17 
scatter_size=1
sns.set(style='white') #font
dot_color = 'grey' 
line_color = 'black'
xlab = 'Simulated age'
ylab = 'Predicted age'
n_jobs=6


def correct_Bio_Age(corr_arr):
    corr_arr_BioAgeCorr = corr_arr.copy()
    bio = corr_arr_BioAgeCorr.Bio_Age.values
    bio2 = [(((y - (scipy.special.erfinv(
        0.5 - (((scipy.special.erf(((372 - y) / (192 / 3)) / np.sqrt(2)) / 2) + 0.5) / 2)) * np.sqrt(2) * 2 * (
                            192 / 3))))) for y in bio]
    corr_arr_BioAgeCorr.Bio_Age = bio2

    return corr_arr_BioAgeCorr


def make_binary(df, filter_genes='WBG', log=False, q=0.5):
    df_div = df.copy()
    df_div = df_div.filter(regex=filter_genes)
    df_div[df_div == 0] = np.nan
    df_div['Median'] = df_div.quantile(q=q, axis=1)  # q=0.5 is the Median
    df_div = df_div.filter(regex=filter_genes).div(df_div.Median, axis=0)
    df_div[df_div.isna()] = 0
    df_div[df_div <= 1] = 0
    df_div[df_div > 1] = 1

    df_div['Bio_Age'] = df.Bio_Age

    return df_div



# no prediction possible as long as the values are not kept between 0 and 1
def random_epi_nolimit(ground, samples_per_age=3, epi_sites=20000, noise_ground=0.01, noise_age=0.05, age_steps=100):
    samples = []
    ages = []
    for age in range(age_steps):
        for _ in range(samples_per_age):  # 100 samples per age group
            # add some starting noise
            x = ground + noise_ground * np.random.randn(epi_sites)

            for _ in range(age):
                noise = noise_age * np.random.randn(epi_sites)
                x = x + noise
            ages.append(age)
            samples.append(x)
    return samples, ages


# same as above, but limit the data between 0 and 1
def random_epi(ground, samples_per_age=3, epi_sites=20000, noise_ground=0.01, noise_age=0.05, age_steps=100):
    samples = []
    ages = []
    for age in range(age_steps):
        for _ in range(samples_per_age):  # 100 samples per age group
            # add some starting noise
            x = ground + noise_ground * np.random.randn(epi_sites)
            x[x > 1] = 1
            x[x < 0] = 0

            for _ in range(age):
                noise = noise_age * np.random.randn(epi_sites)
                x = x + noise
                x[x > 1] = 1
                x[x < 0] = 0
            ages.append(age)
            samples.append(x)
    return samples, ages


# with logit
from scipy.special import logit, expit
def random_epi_logit(ground, samples_per_age=3, epi_sites=20000, noise_ground=0.01, noise_age=0.2, age_steps=100):
    samples = []
    ages = []
    for age in range(age_steps):
        for _ in range(samples_per_age):  # 100 samples per age group
            # add some starting noise
            x = logit(ground)
            x = x + noise_ground * np.random.randn(epi_sites)
            
            for _ in range(age):
                noise = noise_age * np.random.randn(epi_sites)
                x = x + noise
            x = expit(x)
            ages.append(age)
            samples.append(x)
    return samples, ages

def random_epi_logit_only(ground, samples_per_age=3, epi_sites=20000, noise_ground=0.01, noise_age=0.2, age_steps=100):
    samples = []
    ages = []
    for age in range(age_steps):
        for _ in range(samples_per_age):  # 100 samples per age group
            # add some starting noise
            x = logit(ground)
            x = x + noise_ground * np.random.randn(epi_sites)
            
            for _ in range(age):
                noise = noise_age * np.random.randn(epi_sites)
                x = x + noise

            ages.append(age)
            samples.append(x)
    return samples, ages


def random_epi_logit_everystep(ground, samples_per_age=3, epi_sites=20000, noise_ground=0.01, noise_age=0.05, age_steps=100):
    samples = []
    ages = []
    for age in range(age_steps):
        for _ in range(samples_per_age):  # 100 samples per age group
            # add some starting noise
            x = logit(ground)
            x = x + noise_ground * np.random.randn(epi_sites)
            x = expit(x)
            for _ in range(age):
                noise = noise_age * np.random.randn(epi_sites)
                x = logit(x)
                x = x + noise
                x = expit(x)
            ages.append(age)
            samples.append(x)
    return samples, ages

@ignore_warnings(category=ConvergenceWarning)
def pred_and_plot(samples, ages, samples2, ages2, xlab, ylab, savepic=True, tick_step=25, fontsize=12,
                         height=3.2, regr=None,  scatter_size=1, color='grey', n_jobs=1, line_color='black'):
    stats = []
    if regr:
        pred_y = regr.predict(samples2)
    else:
        regr = ElasticNetCV(random_state=0, max_iter=1000, l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], n_jobs=n_jobs)
        regr.fit(samples, ages)
        pred_y = regr.predict(samples2)

    if savepic:
        sns.set(font='Times New Roman', style='white')
        g = sns.jointplot(x=ages2, y=pred_y, kind='reg', height=height,
                          scatter_kws={'s': scatter_size}, color=color, joint_kws={'line_kws':{'color':line_color}}) 
        g.ax_joint.set_ylim([0, 99])
        lims = [0, 99]  
        g.ax_joint.plot(lims, lims, ':k')
        g.set_axis_labels(xlab, ylab, fontsize=fontsize)
        loc = plticker.MultipleLocator(base=tick_step)
        g.ax_joint.xaxis.set_major_locator(loc)
        g.ax_joint.yaxis.set_major_locator(loc)
        if isinstance(tick_step, int):
            g.ax_joint.set_xticklabels([int(tt) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
            g.ax_joint.set_yticklabels([int(tt) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
        else:
            g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
            g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
        plt.tight_layout()
        plt.show()
        #plt.savefig(f'{outname}_fontsize{fontsize}_height{height}{filetype}')
        plt.close()
    stats.append(pearsonr(ages2, pred_y))
    stats.append(spearmanr(ages2, pred_y))
    stats.append(r2_score(ages2, pred_y))
    stats.append(np.median(abs(ages2 - pred_y)))
    return regr, stats



def get_noise_func_parm(t, start_ind, end_ind, step=10, normalize=None):
    '''
    compute the difference between start_ind - end_ind and subsequent the noise function
    :param t: Dataframe with datasets in the columns
    :param start_ind: left side of subtraction
    :param end_ind: right side
    :param step: how many quantiles to compute for the noise function
    :return: 
    '''
    d = {}
    d['Q1'] = []
    d['Q2'] = []
    d['Param'] = []

    for i in np.array(range(step)) / step:
        c1 = t.iloc[:, start_ind]
        c2 = t.iloc[:, end_ind]
        q1 = c1.quantile(q=i)
        q2 = c1.quantile(q=i + (1 / step))
        # get the quantile from the young dataset
        if i == 0:
            q1 = 0  
            d['Q1'].append(q1)
            d['Q2'].append(q2)
            c1_q = c1[(c1 >= q1) & (c1 <= q2)]
        elif i == (step - 1) / step:
            q2 = 1
            d['Q1'].append(q1)
            d['Q2'].append(q2)
            c1_q = c1[(c1 > q1) & (c1 <= q2)]
        else:
            d['Q1'].append(q1)
            d['Q2'].append(q2)
            c1_q = c1[(c1 > q1) & (c1 <= q2)]
        # get the same sites for the older dataset
        c2_q = c2[c2.index.isin(c1_q.index)]
        rs = len(c1_q)
        # old - young
        dif1 = (c2_q - c1_q).values
        if normalize == 'Mean':
            dif1 = dif1 - np.mean(dif1)
        elif normalize == 'Median':
            dif1 = dif1 - np.median(dif1)

        listdif = list(dif1)
        f = Fitter(listdif, distributions=['lognorm'], timeout=120)
        f.fit()
        d['Param'].append(f.get_best())
    df = pd.DataFrame(d)
    return df


# take the ground array, and the noise df to generate random noise fitting to the ground truth
def apply_biol_noise_ground(ground, noise_ground_df):
    c3 = pd.DataFrame(ground)
    c3.columns = ['Ground']
    c3['Rand'] = np.nan
    for i in range(len(noise_ground_df)):
        q1 = noise_ground_df['Q1'][i]
        q2 = noise_ground_df['Q2'][i]
        if q1 == 0:
            r1 = scipy.stats.lognorm.rvs(size=len(c3[(c3['Ground'] >= q1) & (c3['Ground'] <= q2)]),
                                         scale=noise_ground_df['Param'][i]['lognorm']['scale'],
                                         loc=noise_ground_df['Param'][i]['lognorm']['loc'],
                                         s=noise_ground_df['Param'][i]['lognorm']['s'])
            c3.loc[(c3['Ground'] >= q1) & (c3['Ground'] <= q2), 'Rand'] = r1
        else:
            r1 = scipy.stats.lognorm.rvs(size=len(c3[(c3['Ground'] > q1) & (c3['Ground'] <= q2)]),
                                         scale=noise_ground_df['Param'][i]['lognorm']['scale'],
                                         loc=noise_ground_df['Param'][i]['lognorm']['loc'],
                                         s=noise_ground_df['Param'][i]['lognorm']['s'])
            c3.loc[(c3['Ground'] > q1) & (c3['Ground'] <= q2), 'Rand'] = r1
    return c3



def random_epi_biol_age(ground, noise_ground_df, noise_age_df, samples_per_age=3, epi_sites=20000, age_steps=1,
                        age_start=0, age_end=100, noise_norm=1):
    samples = []
    ages = []
    for age in range(age_start, age_end, age_steps):
        for _ in range(samples_per_age):  # 100 samples per age group
            # add some starting noise
            n = apply_biol_noise_ground(ground, noise_ground_df)
            x = (n.Ground + n.Rand).values
            x[x > 1] = 1
            x[x < 0] = 0

            for _ in range(age):
                n = apply_biol_noise_ground(ground, noise_age_df) # measure noise always form ground
                x = (x + (n.Rand / noise_norm)).values
                x[x > 1] = 1
                x[x < 0] = 0
            ages.append(age)
            samples.append(x)
    return samples, ages



####### single cell
# the maintenance is fixed to the same value
# use binary NOT to do all states at once
def update_cells_fast_fixed(cells, Em, cell_num=100):
    lens = len(cells)
    flip = np.random.uniform(size=[lens,cell_num])>=Em
    cells[flip] = ~cells[flip]
    return cells

def simulate_cells_for_age_fixed(ground, Em, Ed=0, samples_per_age=3, age_steps=30, cell_num=100, deviate_ground=True):
    samples = []
    ages = []
    # generate cell_num cells for each site
    for _ in range(samples_per_age):
        for age in range(1, age_steps):
            x = ground
            if deviate_ground:
                x = x + 0.01 * np.random.randn(len(ground)) # slightly deviate the ground state
                x[x > 1] = 1
                x[x < 0] = 0
            cells = np.array([int(cell_num * g) * [True] + (cell_num - int(cell_num * g)) * [False] for g in x])
            for _ in range(age):
                cells = update_cells_fast_fixed(cells, Em, cell_num=cell_num)
            samples.append(cells.mean(axis=1))  # compute bulk average again and append
            ages.append(age)
    return samples, ages


def get_noise_Em_all_new(t, sample_name, Em_lim=0.95, Ed_lim=0.23):
    '''
    Compute the average Em value for sample_name based on 
    1+Ed(z-1)/z, where z is the equilibrium value
    :param t: Dataframe with datasets in the columns
    :param sample_name: sample of interest, usually the oldest to get the equilibrium value

    :param step: how many quantiles to compute for the noise function
    :return: 
    '''
    d = {}
    d['Site'] = []
    d['Value'] = []
    d['Em'] = []
    d['Ed'] = []

    c1 = t[sample_name]

    for row in c1.index:
        d['Site'].append(row)
        eq = c1.loc[row]  # median equilibrium value that we define
        d['Value'].append(eq)
        Ed = ((Em_lim - 1) * eq) / (eq - 1)
        if Ed > Ed_lim:
            Ed = np.random.randint(1, int(Ed_lim * 1000)) / 1000 
        Em = 1 + Ed * (eq - 1) / eq

        d['Em'].append(Em)
        d['Ed'].append(Ed)
    df = pd.DataFrame(d)
    return df




def update_cells_fast_empirical_noquantile(cells, eml, edl, cell_num=100):
    
    lenc = len(cells[0])
    for i, c in enumerate(cells):  # this loops over the 2000 sites --> every site has a specific Em
        Em = eml[i]
        Ed = edl[i]
        flip = ((c == False) & (np.random.uniform(size=lenc) <= Ed)) | ((c == True) & (np.random.uniform(size=lenc) >= Em))
        c[flip] = ~c[flip]
    return cells


def simulate_for_age_empirical_noquantile(ground, Em_df, samples_per_age=3, age_steps=30, cell_num=100, deviate_ground=True):
    samples = []
    ages = []
    Em_df = Em_df.sort_values(by='Site')
    eml = Em_df.Em.values
    edl = Em_df.Ed.values
    for _ in range(samples_per_age):
        for age in range(1, age_steps):
            x = ground
            if deviate_ground:
                x = x + 0.01 * np.random.randn(len(ground)) # slightly deviate the ground state
                x[x > 1] = 1
                x[x < 0] = 0
            cells = np.array([int(cell_num * g) * [True] + (cell_num - int(cell_num * g)) * [False] for g in x])

            for _ in range(age):
                cells = update_cells_fast_empirical_noquantile(cells, eml, edl, cell_num=cell_num)
            samples.append(cells.mean(axis=1))
            ages.append(age)
    return samples, ages

#### Code for published clocks
def anti_transform_age(exps):
    adult_age = 20
    ages = []
    for exp in exps:
        if exp < 0:
            age = (1 + adult_age) * (np.exp(exp)) - 1
            ages.append(age)
        else:
            age = (1 + adult_age) * exp + adult_age
            ages.append(age)
    ages = np.array(ages)
    return ages


def get_clock(clock_csv_file, sep=','):
    coef_data = pd.read_csv(clock_csv_file, sep=sep)
    if ('Intercept' in coef_data.CpGmarker.values) or ('(Intercept)' in coef_data.CpGmarker.values):
        intercept = coef_data[0:1].CoefficientTraining[0]
        coef_data = coef_data.drop(0)
    else:
        intercept = 0
    coef_data = coef_data[coef_data.CpGmarker.str.startswith('cg')]
    coef_data = coef_data.sort_values(by='CpGmarker')
    horvath_cpgs = np.array(coef_data.CpGmarker)
    coefs = np.array(coef_data.CoefficientTraining)
    horvath_model = linear_model.LinearRegression()
    horvath_model.coef_ = coefs
    horvath_model.intercept_ = intercept
    return horvath_cpgs, horvath_model


def get_clock_data(dat, cpg_sites, young16y='GSM1007467', old88y='GSM1007832'):
    h = dat.loc[cpg_sites]
    data = h[[young16y, old88y]]  
    data = data.sort_index()
    return data


def get_samples(dat, cpg_sites, age_steps=30, cell_num=100, Em_lim=0.95, Ed_lim=0.23, young16y='GSM1007467',
                old88y='GSM1007832'):
    data = get_clock_data(dat, cpg_sites=cpg_sites, young16y=young16y, old88y=old88y)
    Em_df_new = get_noise_Em_all_new(data, old88y, Em_lim=Em_lim, Ed_lim=Ed_lim)
    ground = np.array(data[young16y])
    samples_emp_noquantile, ages_emp_noquantile = simulate_for_age_empirical_noquantile(ground, Em_df_new,
                                                                                        samples_per_age=1,
                                                                                        age_steps=age_steps,
                                                                                        cell_num=cell_num)
    return samples_emp_noquantile, ages_emp_noquantile, data[[young16y]]




def get_samples_fixed(dat, cpg_sites, Em, Ed, samples_per_age=1, age_steps=30, cell_num=100, young16y='GSM1007467', old88y='GSM1007832'):
    data = get_clock_data(dat, cpg_sites=cpg_sites, young16y=young16y, old88y=old88y)
    ground = np.array(data[young16y])
    samples_emp_noquantile, ages_emp_noquantile = simulate_cells_for_age_fixed(ground, Em, Ed, samples_per_age=samples_per_age,
                                                                               age_steps=age_steps, cell_num=cell_num)
    return samples_emp_noquantile, ages_emp_noquantile, data[[young16y]]



# complete random Em and Ed
def get_noise_Em_all_random(t, Em_lim=0.95, Ed_lim=0.23):

    d = {}
    d['Site'] = []
    d['Em'] = []
    d['Ed'] = []

    for row in t.index:
        d['Site'].append(row)
        # completely random within the limits
        Ed = np.random.randint(1, int(Ed_lim * 10000)) / 10000
        Em =np.random.randint(int(Em_lim * 10000), 9999) / 10000

        d['Em'].append(Em)
        d['Ed'].append(Ed)
    df = pd.DataFrame(d)
    return df


def get_samples_random(dat, cpg_sites, age_steps=30, cell_num=100, Em_lim=0.95, Ed_lim=0.23, young16y='GSM1007467',
                old88y='GSM1007832'):
    data = get_clock_data(dat, cpg_sites=cpg_sites, young16y=young16y, old88y=old88y)
    Em_df_new = get_noise_Em_all_random(data, Em_lim=Em_lim, Ed_lim=Ed_lim)
    ground = np.array(data[young16y])
    samples_emp_noquantile, ages_emp_noquantile = simulate_for_age_empirical_noquantile(ground, Em_df_new,
                                                                                        samples_per_age=1,
                                                                                        age_steps=age_steps,
                                                                                        cell_num=cell_num)
    return samples_emp_noquantile, ages_emp_noquantile, data[[young16y]]




@ignore_warnings(category=ConvergenceWarning)
def get_prediction(samples_emp_noquantile, ages_emp_noquantile, data, clock_model, outname, scatter_size=1,
                   tick_step=25, fontsize=12, height=3.2, filetype='.pdf', kind='scatter', lim_ax = False, loc_tick=False, color='grey', line_color='black', dot_color = 'grey', tight=True,xlab='Simulated age', ylab='Predicted epigenetic age'):

    for i, s in enumerate(samples_emp_noquantile):
        data[f'Sample{i}'] = s
    data = data.T
    if len(clock_model.coef_) == 353:
        pred = anti_transform_age(clock_model.predict(data))
    else:
        pred = clock_model.predict(data)
    pear = pearsonr(ages_emp_noquantile, pred[1:])
    spear = spearmanr(ages_emp_noquantile, pred[1:])
    r2 = r2_score(ages_emp_noquantile, pred[1:])


    sns.set(style='white')
    if kind=='scatter':
        g = sns.jointplot(x=ages_emp_noquantile, y=pred[1:], kind=kind, height=height, s=4, color=color)

    else:
        g = sns.jointplot(x=ages_emp_noquantile, y=pred[1:], kind=kind, height=height, scatter_kws={'s': scatter_size}, color=color, joint_kws={'line_kws':{'color':line_color}})
    if lim_ax:
        g.ax_joint.set_ylim([0, 80])
        lims = [0, 80] 
    g.set_axis_labels(xlab, ylab, fontsize=fontsize)
    if loc_tick:
        loc = plticker.MultipleLocator(base=tick_step)
        g.ax_joint.xaxis.set_major_locator(loc)
        g.ax_joint.yaxis.set_major_locator(loc)
    if isinstance(tick_step, int):
        g.ax_joint.set_xticklabels([int(tt) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
        g.ax_joint.set_yticklabels([int(tt) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
    else:
        g.ax_joint.set_xticklabels([round(tt,3) for tt in g.ax_joint.get_xticks()], fontsize=fontsize)
        g.ax_joint.set_yticklabels([round(tt,3) for tt in g.ax_joint.get_yticks()], fontsize=fontsize)
    if tight:
        plt.tight_layout()

    plt.savefig(f'{plot_path}{outname}_Prediction.pdf')
    plt.close()

    xlab = 'Ground state values'
    ylab = 'Ground state values + 100x Single cell noise'
    tick_step = 0.25
    sns.set(style='white')

    g = sns.jointplot(x=data.iloc[0], y=data.iloc[-1], kind='scatter', height=height, s=4, color=dot_color)

    g.set_axis_labels(xlab, ylab, fontsize=fontsize)
    loc = plticker.MultipleLocator(base=tick_step)
    g.ax_joint.xaxis.set_major_locator(loc)
    g.ax_joint.yaxis.set_major_locator(loc)
    g.ax_joint.plot([0,1], [0,1], ':k')
    g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=fontsize)
    g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(
        f'{plot_path}{outname}_100xNoise_fontsize{fontsize}_height{height}{filetype}')
    plt.close()
    return pear, spear, r2, pred



def get_clock_correlations(Tissues, prediction_output_file, numsam=5):
    tmp = pd.read_csv(prediction_output_file, index_col=0)
    tmp = tmp.dropna(subset=['Age'])
    tmp.Age = tmp.Age.astype(float)
    tmp = tmp[~tmp.MaxAge.isin(['Cervus canadensis', 'Cervus elaphus'])]  # no proper meta data
    tmp.MaxAge = tmp.MaxAge.astype(float)
    tmp['RelAge'] = tmp.Age / tmp.MaxAge
    tmp = tmp[(tmp.Experiment.isna())]

    df_median_all = pd.DataFrame()
    cl = prediction_output_file.split('/')[-1][:-4]

    pears = {}
    pears['Corr'] = []
    pears['p'] = []
    pears['Tissue'] = []
    pears['Organism'] = []
    pears['NumSamples'] = []

    for s in tmp.Organism.unique():
        for t in Tissues:
            try:
                pears['Corr'].append(pearsonr(tmp[(tmp.Organism == s) & (tmp.Tissue == t)].RelAge, tmp[
                    (tmp.Organism == s) & (tmp.Tissue == t)].clock)[0])
                pears['p'].append(pearsonr(tmp[(tmp.Organism == s) & (tmp.Tissue == t)].RelAge, tmp[
                    (tmp.Organism == s) & (tmp.Tissue == t)].clock)[1])
                pears['NumSamples'].append(len(tmp[(tmp.Organism == s) & (tmp.Tissue == t)]))
                pears['Organism'].append(s)

                pears['Tissue'].append(t)
            except:
                continue

    df = pd.DataFrame(pears)
    df = df.dropna()
    df = df[df.NumSamples >= numsam]
    df['FDR_bh'] = statsmodels.stats.multitest.multipletests(df.p, method='fdr_bh')[1]
    df['Bonferroni'] = statsmodels.stats.multitest.multipletests(df.p, method='Bonferroni')[1]

    df_median = df.groupby('Organism').Corr.median().sort_values()
    df_median = df_median.to_frame()

    maxage = tmp[['Organism', 'Species_Name', 'MaxAge']]
    maxage = maxage.drop_duplicates()
    maxage = maxage.set_index('Organism')

    df_median = df_median.join(maxage)
    df_median = df_median.sort_values(by='MaxAge')

    df_median['Clock'] = cl
    if len(df_median_all) == 0:

        df_median = df_median[['Corr']]
        df_median.columns = [cl]
        df_median_all = df_median
    else:
        df_median = df_median[['Corr']]
        df_median.columns = [cl]
        df_median_all = df_median_all.join(df_median[[cl]], how='outer')


    df_median_all = df_median_all.join(maxage)
    
    df_median_all = df_median_all.sort_values(by='MaxAge')
    df_median_all = df_median_all.dropna()
    
    return df_median_all



def plot_circles(Tissues, prediction_output_file,numsam, outname):
    df_median_all = get_clock_correlations(Tissues=Tissues,prediction_output_file=prediction_output_file, numsam=numsam) 
    # add taxonomy data
    tax = pd.read_csv(f'{input_path}taxonomy_anage_meta.csv', index_col=0)
    df_median_all =df_median_all.join(tax[['Order']])

    # add missing manually
    df_median_all.loc['Notamacropus rufogriseus', 'Order'] = 'Diprotodontia'
    df_median_all.loc['Osphranter rufus', 'Order'] = 'Diprotodontia'
    df_median_all.loc['Panthera leo', 'Order'] = 'Carnivora'
    df_median_all.loc['Equus asinus somalicus', 'Order'] = 'Perissodactyla'
    cl = prediction_output_file.split('/')[-1][:-4]
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    # plot on a circle
    ydata = df_median_all[cl].values
    data_len = len(ydata)
    theta = np.linspace(0, (2 - 2 / len(ydata)) * np.pi, len(ydata))
    r = ydata
    ax.plot(theta, r, 'g')

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    for angle, radius, label in zip(theta, r, df_median_all.Species_Name.values):
        x = angle
        y = 1.3
        ax.text(x, y, label, ha='center',
                fontsize=8, rotation=np.degrees(-angle), va='center')

    ax.set_xticklabels([])
    ax.set_yticklabels([round(x,1) for x in ax.get_yticks()], fontsize=8)
    plt.tight_layout(pad=0.1)
    plt.legend([f'Clock1 Median Correlation: {df_median_all[cl].median():.2f}'], #noise_clock_Empirical_maxage67_l10.001_coefs_
loc='upper right')
    # add the colors
    taxo = {s:i for i,s in enumerate(df_median_all.Order.unique())}
    colors = plt.cm.tab20.colors[:len(taxo)]
    for i in range(len(theta)-1):
        ax.fill_between([theta[i]- np.pi / len(ydata), theta[i + 1]- np.pi / len(ydata)], 0, 0.85, 
                    color=colors[taxo[df_median_all.iloc[i].Order]], alpha=0.5)

    ax.fill_between([ theta[i+1]- np.pi / len(ydata), np.pi*2- np.pi / len(ydata)], 0, 0.85,
                    color=colors[taxo[df_median_all.iloc[i +1].Order]], alpha=0.5)


    plt.savefig(f'{plot_path}CirclePlot_{outname}.pdf')
    plt.close()

    fig, ax = plt.subplots()
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i,0), 1, 1, color=color))
    ax.set_xlim(0,len(colors))
    ax.set_xticks(range(len(colors)))
    ax.set_xticklabels(taxo.keys())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{plot_path}CirclePlot_{outname}_COLORMAP.pdf')
    plt.close()

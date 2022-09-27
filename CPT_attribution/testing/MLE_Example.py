import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from statsmodels import api
#https://analyticsindiamag.com/maximum-likelihood-estimation-python-guide/


def main():
    # # generate an independent variable
    # x = np.linspace(-10, 30, 100)

    # # generate a normally distributed residual
    # e = np.random.normal(10, 5, 100)
    # # generate ground truth
    # y = 10 + 4 * x + e

    # generate an independent variable
    x = np.linspace(-10, 30, 100)

    # generate ground truth
    params0 = [10, 4, 0]
    y0 = sample_model(params0, x, noise=False)
    y = sample_model(params0,x)

    plt.plot(x,y0,label='GT')
    plt.scatter(x,y,label='Samples')
    plt.legend()
    plt.show()
    # df = pd.DataFrame({'x': x, 'y': y})
    # df.head()

    # sns.regplot(x='x', y='y', data=df)
    # plt.show()

    # OLS
    # features = api.add_constant(df.x)
    # model = api.OLS(y, features).fit()
    # model.summary()

    res = model.resid
    standard_dev = np.std(res)
    print(standard_dev)

    # minimize arguments: function, intial_guess_of_parameters, method
    mle_model = minimize(MLE_Norm, np.array([2, 2, 2]), method='L-BFGS-B')
    mle_model['x']
    print(mle_model)

def sample_model(parameters,x,noise=True):
    const, beta, std_dev = parameters
    # generate a normally distributed residual
    e = np.random.normal(10, 5, 100) if noise else 0
    y = const + beta * x + e # sample model/ground truth
    return y

# MLE function
# ml modeling and neg LL calculation
def MLE_Norm(parameters):
    # extract parameters
    const, beta, std_dev = parameters
    # predict the output
    pred = const + beta * x
    # Calculate the log-likelihood for normal distribution
    LL = np.sum(stats.norm.logpdf(y, pred, std_dev))
    # Calculate the negative log-likelihood
    neg_LL = -1 * LL
    return neg_LL



main()
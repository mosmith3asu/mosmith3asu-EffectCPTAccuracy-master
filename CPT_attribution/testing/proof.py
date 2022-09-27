import numpy as np
import matplotlib.pyplot as plt


def main():
    pass
    # N=100
    # p= 0.50
    # res = 50
    # diff = np.zeros(res)
    # Deltas = np.linspace(0, 0.49, res)
    # for i,d in enumerate(Deltas):
    #     LLeps = N * (p + d) * np.log(p + d) + N * (1 - (p + d)) * np.log(1 - (p + d))
    #     LL = N * (p) * np.log(p) + N * (1 - p) * np.log(1 - p)
    #     diff[i] = LLeps - LL
    # plt.plot(Deltas, diff,label='$LL(p_c + \delta) - LL > 0$  $ \\forall \delta > 0 $')
    #
    #
    # Deltas = -1*Deltas#np.linspace(-0.49,0 , res)
    # for i, d in enumerate(Deltas):
    #     LLeps = N * (p + d) * np.log(p + d) + N * (1 - (p + d)) * np.log(1 - (p + d))
    #     LL = N * (p) * np.log(p) + N * (1 - p) * np.log(1 - p)
    #     diff[i] = LLeps - LL
    # plt.plot(Deltas, diff,label='$LL(p_c + \delta) - LL < 0$  $\\forall \delta < 0 $')
    # plt.xlabel('Delta $\delta$')
    # plt.ylabel('Log Diff $LL(p_c + \delta)$')
    # plt.legend()
    # plt.show()

    N = 100
    p = 0.50
    res = 50
    diff = np.zeros(res)
    Deltas = np.linspace(0, 0.49, res)
    for i, d in enumerate(Deltas):
        Leps = np.power(p+d,N*(p+d))*np.power(1-(p+d),N*(1-(p+d)))
        LL = np.power(p,N*p)*np.power(1-p,N*(1-p))
        diff[i] = Leps - LL
    plt.plot(Deltas, diff, label='$L(p_c + \delta) - L > 0$  $ \\forall \delta > 0 $')

    Deltas = -1 * Deltas  # np.linspace(-0.49,0 , res)
    for i, d in enumerate(Deltas):
        Leps = np.power(p + d, N * (p + d)) * np.power(1 - (p + d), N * (1 - (p + d)))
        LL = np.power(p, N * p) * np.power(1 - p, N * (1 - p))
        diff[i] = Leps - LL
    plt.plot(Deltas, diff, label='$L(p_c + \delta) - L < 0$  $ \\forall \delta < 0 $')
    plt.xlabel('Delta $\delta$')
    plt.ylabel('Log Diff $LL(p_c + \delta)$')
    plt.legend()
    plt.show()

    # N=100
    # p= 0.01
    # res = 50
    # diff = np.zeros(res)
    # Deltas = np.linspace(0, 0.99, res)
    # for i,d in enumerate(Deltas):
    #     LLeps = N * (p + d) * np.log(p + d) + N * (1 - (p + d)) * np.log(1 - (p + d))
    #     LL = N * (p) * np.log(p) + N * (1 - p) * np.log(1 - p)
    #     diff[i] = LLeps - LL
    # plt.plot(Deltas, diff,label='$LL(p_c + \delta) - LL > 0$  $ \\forall \delta > 0 $')
    # plt.xlabel('Delta $\delta$')
    # plt.ylabel('Log Diff $LL(p_c + \delta)$')
    # plt.legend()
    # plt.show()



def subfun():
    pass


if __name__ == "__main__":
    main()

"""
Implementation of full GP into a ROS node for predicting oscillations
author: Seungwook Lee
date: 2022-11-30
"""

import rospy
from std_msgs.msg import Float64

# http://krasserm.github.io/2018/03/19/gaussian-processes/

import numpy as np
from gaussian_processes_util import plot_gp, plot_gp_compare, plot_gp_compare6
import gaussian_processes_util
import matplotlib.pyplot as plt
from full_gp_vs_FITC import kernel, posterior_predictive

class GP():
    def __init__(self):
        self.dt = 0.2 # sec
        self.time_window = 5 # sec
        self.num_samples = int(self.time_window / self.dt)
        self.plot = True

        self.X = np.array([]) # timestamp
        self.mu = None # mean of the prior
        self.cov = None # covariance of the prior
        self.noise = None
        self.noise_assumption = None
        self.X_train = np.array([])
        self.Y_train = np.array([])
        self.mu_s = None
        self.cov_s = None
        self.l = None
        self.sigma_f = None
        self.posterior_samples = None
        self.timestamp0 = rospy.Time.now().to_sec()
        
        self.prediction_pub = rospy.Publisher("prediction", Float64 ,queue_size=1)
        self.data_sub = rospy.Subscriber("data", Float64, self.dataSubCallback)

    def setParam(self, l: float, sigma_f: float):
        self.l = l
        self.sigma_f = sigma_f

    def setNoise(self, noise, noise_assumption):
        self.noise = noise # 0.8, 1e-8
        self.noise_assumption = noise_assumption

    def computePrior(self, X):
        self.mu = np.zeros(X.shape)
        self.cov = kernel(X, X)

    def computePosterior(self, X, X_train, Y_train, sigma_y=1):
        self.mu_s, self.cov_s = posterior_predictive(X, X_train, Y_train, sigma_y)

    def computeSamples(self, mu_s, cov_s, n):
        self.posterior_samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, n)

    def dataSubCallback(self, msg):
        # timestamp = msg.header.timestamp.to_sec() # pseudo-code
        timestamp = rospy.Time.now().to_sec() - self.timestamp0
        value = msg.data # pseudo-code
        print(value)
        self.X_train = self.appendAndPop(self.X_train, timestamp, self.num_samples)
        self.Y_train = self.appendAndPop(self.Y_train, value, self.num_samples)
        self.X = self.extrapolate(self.X_train, self.dt, 2*self.num_samples)

    def appendAndPop(self, np_array, value, size):
        np_array = np.append(np_array, np.array([value]))
        while np_array.size > size:
            np_array = np.delete(np_array, 0)
        return np_array.reshape(-1,1)

    def extrapolate(self, np_array, dvalue, size):
        while np_array.size < size and np_array.size > 0:
            np_array = np.append(np_array, np.array([np_array[-1]+dvalue]))
        return np_array.reshape(-1,1)
        
    def getPrediction(self):
        print(f"X size: {self.X.size}")
        if self.X.size > 10:
            self.computePrior(self.X)
            self.computePosterior(self.X, self.X_train, self.Y_train, sigma_y=self.noise_assumption)
            self.computeSamples(self.mu_s, self.cov_s, 1)
            return True
        else:
            return False

    def plot_gp_realtime(mu, cov, X, X_train=None, Y_train=None, samples=[]):
        X = X.ravel()
        mu = mu.ravel()
        uncertainty = 1.96 * np.sqrt(np.diag(cov))
        
        plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
        plt.plot(X, mu, label='Mean')
        for i, sample in enumerate(samples):
            plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
        if X_train is not None:
            plt.plot(X_train, Y_train, 'rx')
        plt.legend()

class Plot():
    def __init__(self):
        self.plt = plt
        self.fig = None
        self.ax  = None

    def initPlot(self, X_axis: str, Y_axis: str, title: str):
        # enable interactive mode
        self.plt.ion()
        
        # creating subplot and figure
        self.fig = self.plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.uncerntainty_plot, = self.ax.plot(0, 0)
        self.mean_plot, = self.ax.plot(0, 0, label='Mean')
        self.sample_plot, = self.ax.plot(0, 0, lw=1, ls='--', label='Sample')
        self.data_plot, = self.ax.plot(0, 0, 'rx')
        
        # setting labels
        self.plt.xlabel(X_axis)
        self.plt.ylabel(Y_axis)
        self.plt.title(title)

    def updatePlot(self, X, mu, cov, X_train, Y_train, samples):
        # updating the value of x and y
        X = X.ravel()
        mu = mu.ravel()
        uncertainty = 1.96 * np.sqrt(np.diag(cov))

        self.plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
        self.mean_plot.set_xdata(X)
        self.mean_plot.set_ydata(mu)

        self.sample_plot.set_xdata(X)
        for i, sample in enumerate(samples):
            self.sample_plot.set_ydata(mu)

        self.data_plot.set_xdata(X_train)
        self.data_plot.set_ydata(Y_train)
    
        # re-drawing the figure
        self.fig.canvas.draw()
        
    def flushPlot(self):
        # to flush the GUI events
        self.fig.canvas.flush_events()

if __name__ == "__main__":
    rospy.init_node("full_gp")

    gp = GP()
    gp.setParam(l=1.0, sigma_f=1.0)
    gp.setNoise(0.1, 1.0)

    plot = Plot()
    plot.initPlot("time (s)", "data", "Gaussian Process Prediction")

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        if gp.getPrediction():
            plot.updatePlot(gp.X, gp.mu_s, gp.cov_s, gp.X_train, gp.Y_train, gp.posterior_samples)
        plot.flushPlot()
        r.sleep()

from scipy import stats
import numpy as np


def hjorth(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided,
           recommended to speed up)
    In case 1, D is computed using Numpy's Difference function.
    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    Parameters
    ----------
    X
        list
        a time series
    D
        list
        first order differential sequence of a time series
    Returns
    -------
    As indicated in return line
    Hjorth mobility and complexity
    """

    if D is None:
        D = np.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = np.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return np.sqrt(M2 / TP), np.sqrt(
        float(M4) * TP / M2 / M2
    )  # Hjorth Mobility and Complexity

def mean(data):
    return np.mean(data,axis=1)
    
def std(data):
    return np.std(data,axis=1)

def ptp(data):
    return np.ptp(data,axis=1)

def var(data):
        return np.var(data,axis=1)

def minim(data):
      return np.min(data,axis=1)


def maxim(data):
      return np.max(data,axis=1)

def argminim(data):
      return np.argmin(data,axis=1)


def argmaxim(data):
      return np.argmax(data,axis=1)

def mean_square(data):
      return np.mean(data**2,axis=1)

def rms(data): #root mean square
      return  np.sqrt(np.mean(data**2,axis=1))  

def abs_diffs_signal(data):
    return np.sum(np.abs(np.diff(data,axis=1)),axis=1)


def skewness(data):
    return stats.skew(data,axis=1)

def kurtosis(data):
    return stats.kurtosis(data,axis=1)

def hjorth_params(data):
  mobility_results = np.zeros((data.shape[0], data.shape[2]))
  complexity_results = np.zeros((data.shape[0], data.shape[2]))
  #  Calculate the Hjorth parameters for each signal for each patient and channel.
  for patient in range(data.shape[0]):
      for channel in range(data.shape[2]):
          signal = data[patient, :, channel]  # Signal from patient i and channel j
          
          # Calculate Hjorth parameters for the current signal
          mobility, complexity = hjorth(signal)        
          # store the parameters in the result ndarray
          mobility_results[patient, channel] = mobility
          complexity_results[patient, channel] = complexity
  return np.array(mobility_results), np.array(complexity_results)

def concatenate_features(data):
    return np.stack((mean(data),std(data) ,kurtosis(data), hjorth_params(data)[0], hjorth_params(data)[1],
                      var(data), ptp(data) 
                          
    ),axis=1)

    #return np.stack((mean(data),std(data),ptp(data),var(data),minim(data),maxim(data),
    #                      mean_square(data),rms(data),abs_diffs_signal(data),
    #                      skewness(data),kurtosis(data), hjorth_params(data)[0], hjorth_params(data)[1]),axis=1)
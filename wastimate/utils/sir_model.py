# Filename: SIR_Final_Rank_Figure.py
# Description: Optional module to plot the
# results of SIR method
# Authors: Papathanasiou, J5.
import matplotlib.pyplot as plt
import numpy as np

# Plot final rank
def plot(a, b, c):
    """Creates and displays a 2D scatter plot with annotated data points.

    This function visualizes a 1D array of "flows" as a scatter plot. Each point
    on the plot is labeled with its flow value and a corresponding label from a
    third input array. The plot is styled with a title, axis labels, and a grid.

    Parameters
    ----------
    a : numpy.ndarray
        A 1D array representing the flow values to be plotted.
    b : str
        A string describing the plotting method, used to set the plot's title.
    c : list or numpy.ndarray
        A 1D array of labels or identifiers for each flow value.

    Returns
    -------
    None
        The function does not return a value but displays a plot and saves it as
        a file.
    """

    flows = a
    yaxes_list = [0.2] * np.size(flows, 0)
    plt.plot(yaxes_list, flows, 'ro')
    
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    
    plt.axis([0, 0.7, min(flows) - 0.05, max(flows) + 0.05])
    plt.title(b + " results")
    plt.ylabel("Flows")
    plt.grid(True)
    
    z1 = []
    for i in range(np.size(flows, 0)):
        z1.append(f'   ({c[i]})')
    z = [str(a) + b for a, b in zip(flows, z1)]
    
    for X, Y, Z in zip(yaxes_list, flows, z):
        plt.annotate('{}'.format(Z), xy = (X, Y), xytext = (10, -4), ha = 'left', textcoords = 'offset points')
        
    plt.show()
    
# Calculate the preference degrees
def pref_func(a, b, c, d, e, m):
    """Calculates the preference degree between two action performances.

    This function implements six different types of preference functions, which
    are commonly used in multi-criteria decision analysis (MCDA), such as 
    PROMETHEE. The preference degree quantifies how much one action's
    performance is preferred over another's for a given criterion.

    Parameters
    ----------
    a : float
        The performance value of the first action.
    b : float
        The performance value of the second action.
    c : float
        The "indifference" threshold (q). A difference smaller than this
        is considered indifferent.
    d : float
        The "preference" threshold (p). A difference larger than this
        results in a full preference.
    e : str
        A single-character string indicating the type of preference function
        to use:
        - 'u' : Usual
        - 'us' : U-shape
        - 'vs' : V-shape
        - 'le' : Level
        - 'li' : Linear
        - 'g' : Gaussian
    m : int
        A flag indicating the direction of optimization. If `m` is 1, it means
        the criterion is to be minimized, so the roles of `a` and `b` are 
        swapped. If `m` is 0, it is a maximization criterion.

    Returns
    -------
    float
        The calculated preference degree, ranging from 0.0 to 1.0.
    """

    f = float(1.0)
    if m == 1:
        temp = a
        a = b
        b = temp
    if e == 'u': # Usual preference function
        if b - a > 0:
            f = 1
        else:
            f = 0
    elif e == 'us': # U-shape preference function
        if b - a > c:
            f = 1
        elif b - a <= c:
            f = 0
    elif e == 'vs': # V-shape preference function
        if b - a > d:
            f = 1
        elif b - a <= 0:
            f = 0
        else:
            f = (b - a) / d
    elif e == 'le': # Level preference function
        if b - a > d:
            f = 1
        elif b - a <= c:
            f = 0
        else:
            f = 0.5
    elif e == 'li': # Linear preference function
        if b - a > d:
            f = 1
        elif b - a <= c:
            f = 0
        else:
            f = ((b - a) - c) / (d - c)
    elif e == 'g': # Gaussian preference function
        if b - a > 0:
            f = 1 - np.math.exp(-(np.math.pow(b - a, 2)
                               / (2 * d ** 2)))
        else:
            f = 0

    return f

# Calculate S and I matrices
def SImatrix(x, p, c, d):
    """Calculates the S (Superiority) and I (Inferiority) matrices.

    This function computes the S and I matrices, which represent the aggregated
    preference for each alternative over all other alternatives across all 
    criteria. 

    Parameters
    ----------
    x : numpy.ndarray
        A 2D array where rows represent alternatives and columns represent
        criteria.
    p : numpy.ndarray
        A 2D array containing the preference parameters (q and p) for all
        criteria.
        The first row contains the 'q' values and the second row contains the
        'p'values for each criterion.
    c : numpy.ndarray
        A 1D array where each element indicates whether a criterion is to be
        minimized (0) or maximized (1).
    d : numpy.ndarray
        A 1D array where each element is a single-character string representing
        the preference function type for a specific criterion (e.g., 'u', 'li',
        'g').

    Returns
    -------
    numpy.ndarray
        A 2D array representing either the S or I matrix.
    """

    SI = np.zeros((np.size(x, 0), np.size(x, 1)))
    for i in range(np.size(x, 1)):
        for j in range(np.size(x, 0)):
            k = 0
            for h in range(np.size(x, 0)):
                k = k + pref_func(x[j, i], x[h, i], p[0, i], p[1, i], d[i], c[i])
                SI[j, i] = k

    return SI

# Calculate S- and I-flow for SIR-SAW
def SIflowsSAW(w, SI):
    """Calculates the S- or I-flow for the SIR-SAW method.

    This function computes the S-flow or I-flow for each alternative using a
    Simple Additive Weighting (SAW) approach. It calculates a weighted sum of
    the preferences for each alternative across all criteria.

    Parameters
    ----------
    w : numpy.ndarray
        A 1D array of weights for each criterion.
    SI : numpy.ndarray
        The S or I matrix.

    Returns
    -------
    numpy.ndarray
        A 1D array representing the S-flow or I-flow for each alternative.
    """

    k = np.zeros(np.size(SI, 0))
    for i in range(np.size(SI, 0)):
        for j in range(np.size(SI, 1)):
            k[i] = k[i] + w[j] * SI[i, j]

    return k

# Calculate SIplus and SIminus for SIR-TOPSIS
def SIRTOPSIS(w, SI, l):
    """Calculates the SIplus and SIminus values for the SIR-TOPSIS method.

    This function is a core part of the SIR-TOPSIS algorithm. It computes the
    `SIplus` and `SIminus` values, which represent the distance of each
    alternative from the "ideal best" and "ideal worst" solutions, respectively.
    It identifies the best and worst performance values for each criterion
    and then calculates the weighted distance of each alternative from these
    ideal points using a specified distance metric.

    Parameters
    ----------
    w : numpy.ndarray
        A 1D array of weights for each criterion.
    SI : numpy.ndarray
        The S or I matrix.
    l : float
        The distance metric to be used (e.g., Euclidean distance where `l=2`).

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two 1D arrays:
        - `SIplus`: The distance of each alternative from the ideal best 
          solution.
        - `SIminus`: The distance of each alternative from the ideal worst
          solution.
    """

    SIplus = np.zeros((np.size(SI, 0)))
    SIminus = np.zeros((np.size(SI, 0)))
    bb = []
    cc = []
    for i in range((np.size(w, 0))):
        bb.append(np.amax(SI[:, i:i + 1]))
        bbb = np.array(bb)
        cc.append(np.amin(SI[:, i:i + 1]))
        ccc = np.array(cc)
    for i in range((np.size(SI, 0))):
        for j in range((np.size(SI, 1))):
            SIplus[i] = SIplus[i] + np.math.pow(w[j]
                                             * abs(SI[i, j] - bbb[j]), l)
            SIminus[i] = SIminus[i] + np.math.pow(w[j]
                                               * abs(SI[i, j] - ccc[j]), l)
        SIplus[i] = np.math.pow(SIplus[i], 1 / l)
        SIminus[i] = np.math.pow(SIminus[i], 1 / l)

    return SIplus, SIminus

# Calculate the S- and I-flow for SIR-TOPSIS
def SIflowsTOPSIS(p, m):
    """Calculates the S- or I-flow for the SIR-TOPSIS method.

    This function computes the final flow score for each alternative based on
    its distance from the ideal best and ideal worst solutions. The flow is a
    ratio that normalizes these distances, where a higher value indicates a
    better performance relative to the ideal best and worst solutions.

    Parameters
    ----------
    p : numpy.ndarray
        A 1D array of `SIplus` values (distance from ideal best).
    m : numpy.ndarray
        A 1D array of `SIminus` values (distance from ideal worst).

    Returns
    -------
    numpy.ndarray
        A 1D array representing the S-flow or I-flow for each alternative.
    """

    return m / (m + p)

# Calculate n-flow
def Nflow(s, i):
    """Calculates the Net-flow (N-flow) for each alternative.

    The Net-flow is a single, aggregated score for each alternative, calculated
    by subtracting its I-flow (inferiority) from its S-flow (superiority). A
    higher Net-flow indicates that an alternative is strongly preferred over
    others and is less inferior to them.

    Parameters
    ----------
    s : numpy.ndarray
        A 1D array representing the S-flow for each alternative.
    i : numpy.ndarray
        A 1D array representing the I-flow for each alternative.

    Returns
    -------
    numpy.ndarray
        A 1D array of Net-flow values.
    """

    return s - i

# Calculate r-flow
def Rflow(s, i):
    """Calculates the Ranking-flow (R-flow) for each alternative.

    The Ranking-flow provides a normalized score for each alternative by
    dividing its S-flow by the sum of its S-flow and I-flow. This value
    ranges between 0 and 1, where a value closer to 1 indicates a
    more favorable ranking.

    Parameters
    ----------
    s : numpy.ndarray
        A 1D array representing the S-flow for each alternative.
    i : numpy.ndarray
        A 1D array representing the I-flow for each alternative.

    Returns
    -------
    numpy.ndarray
        A 1D array of Ranking-flow values.
    """

    return s / (s + i)

def running_mean(x, y, N):
    """Calculates the running mean of a 1D array `x` and truncates `y` to match
    the length.

    This function computes a moving average over a specified window size (`N`)
    for the input array `x`. It then adjusts the length of the `y` array to
    match the new, shorter length of the running mean result, ensuring they can
    be plotted together.

    Parameters
    ----------
    x : numpy.ndarray
        The 1D array for which to compute the running mean.
    y : numpy.ndarray
        The 1D array to be truncated.
    N : int
        The size of the moving window.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two 1D arrays:
        - The new array with the running mean of `x`.
        - The `y` array truncated to the correct length.
    """
    
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    
    x_new = (cumsum[N:] - cumsum[:-N]) / float(N)
    y_new = y[int((len(x)-len(x_new))/2):-int((len(x)-len(x_new))/2)]
    return x_new, y_new


def Score(x, p, c_S, d, w):
    """Calculates the final Ranking-flow (R-flow) score for each alternative.

    This function orchestrates the main steps of the SIR multi-criteria
    decision analysis (MCDA) method. It computes both the Superiority (S) and
    Inferiority (I) matrices, and then uses a specified method (SIR-SAW or
    SIR-TOPSIS) to calculate the S-flow and I-flow. Finally, it combines these
    flows to compute the final R-flow, which provides a comprehensive score
    for ranking the alternatives.

    Parameters
    ----------
    x : numpy.ndarray
        A 2D array of action performances.
    p : numpy.ndarray
        A 2D array of preference parameters (q and p thresholds).
    c_S : numpy.ndarray
        A 1D array indicating maximization (1) or minimization (0) for S-flow
        calculation.
    d : numpy.ndarray
        A 1D array of strings specifying the preference function type for each
        criterion.
    w : numpy.ndarray
        A 1D array of weights for each criterion.

    Returns
    -------
    numpy.ndarray
        A 1D array of Ranking-flow (R-flow) scores for each alternative.
    """
    
    c_I = np.abs(c_S - 1)

    # calculate S and I matrix
    S = SImatrix(x, p, c_S, d)
    I = SImatrix(x, p, c_I, d)

    c = 1
    if c == 1: # SIR-SAW
        # calculate S-flow
        Sflow = SIflowsSAW(w, S)

        # calculate I-flow
        Iflow = SIflowsSAW(w, I)

    else: # SIR-TOPSIS
        # calculate S-flow
        Splus, Sminus = SIRTOPSIS(w, S, 2)
        Sflow = SIflowsTOPSIS(Splus, Sminus)

        # calculate I-flow
        Iplus, Iminus = SIRTOPSIS(w, I, 2)
        Iflow = SIflowsTOPSIS(Iplus, Iminus)

    # calculate r-flow
    rflow = Rflow(Sflow, Iflow)
    
    return rflow

if __name__ == "__main__":
    ##############
    ### INPUTS ###
    ##############
    
    # action performances array - the potential alternatives/options
    
    #              Feasibility, Availability, Complexity and Maintainability, Robustness
    #              Regulation compliance, Competence, Safety,
    #              Site Location, Local Acceptance, International Acceptance,
    #              Overnight cost, cost scaling
    
    x = np.array([[3, 5, 3,  4, 0, 5,   5, 5, 4,   395],  # Very Deep Boreholes
                  [5, 2, 5,  5, 0, 5,   4, 5, 5,   698]]) # Mined deep geological repositories
    
    labels = ["Deep Boreholes",
              "Mined Geological Repository"]
    
    # indifference and absolute preference thresholds
    p = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 10, 10, 10, 10, 10, 10, 10, 10, 1000]])
    
    # criteria min (0) or max (1) optimization array for
    c_S = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    
    # preference function array
    d = (['li', 'li', 'li', 'li', 'li', 'li', 'li', 'li', 'li', 'li'])
    
    # weights of criteria
    w = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    score = Score(x, p, c_S, d, w)
    print(score)
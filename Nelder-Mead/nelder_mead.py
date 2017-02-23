import numpy as np
import matplotlib.pyplot as plt
import operator


# ====================================================
# FUNCTIONS TO MINIMIZE
# ====================================================
def f(p):
    x, y = p[0], p[1]
    return x**2 - 4*x + y**2 - y - x*y

def g(p):
    x, y = p[0], p[1]
    return -np.exp(1. - x**2 - y**2)


# ====================================================
# INITIAL TRIANGLE BGW FUNCTIONS
# ====================================================
def init_rtriangle(x=(0.,1.), y=(0.,1.)):
    xp = (np.max(x) - np.min(x)) * np.random.random(3) + np.min(x)
    yp = (np.max(y) - np.min(y)) * np.random.random(3) + np.min(y)
    return list(zip(xp, yp))

def evaluate(f, V):
    return [ f(V[i]) for i in range(len(V)) ]

def BGW(V, fV):
    T = [ (V[i], fV[i]) for i in range(len(V)) ]
    return sorted(T, key=operator.itemgetter(1))

def print_list(BGW):
    for value in BGW:
        print("Vertex: {} \nValue: {}\n".format(value[0], value[1]))

def update(A, B, C, f):
    V  = [A, B, C]
    fV = evaluate(f, V)
    return BGW(V, fV)


# ====================================================
# * MIDPOINT OF THE GOOD SITE
# * REFLECTION USING THE POINT R
# * EXPANSION USING THE POINT E
# * CONTRACTION USING THE POINT C
# * SHRINK TOWARD B
# ====================================================
def midpoint(BGW, A=0, B=1):
    return ((BGW[A][0][0] + BGW[B][0][0])/2.,
            (BGW[A][0][1] + BGW[B][0][1])/2.)

def reflection(BGW):
    M = midpoint(BGW)
    W = BGW[2][0]
    return (2.*M[0] - W[0], 2.*M[1] - W[1])

def expansion(BGW):
    M = midpoint(BGW)
    R = reflection(BGW)
    return (2.*R[0] - M[0], 2.*R[1] - M[1])

def contraction(BGW, f):
    M = midpoint(BGW)
    R = reflection(BGW)
    W = BGW[2][0]
    C1 = ((R[0] + M[0])/2., (R[1] + M[1])/2.)
    C2 = ((W[0] + M[0])/2., (W[1] + M[1])/2.)
    if f(C1) <= f(C2):
        return C1
    return C2

def shrink(BGW, f):
    M, S  = midpoint(BGW), midpoint(BGW, A=0, B=2)
    return M, S


# ====================================================
# NELDER-MEAD ALGORITHM
# ====================================================
def nelder_mead(BGW, f, iterations, plot=False):
    print("k \t {:^20}   \t {:^20}   \t {:^20}".format("Best point", "Good point", "Worst point"))
    for i in range(iterations):
        print("{} \t f({:.2f},{:.2f}) = {:.2f}   \t f({:.2f},{:.2f}) = {:.2f}   \t f({:.2f},{:.2f}) = {:.2f}"
                .format(i+1, BGW[0][0][0], BGW[0][0][1], f(BGW[0][0]),
                        BGW[1][0][0], BGW[1][0][1], f(BGW[1][0]),
                        BGW[2][0][0], BGW[2][0][1], f(BGW[2][0])))

        R, G = reflection(BGW), BGW[1][0]
        if f(R) < f(G):
            # Case (i):
            B = BGW[0][0]
            if f(B) < f(R):
                BGW = update(B, G, R, f)
            else:
                E = expansion(BGW)
                if f(E) < f(B):
                    BGW = update(B, G, E, f)
                else:
                    BGW = update(B, G, R, f)
        else:
            # Case (ii):
            B = BGW[0][0]
            W = BGW[2][0]
            if f(R) < f(W):
                BGW = update(B, G, R, f)
            C = contraction(BGW, f)
            W = BGW[2][0]
            if f(C) < f(W):
                BGW = update(B, G, C, f)
            else:
                M, S = shrink(BGW, f)
                BGW = update(B, M, S, f)

        if plot:
            bgw = np.array([BGW[i][0] for i in range(len(BGW))])
            plt.clf()
            plt.contourf(X, Y, Z)
            plt.plot((bgw[0,0], bgw[1,0]), (bgw[0,1], bgw[1,1]), '-c')
            plt.plot((bgw[2,0], bgw[1,0]), (bgw[2,1], bgw[1,1]), '-c')
            plt.plot((bgw[0,0], bgw[2,0]), (bgw[0,1], bgw[2,1]), '-c')
            plt.plot(bgw[0,0], bgw[0,1], '.r')
            plt.savefig("{}.png".format(i+1))
            plt.show(block=False)
            plt.pause(0.1)
    plt.show()
    plt.imsave("{}.png".format(i+1))


# ====================================================
#                   ----- MAIN -----
# ====================================================
if __name__ == "__main__":
    # Generate initial triangle and evaluate f on it:
    V  = [(0., 0.), (1.2, 0.), (0., 0.8)]
    # V  = init_rtriangle((-1.,1.), (-1.,1.))
    fV = evaluate(f, V)
    T  = BGW(V, fV)
    # print_list(T)

    # Create domain for plotting:
    X = np.linspace(0, 5, 1e3)
    Y = np.linspace(0, 5, 1e3)
    X, Y = np.meshgrid(X, Y)
    Z = X**2 - 4*X + Y**2 - Y - X*Y
    # X = np.linspace(-1, 1, 1e3)
    # Y = np.linspace(-1, 1, 1e3)
    # X, Y = np.meshgrid(X, Y)
    # Z = -np.exp(1. - X**2 - Y**2)

    # Nealder-Mead Algorithm:
    nelder_mead(T, f, 15, True)

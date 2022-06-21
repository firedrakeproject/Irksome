# Linear basis functions
def linear_0(tau):
    return 1 - tau

def linear_1(tau):
    return tau

# Quadratic basis functions
def quadratic_0(tau):
    return 2*tau**2 - 3*tau + 1

def quadratic_1(tau):
    return -4*tau**2 + 4*tau

def quadratic_2(tau):
    return 2*tau**2 - tau

# Cubic basis functions
def cubic_0(tau):
    return -9/2*tau**3 + 9*tau**2 - 11/2*tau + 1

def cubic_1(tau):
    return 27/2*tau**3 - 45/2*tau**2 + 9*tau

def cubic_2(tau):
    return -27/2*tau**3 + 18*tau**2 - 9/2*tau

def cubic_3(tau):
    return 9/2*tau**3 - 9/2*tau**2 + tau


# Dictionary collecting the lagrange basis functions on [0, 1] for error
# norm testing in spacetime
lagrange_bases = {"linear": [linear_0, linear_1],
                  "quadratic": [quadratic_0, quadratic_1, quadratic_2],
                  "cubic": [cubic_0, cubic_1, cubic_2, cubic_3]}
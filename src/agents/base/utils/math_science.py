import math

# --- Core Mathematical & Scientific Components ---
def sigmoid(x: float) -> float:
    """f(x) = 1 / (1 + e^(-x))"""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def sigmoid_derivative(x: float) -> float:
    """f'(x) = sigmoid(x) * (1 - sigmoid(x))"""
    s_x = sigmoid(x)
    return s_x * (1 - s_x)

def relu(x: float) -> float:
    """f(x) = max(0, x)"""
    return max(0.0, x)

def relu_derivative(x: float) -> float:
    """f'(x) = 1 if x > 0 else 0"""
    return 1.0 if x > 0 else 0.0

def tanh(x: float) -> float:
    """f(x) = (e^x - e^-x) / (e^x + e^-x)"""
    try:
        return math.tanh(x)
    except OverflowError:
        return -1.0 if x < 0 else 1.0

def tanh_derivative(x: float) -> float:
    """f'(x) = 1 - tanh(x)^2"""
    tanh_x = tanh(x)
    return 1.0 - tanh_x**2

def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """f(x) = x if x > 0 else alpha * x"""
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x: float, alpha: float = 0.01) -> float:
    """f'(x) = 1 if x > 0 else alpha"""
    return 1.0 if x > 0 else alpha

def elu(x: float, alpha: float = 1.0) -> float:
    """Exponential Linear Unit: f(x) = x if x > 0 else alpha*(e^x - 1)"""
    return x if x > 0 else alpha * (math.exp(x) - 1)

def elu_derivative(x: float, alpha: float = 1.0) -> float:
    """f'(x) = 1 if x > 0 else elu(x) + alpha"""
    return 1.0 if x > 0 else elu(x, alpha) + alpha

def swish(x: float) -> float:
    """Swish activation: f(x) = x * sigmoid(x)"""
    return x * sigmoid(x)

def swish_derivative(x: float) -> float:
    """Swish derivative: f'(x) = sigmoid(x) + x * sigmoid(x)*(1 - sigmoid(x))"""
    sig = sigmoid(x)
    return sig + x * sig * (1 - sig)

def softmax(x: list) -> list:
    """Softmax function for numerical stability"""
    max_x = max(x)
    e_x = [math.exp(num - max_x) for num in x]
    sum_e_x = sum(e_x)
    return [num / sum_e_x for num in e_x]

# --- Loss Functions & Derivatives ---
def mse(y_true: list, y_pred: list) -> float:
    """Mean Squared Error"""
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists must have the same length.")
    return sum((t - p)**2 for t, p in zip(y_true, y_pred)) / len(y_true)

def mse_derivative(y_true: list, y_pred: list) -> list:
    """Derivative of MSE with respect to y_pred"""
    return [2*(p - t)/len(y_true) for t, p in zip(y_true, y_pred)]

def cross_entropy(y_true: list, y_pred: list, epsilon: float = 1e-12) -> float:
    """Cross Entropy Loss (log loss) for binary and multi-class classification"""
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists must have the same length.")
    total_loss = 0.0
    for t, p in zip(y_true, y_pred):
        # Clip predictions to avoid log(0)
        p_clipped = max(min(p, 1 - epsilon), epsilon)
        total_loss += t * math.log(p_clipped) + (1 - t) * math.log(1 - p_clipped)
    return -total_loss / len(y_true)  # Average over all samples

def cross_entropy_derivative(y_true: list, y_pred: list) -> list:
    """Derivative of cross-entropy loss with respect to pre-activation (z) for sigmoid output"""
    return [p - t for t, p in zip(y_true, y_pred)]

# --- Numerical Methods ---
def newton_raphson(f, df, x0: float, tol: float = 1e-6, max_iter: int = 100) -> float:
    """Find root using Newton-Raphson method"""
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = df(x)
        if dfx == 0:
            break
        x -= fx / dfx
    return x  # May not converge

def euler_method(f, t0: float, y0: float, h: float, t_end: float) -> list:
    """Solve ODE using Euler's method (dy/dt = f(t, y))"""
    t, y = t0, y0
    result = [(t, y)]
    while t < t_end:
        y += h * f(t, y)
        t += h
        result.append((t, y))
    return result

def simpsons_rule(f, a: float, b: float, n: int) -> float:
    """Numerical integration using Simpson's 1/3 rule"""
    if n % 2 != 0:
        raise ValueError("n must be even.")
    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n):
        x = a + i*h
        integral += 4 * f(x) if i % 2 else 2 * f(x)
    return integral * h / 3

# --- Physics Equations ---
G = 6.67430e-11  # Gravitational constant (m³ kg⁻¹ s⁻²)

def kinetic_energy(mass: float, velocity: float) -> float:
    """Kinetic energy: ½mv²"""
    return 0.5 * mass * velocity ** 2

def gravitational_force(m1: float, m2: float, distance: float) -> float:
    """Newton's law of universal gravitation"""
    return G * m1 * m2 / (distance ** 2)

def ohms_law(voltage: float = None, current: float = None, resistance: float = None) -> float:
    """Ohm's Law: V = IR (provide any two parameters)"""
    provided = sum(1 for param in [voltage, current, resistance] if param is not None)
    if provided != 2:
        raise ValueError("Exactly two parameters must be provided")
    
    if voltage is None:
        return current * resistance
    elif current is None:
        return voltage / resistance
    else:
        return voltage / current

# --- Fundamental Math Operations ---
def quadratic_formula(a: float, b: float, c: float) -> tuple:
    """Solve ax² + bx + c = 0"""
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return (None, None)
    sqrt_d = math.sqrt(discriminant)
    return ((-b + sqrt_d)/(2*a), (-b - sqrt_d)/(2*a))

def factorial(n: int) -> int:
    """Iterative factorial implementation"""
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

def standard_deviation(data: list, population: bool = False) -> float:
    """Calculate standard deviation (sample by default)"""
    n = len(data)
    if n < 2:
        raise ValueError("Data must contain at least two elements")
    mean = sum(data)/n
    variance = sum((x - mean)**2 for x in data)/(n - (0 if population else 1))
    return math.sqrt(variance)

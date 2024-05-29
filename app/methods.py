from sympy import Symbol, sympify, Abs, diff
from scipy.interpolate import interp1d, CubicSpline
import numpy as np
import sympy
from sympy import symbols, sympify
from sympy import Symbol, sympify, Abs

def biseccion(function_expr, Tol, Niter, a, b):
    x = symbols('x')
    function = sympify(function_expr)

    f_a = function.evalf(subs={x: a})
    f_b = function.evalf(subs={x: b})
    tolerance = 0.5 * 10 ** -Tol

    output = {
        "columns": ["iter", "a", "xm", "b", "f(xm)", "E"],
        "iterations": Niter,
        "errors": list(),
        "results": list(),
        "root": None,
        "x_vals": [],
        "y_vals": []
    }

    if f_a == 0:
        output["root"] = a
        output["results"].append([0, a, a, b, f_a, 0])
        return output
    elif f_b == 0:
        output["root"] = b
        output["results"].append([0, a, b, b, f_b, 0])
        return output
    elif f_a * f_b > 0:
        output["errors"].append("The interval is inadequate")
        return output

    x_values = []
    f_values = []
    errors = []
    iterations = []

    iteration_count = 0
    lower = a
    upper = b
    mid_point = (lower + upper) / 2.0
    f_mid = function.evalf(subs={x: mid_point})
    x_values.append(mid_point)
    f_values.append(f_mid)
    errors.append(100.0)  # Initial error set to 100%
    iterations.append(iteration_count)

    previous_mid_point = mid_point

    while iteration_count < Niter:
        iteration_count += 1

        tempX = function.evalf(subs={x: lower}) * f_mid

        if tempX < 0 or tempX == 0:
            upper = mid_point
        else:
            lower = mid_point

        mid_point = (lower + upper) / 2.0
        f_mid = function.evalf(subs={x: mid_point})

        error = abs(mid_point - previous_mid_point) if Tol == 1 else abs(mid_point - previous_mid_point) / abs(mid_point)
        x_values.append(mid_point)
        f_values.append(f_mid)
        errors.append(error)
        iterations.append(iteration_count)

        output["results"].append([iteration_count, lower, mid_point, upper, f_mid, error])

        if error < tolerance:
            break

        previous_mid_point = mid_point

    message = f"{mid_point} is a root of f(x)" if f_mid == 0 else f"The approximate solution is: {mid_point}, with a tolerance = {tolerance}" if errors[-1] < tolerance else f"Failed in {Niter} iterations"
    output["root"] = mid_point
    output["message"] = message
    output["x_vals"] = x_values
    output["y_vals"] = f_values

    return output

def puntoFijo(X0, Tol, Niter, fx, gx):
    output = {
        "columns": ["iter", "xi", "g(xi)", "f(xi)", "E"],
        "iterations": Niter,
        "errors": list()
    }

    # configuración inicial
    datos = list()
    x = symbols('x')
    i = 1
    Tol = float(Tol)
    error = 1.000

    Fx = sympify(fx)
    Gx = sympify(gx)

    # Iteración 0
    xP = X0  # Valor inicial (Punto de evaluación)
    xA = 0.0

    Fa = Fx.subs(x, xP)  # Función evaluada en el valor inicial
    Fa = Fa.evalf()

    Ga = Gx.subs(x, xP)  # Función G evaluada en el valor inicial
    Ga = Ga.evalf()

    datos.append([0, '{:^15.7f}'.format(float(xA)), '{:^15.7f}'.format(
        float(Ga)), '{:^15.7E}'.format(float(Fa))])
    try:
        while (error > Tol) and (i < Niter):  # Se repite hasta que el error sea menor a la tolerancia
            # Se evalúa el valor inicial en G, para posteriormente evaluar este valor en la función F siendo-> Xn=G(x) y F(xn) = F(G(x))
            Ga = Gx.subs(x, xP)  # Función G evaluada en el punto inicial
            xA = Ga.evalf()

            Fa = Fx.subs(x, xA)  # Función evaluada en el valor de la evaluación de G
            Fa = Fa.evalf()

            error = Abs(xA - (xP))  # Se calcula el error

            xP = xA  # Nuevo punto de evaluación (Punto inicial)

            datos.append([i, '{:^15.7f}'.format(float(xA)), '{:^15.7f}'.format(
                float(Ga)), '{:^15.7E}'.format(float(Fa)), '{:^15.7E}'.format(float(error))])

            i += 1

    except BaseException as e:
        output["errors"].append("Error in data: " + str(e))
        return output

    output["results"] = datos
    output["root"] = xA
    return output

def newton(x0, Tol, Niter, fx, df):
    output = {
        "columns": ["N", "xi", "F(xi)", "E"],
        "errors": list()
    }

    # configuración inicial
    datos = list()
    x = symbols('x')
    Fun = sympify(fx)
    DerF = sympify(df)

    xn = []
    derf = []
    xi = x0  # Punto de inicio
    f = Fun.evalf(subs={x: x0})  # función evaluada en x0
    derivada = DerF.evalf(subs={x: x0})  # función derivada evaluada en x0
    c = 0
    Error = 100
    xn.append(xi)

    try:
        datos.append([c, '{:^15.7f}'.format(x0), '{:^15.7f}'.format(f)])

        # Al evaluar la derivada en el punto inicial, se busca que sea diferente de 0, ya que al serlo nos encontramos en un punto de inflexión
        #(No se puede continuar ya que la tangente es horizontal)
        while Error > Tol and f != 0 and derivada != 0 and c < Niter:  # El algoritmo converge o se alcanzó límite de iteraciones fijado

            xi = xi - f / derivada  # Estimación del siguiente punto aproximado a la raíz (nuevo valor inicial)
            derivada = DerF.evalf(subs={x: xi})  # Evaluación de la derivada con el nuevo valor inicial (xi)
            f = Fun.evalf(subs={x: xi})  # Evaluación de la función con el nuevo valor inicial (xi)
            xn.append(xi)
            c = c + 1
            Error = abs(xn[c] - xn[c - 1])  # Se reduce entre cada iteración (Representado por el tramo)
            derf.append(derivada)
            datos.append([c, '{:^15.7f}'.format(float(xi)), '{:^15.7E}'.format(
                float(f)), '{:^15.7E}'.format(float(Error))])

    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))
        return output

    output["results"] = datos
    output["root"] = xi
    return output

def reglaFalsa(a, b, Niter, Tol, fx):
    output = {
        "columns": ["iter", "a", "xm", "b", "f(xm)", "E"],
        "iterations": Niter,
        "errors": list()
    }

    # configuración inicial
    datos = list()
    x = symbols('x')
    i = 1
    cond = Tol
    error = 1.0000000

    Fun = sympify(fx)

    xm = 0
    xm0 = 0
    Fx_2 = 0
    Fx_3 = 0
    Fa = 0
    Fb = 0

    try:
        while (error > cond) and (i < Niter):
            if i == 1:
                Fx_2 = Fun.subs(x, a)
                Fx_2 = Fx_2.evalf()
                Fa = Fx_2

                Fx_2 = Fun.subs(x, b)
                Fx_2 = Fx_2.evalf()
                Fb = Fx_2

                xm = (Fb * a - Fa * b) / (Fb - Fa)
                Fx_3 = Fun.subs(x, xm)
                Fx_3 = Fx_3.evalf()
                datos.append([i, '{:^15.7f}'.format(a), '{:^15.7f}'.format(xm), '{:^15.7f}'.format(b), '{:^15.7E}'.format(Fx_3)])
            else:
                if (Fa * Fx_3 < 0):
                    b = xm
                else:
                    a = xm

                xm0 = xm
                Fx_2 = Fun.subs(x, a)  # Función evaluada en a
                Fx_2 = Fx_2.evalf()
                Fa = Fx_2

                Fx_2 = Fun.subs(x, b)  # Función evaluada en b
                Fx_2 = Fx_2.evalf()
                Fb = Fx_2

                xm = (Fb * a - Fa * b) / (Fb - Fa)  # Calcular intersección en la recta en el eje x

                Fx_3 = Fun.subs(x, xm)  # Función evaluada en xm (f(xm))
                Fx_3 = Fx_3.evalf()

                error = Abs(xm - xm0)
                er = sympify(error)
                error = er.evalf()
                datos.append([i, '{:^15.7f}'.format(a), '{:^15.7f}'.format(xm), '{:^15.7f}'.format(b), '{:^15.7E}'.format(Fx_3), '{:^15.7E}'.format(error)])
            i += 1
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))

        return output

    output["results"] = datos
    output["root"] = xm
    return output

def secante(fx, tol, Niter, x0, x1):
    output = {
        "columns": ["iter", "xi", "f(xi)", "E"],
        "errors": list()
    }

    results = list()
    x = symbols('x')
    i = 0
    cond = tol
    error = 1.0000000

    Fun = sympify(fx)

    y = x0
    Fx0 = Fun
    Fx1 = Fun

    try:
        while (error > cond) and (i < Niter):  # criterios de parada
            if i == 0:
                Fx0 = Fun.subs(x, x0)  # Evaluacion en el valor inicial X0
                Fx0 = Fx0.evalf()
                results.append([i, '{:^15.7f}'.format(float(x0)), '{:^15.7E}'.format(float(Fx0))])
            elif i == 1:
                Fx1 = Fun.subs(x, x1)  # Evaluacion en el valor inicial X1
                Fx1 = Fx1.evalf()
                results.append([i, '{:^15.7f}'.format(float(x1)), '{:^15.7E}'.format(float(Fx1))])
            else:
                y = x1
                # Se calcula la secante
                x1 = x1 - (Fx1 * (x1 - x0) / (Fx1 - Fx0))  # Punto de corte del intervalo usando la raiz de la secante, (xi+1)
                x0 = y

                Fx0 = Fun.subs(x, x0)  # Evaluacion en el valor inicial X0
                Fx0 = Fx0.evalf()

                Fx1 = Fun.subs(x, x1)  # Evaluacion en el valor inicial X1
                Fx1 = Fx1.evalf()

                error = Abs(x1 - x0)  # Tramo

                results.append([i, '{:^15.7f}'.format(float(x1)), '{:^15.7E}'.format(float(Fx1)), '{:^15.7E}'.format(float(error))])
            i += 1
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))

        return output

    output["results"] = results
    output["root"] = y
    return output

def raicesMultiples(fx, x0, tol, niter):
    output = {
        "columns": ["iter", "xi", "f(xi)", "E"],
        "iterations": niter,
        "errors": list()
    }

    # Configuraciones iniciales
    results = list()
    x = symbols('x')
    cond = tol
    error = 1.0000
    ex = sympify(fx)

    d_ex = diff(ex, x)  # Primera derivada de Fx
    d2_ex = diff(d_ex, x)  # Segunda derivada de Fx

    xP = x0
    ex_2 = ex.subs(x, x0)  # Función evaluada en x0
    ex_2 = ex_2.evalf()

    d_ex2 = d_ex.subs(x, x0)  # Primera derivada evaluada en x0
    d_ex2 = d_ex2.evalf()

    d2_ex2 = d2_ex.subs(x, x0)  # Segunda derivada evaluada en x0
    d2_ex2 = d2_ex2.evalf()

    i = 0
    results.append([i, '{:^15.7E}'.format(x0), '{:^15.7E}'.format(ex_2)])  # Datos con formato dado
    try:
        while (error > cond) and (i < niter):  # Se repite hasta que el intervalo sea lo pequeño que se desee
            if i == 0:
                ex_2 = ex.subs(x, xP)  # Función evaluada en valor inicial
                ex_2 = ex_2.evalf()
            else:
                d_ex2 = d_ex.subs(x, xP)  # Función evaluada en valor inicial
                d_ex2 = d_ex2.evalf()

                d2_ex2 = d2_ex.subs(x, xP)  # Función evaluada en valor inicial
                d2_ex2 = d2_ex2.evalf()

                xA = xP - (ex_2 * d_ex2) / ((d_ex2) ** 2 - ex_2 * d2_ex2)  # Método de Newton-Raphson modificado

                ex_A = ex.subs(x, xA)  # Función evaluada en xA
                ex_A = ex_A.evalf()

                error = Abs(xA - xP)
                error = error.evalf()  # Se calcula el error
                er = error

                ex_2 = ex_A  # Se establece la nueva aproximación
                xP = xA

                results.append([i, '{:^15.7E}'.format(float(xA)), '{:^15.7E}'.format(
                    float(ex_2)), '{:^15.7E}'.format(float(er))])
            i += 1
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))

        return output

    output["results"] = results
    output["root"] = xA
    return output

#Metodos iterativos
def toMatrix(matrix_str):
    return np.array(eval(matrix_str))

def toVector(vector_str):
    return np.array(eval(vector_str))

def jacobi(Ma, Vb, x0, tol, niter):
    output = {
        "iterations": niter,
        "errors": list(),
    }

    A = np.matrix(Ma)
    sX = np.size(x0)
    xA = np.zeros((sX, 1))

    b = np.array(Vb)
    s = b.size
    b = np.reshape(b, (s, 1)) # Rehace el tamaño del vector b

    D = np.diag(np.diag(A)) # Saca la diagonal de la matriz A
    L = -1 * np.tril(A) + D # Saca la matriz Lower de la matriz A
    U = -1 * np.triu(A) + D # Saca la matriz Upper de la matriz A
    LU = L + U

    T = np.linalg.inv(D) @ LU # Obtiene la matriz de Transición multiplicando el inverso de D por la matriz LU
    tFinal = max(abs(np.linalg.eigvals(T)))
    C = np.linalg.inv(D) @ b # Obtiene la matriz de coeficientes multiplicando el inverso de la matriz de D por la matriz b

    output["t"] = T
    output["c"] = C

    xP = x0
    E = 1000
    cont = 0

    steps = {'Step 0': np.copy(xA)}
    try:
        while (E > tol and cont < niter):
            xA = T @ xP + C
            E = np.linalg.norm(xP - xA)
            xP = xA
            cont = cont + 1
            steps[f'Step {cont}'] = np.copy(xA)

    except Exception as e:
        output["errors"].append(str(e))
        return output

    output["steps"] = steps
    output["spectral_radius"] = tFinal
    output["root"] = xA
    return output

def gaussSeidel(Ma, Vb, x0, tol, niter):
    iteraciones = []
    informacion = []
    error = []

    sX = np.size(x0)
    xA = np.zeros((sX, 1))

    A = np.matrix(Ma)

    b = np.array(Vb)
    s = b.size
    b = np.reshape(b, (s, 1))  # Rehace el tamaño del vector b

    D = np.diag(np.diag(A))  # Saca la diagonal de la matriz A
    L = -1 * np.tril(A) + D  # Saca la matriz Lower de la matriz A
    U = -1 * np.triu(A) + D  # Saca la matriz Upper de la matriz A

    T = np.linalg.inv(D - L) @ U  # Obtiene la matriz de Transición multiplicando el inverso de D-L por la matriz U
    tFinal = max(abs(np.linalg.eigvals(T)))
    C = np.linalg.inv(D - L) @ b  # Obtiene la matriz Coeficientes multiplicando el inverso de D-L por la matriz b

    xP = x0
    E = 1000
    cont = 0

    steps = {'Step 0': np.copy(xA)}
    try:
        while E > tol and cont < niter:
            xA = T @ xP + C
            E = np.linalg.norm(xP - xA)
            xP = xA
            cont += 1
            steps[f'Step {cont}'] = np.copy(xA)
            print(xA[:, 0])

    except Exception as e:
        return {"errors": [str(e)]}

    resultado = {
        "t": T,
        "c": C,
        "esp": tFinal,
        "steps": steps,
        "errors": []
    }
    return resultado

def sor(Ma, Vb, x0, w, tol, niter):
    A = np.matrix(Ma)
    b = np.array(Vb)
    s = b.size
    b = np.reshape(b, (s, 1))  # Rehace el tamaño del vector b

    D = np.diag(np.diag(A))  # Saca la diagonal de la matriz A
    L = -1 * np.tril(A) + D  # Saca la matriz Lower de la matriz A
    U = -1 * np.triu(A) + D  # Saca la matriz Upper de la matriz A

    T = np.linalg.inv(D - L) @ U  # Obtiene la matriz de Transición multiplicando el inverso de D-L por la matriz U
    tFinal = max(abs(np.linalg.eigvals(T)))
    C = np.linalg.inv(D - L) @ b  # Obtiene la matriz Coeficientes multiplicando el inverso de D-L por la matriz b

    iteraciones = []
    informacion = []
    cumple = False
    n = len(Ma)
    k = 0
    errores = []

    try:
        while not cumple and k < niter:
            xk1 = np.zeros(n)
            for i in range(n):
                s1 = np.dot(Ma[i][:i], xk1[:i])  # Multiplica los valores de la Matriz A hasta el final de la matriz xk1
                s2 = np.dot(Ma[i][i+1:], x0[i+1:])  # Multiplica la matrizA con el vector de inicio
                xk1[i] = (Vb[i] - s1 - s2) / Ma[i][i] * w + (1 - w) * x0[i]  # Hace las operaciones para obtener el resultado del metodo
            norma = np.linalg.norm(x0 - xk1)
            x0 = xk1  # Actualiza los valores para el próximo ciclo
            iteraciones.append(k)
            informacion.append(np.copy(xk1))
            errores.append(norma)
            cumple = norma < tol
            k += 1
    except Exception as e:
        return {"errors": [str(e)]}

    if k < niter:
        datos = list(zip(iteraciones, informacion))  # Guarda el contador e información
        resultado = {
            "solucion": x0,
            "t": T,
            "c": C,
            "esp": tFinal,
            "informacion": datos,
            "errores": errores,
            "errors": []
        }
        return resultado
    else:
        return {"errors": ["El sistema no converge"]}

#Metodos interpolacion

def toVector(vector_str):
    return [float(i) for i in vector_str.split(",")]

def splineLineal(X, Y):
    output = {
        "errors": list(),
        "results": None,
        "tracers": None
    }

    try:
        X = np.array(X)
        Y = np.array(Y)

        # Verifica que haya suficientes puntos para una interpolación lineal
        if len(X) < 2:
            output["errors"].append(
                "Se requieren al menos 2 puntos para una interpolación lineal.")
            return output

        # Realiza la interpolación lineal
        linear_interpolation = interp1d(X, Y, kind='linear')

        # Genera puntos de muestra para graficar el polinomio interpolado
        x_vals = np.linspace(min(X), max(X), 500)
        y_vals = linear_interpolation(x_vals)

        # Calcula los coeficientes del polinomio
        coef = []
        for i in range(len(X) - 1):
            slope = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])
            intercept = Y[i] - slope * X[i]
            coef.append([slope, intercept])

        # Formatea los polinomios para mostrarlos
        tracers = [f"S{i}(x) = {slope:.2f}*x + {intercept:.2f}" for
                   i, (slope, intercept) in enumerate(coef)]

        # Guarda los resultados en el output
        output["results"] = (X, Y, x_vals, y_vals)
        output["coef"] = coef
        output["tracers"] = tracers
    except Exception as e:
        output["errors"].append("Error in data: " + str(e))

    return output

def splineCuadratica(X, Y):
    output = {
        "errors": list(),
        "results": None,
        "tracers": None
    }

    try:
        X = np.array(X)
        Y = np.array(Y)

        # Verifica que haya suficientes puntos para una interpolación cuadrática
        if len(X) < 3:
            output["errors"].append(
                "Se requieren al menos 3 puntos para una interpolación cuadrática.")
            return output

        # Realiza la interpolación cuadrática
        quadratic_interpolation = interp1d(X, Y, kind='quadratic')

        # Genera puntos de muestra para graficar el polinomio interpolado
        x_vals = np.linspace(min(X), max(X), 500)
        y_vals = quadratic_interpolation(x_vals)

        # Calcula los coeficientes del polinomio
        coef = []
        for i in range(len(X) - 1):
            x_section = X[i:i + 3]
            y_section = Y[i:i + 3]
            poly = np.polyfit(x_section, y_section, 2)
            coef.append(poly)

        # Formatea los polinomios para mostrarlos
        tracers = [
            f"S{i}(x) = {poly[0]:.2f}*x^2 + {poly[1]:.2f}*x + {poly[2]:.2f}" for
            i, poly in enumerate(coef)]

        # Guarda los resultados en el output
        output["results"] = (X, Y, x_vals, y_vals)
        output["coef"] = coef
        output["tracers"] = tracers
    except Exception as e:
        output["errors"].append("Error in data: " + str(e))

    return output

def splineCubica(X, Y):
    output = {
        "errors": list(),
        "results": None,
        "tracers": None
    }

    try:
        X = np.array(X)
        Y = np.array(Y)

        # Verifica que haya suficientes puntos para una interpolación cúbica
        if len(X) < 4:
            output["errors"].append(
                "Se requieren al menos 4 puntos para una interpolación cúbica.")
            return output

        # Realiza la interpolación cúbica
        cs = CubicSpline(X, Y, bc_type='natural')

        # Genera puntos de muestra para graficar el polinomio interpolado
        x_vals = np.linspace(min(X), max(X), 500)
        y_vals = cs(x_vals)

        # Calcula los coeficientes del polinomio
        coef = cs.c.T  # Coeficientes de los polinomios

        # Formatea los polinomios para mostrarlos
        tracers = [
            f"S{i}(x) = {coef[i, 0]:.2f}*x^3 + {coef[i, 1]:.2f}*x^2 + {coef[i, 2]:.2f}*x + {coef[i, 3]:.2f}"
            for i in range(len(coef))]

        # Guarda los resultados en el output
        output["results"] = (X, Y, x_vals, y_vals)
        output["coef"] = coef
        output["tracers"] = tracers
    except Exception as e:
        output["errors"].append("Error in data: " + str(e))

    return output

def vandermonde(a, b):
    copiaB = np.copy(b)
    longitudMatriz = len(a)
    matrizVandermonde = np.vander(a)  # Obtiene la matriz vandermonde con la matriz A
    coeficientes = np.linalg.solve(matrizVandermonde, copiaB)  # Encuentra la Matriz A con vector B

    x = sympy.Symbol('x')
    polinomio = 0
    for i in range(0, longitudMatriz, 1):  # Ciclo para asignarle las x y las potencias al polinomio
        potencia = (longitudMatriz - 1) - i
        termino = coeficientes[i] * (x ** potencia)
        polinomio = polinomio + termino

    datos = {
        "matriz": matrizVandermonde,
        "coeficientes": coeficientes,
        "polinomio": polinomio,
    }

    return datos

def newtonInter(X, Y):
    output = {}

    try:
        X = np.array(X)
        n = X.size
        Y = np.array(Y)
        D = np.zeros((n, n))
        D[:, 0] = Y.T

        for i in range(1, n):
            aux0 = D[i - 1:n, i - 1]
            aux = np.diff(aux0)
            aux2 = X[i:n] - X[0:n - i]
            D[i:n, i] = aux / aux2.T

        Coef = np.diag(D)
        output["D"] = D
        output["Coef"] = Coef
    except Exception as e:
        output["errors"] = [f"Error in data: {str(e)}"]

    return output

def lagrange(x, y):
    n = len(x)
    polynomial = np.poly1d([0.0])

    for i in range(n):
        Li = np.poly1d([1.0])
        den = 1.0
        for j in range(n):
            if j != i:
                Li *= np.poly1d([1.0, -x[j]])
                den *= (x[i] - x[j])
        polynomial += y[i] * Li / den

    return polynomial.coeffs
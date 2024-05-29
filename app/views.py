import sympy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import urllib, base64
from django.shortcuts import render
import numpy as np
from .methods import biseccion, puntoFijo, newton, reglaFalsa, raicesMultiples, \
    jacobi, gaussSeidel, sor, secante, vandermonde, splineLineal, \
    splineCuadratica, splineCubica, newtonInter, lagrange


def infoView(request):
    return render(request, 'home.html')

def menuView(request):
    return render(request, 'metodosMenu.html')

def graficaView(request):
    return render(request, 'grafica.html')

def biseccionView(request):
    datos = {}
    error_message = None
    plot_url = None

    if request.method == 'POST':
        try:
            fx = request.POST["funcion"]
            tol = request.POST["tolerancia"]
            Tol = float(tol)
            niter = request.POST["iteraciones"]
            Niter = int(niter)
            xs = request.POST["xs"]
            Xs = float(xs)
            xi = request.POST["xi"]
            Xi = float(xi)

            datos = biseccion(fx, Tol, Niter, Xs, Xi)

            if not datos.get("errors"):
                # Graficar el resultado
                plt.figure(figsize=(10, 5))
                plt.plot(datos["x_vals"], datos["y_vals"], 'bo-', label='Bisección')
                plt.axhline(0, color='red', lw=0.5)
                plt.title('Método de Bisección')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = datos["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, './metodosPage/biseccion.html', {'data': datos, 'error_message': error_message, 'plot_url': plot_url})

def secanteView(request):
    datos = {}
    error_message = None
    plot_url = None

    if request.method == 'POST':
        try:
            fx = request.POST["funcion"]
            tol = request.POST["tolerancia"]
            Tol = float(tol)
            niter = request.POST["iteraciones"]
            Niter = int(niter)
            X0 = request.POST["xs"]
            x0 = float(X0)
            X1 = request.POST["xi"]
            x1 = float(X1)

            datos = secante(fx, Tol, Niter, x0, x1)

            if not datos.get("errors"):
                # Graficar el resultado
                x_vals = [float(d[1].strip()) for d in datos["results"]]
                y_vals = [float(d[2].strip()) for d in datos["results"]]

                plt.figure(figsize=(10, 5))
                plt.plot(x_vals, y_vals, 'bo-', label='Secante')
                plt.axhline(0, color='red', lw=0.5)
                plt.title('Método de la Secante')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = datos["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, './metodosPage/secante.html', {'data': datos, 'error_message': error_message, 'plot_url': plot_url})

def puntoFijoView(request):
    datos = {}
    error_message = None
    plot_url = None

    if request.method == 'POST':
        try:
            fx = request.POST["funcion-F"]
            gx = request.POST["funcion-G"]
            tol = request.POST["tolerancia"]
            Tol = float(tol)
            niter = request.POST["iteraciones"]
            Niter = int(niter)
            vInicial = request.POST["vInicial"]
            X0 = float(vInicial)

            datos = puntoFijo(X0, Tol, Niter, fx, gx)

            if not datos.get("errors"):
                # Graficar el resultado
                x_vals = [float(d[1].strip()) for d in datos["results"]]
                y_vals = [float(d[3].strip()) for d in datos["results"]]

                plt.figure(figsize=(10, 5))
                plt.plot(x_vals, y_vals, 'bo-', label='Punto Fijo')
                plt.axhline(0, color='red', lw=0.5)
                plt.title('Método de Punto Fijo')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = datos["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, './metodosPage/puntoFijo.html', {'data': datos, 'error_message': error_message, 'plot_url': plot_url})

def newtonView(request):
    datos = {}
    error_message = None
    plot_url = None

    if request.method == 'POST':
        try:
            fx = request.POST["funcion"]
            derf = request.POST["funcion-df"]

            x0 = request.POST["vInicial"]
            X0 = float(x0)

            tol = request.POST["tolerancia"]
            Tol = float(tol)

            niter = request.POST["iteraciones"]
            Niter = int(niter)

            datos = newton(X0, Tol, Niter, fx, derf)

            if not datos.get("errors"):
                # Graficar el resultado
                x_vals = [float(d[1].strip()) for d in datos["results"]]
                y_vals = [float(d[2].strip()) for d in datos["results"]]

                plt.figure(figsize=(10, 5))
                plt.plot(x_vals, y_vals, 'bo-', label='Newton-Raphson')
                plt.axhline(0, color='red', lw=0.5)
                plt.title('Método de Newton-Raphson')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = datos["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, './metodosPage/newton.html', {'data': datos, 'error_message': error_message, 'plot_url': plot_url})

def reglaFalsaView(request):
    datos = {}
    error_message = None
    plot_url = None

    if request.method == 'POST':
        try:
            fx = request.POST["funcion"]

            x0 = request.POST["lowerinterval"]
            X0 = float(x0)

            xi = request.POST["higherinterval"]
            Xi = float(xi)

            tol = request.POST["tolerancia"]
            Tol = float(tol)

            niter = request.POST["iteraciones"]
            Niter = int(niter)

            datos = reglaFalsa(X0, Xi, Niter, Tol, fx)

            if not datos.get("errors"):
                # Graficar el resultado
                x_vals = [float(d[2].strip()) for d in datos["results"]]
                y_vals = [float(d[4].strip()) for d in datos["results"]]

                plt.figure(figsize=(10, 5))
                plt.plot(x_vals, y_vals, 'bo-', label='Regla Falsa')
                plt.axhline(0, color='red', lw=0.5)
                plt.title('Método de Regla Falsa')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = datos["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, './metodosPage/reglaFalsa.html', {'data': datos, 'error_message': error_message, 'plot_url': plot_url})

def raicesMultiplesView(request):
    datos = {}
    error_message = None
    plot_url = None

    if request.method == 'POST':
        try:
            Fx = request.POST["funcion"]

            X0 = request.POST["x0"]
            X0 = float(X0)

            niter = request.POST["iteraciones"]
            Niter = int(niter)

            Tol = request.POST["tolerancia"]
            Tol = float(Tol)

            datos = raicesMultiples(Fx, X0, Tol, Niter)

            if not datos.get("errors"):
                # Graficar el resultado
                x_vals = [float(d[1].strip()) for d in datos["results"]]
                y_vals = [float(d[2].strip()) for d in datos["results"]]

                plt.figure(figsize=(10, 5))
                plt.plot(x_vals, y_vals, 'bo-', label='Raíces Múltiples')
                plt.axhline(0, color='red', lw=0.5)
                plt.title('Método de Raíces Múltiples')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = datos["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, "./metodosPage/raicesMultiples.html", {"data": datos, "error_message": error_message, "plot_url": plot_url})

def jacobiSeidelView(request):
    datos = {}
    error_message = None
    plot_url = None

    if request.method == 'POST':
        try:
            mA = toMatrix(request.POST["matrizA"])
            Vx0 = toVector(request.POST["vectorX0"])
            Vb = toVector(request.POST["vectorB"])

            iter = request.POST["iteraciones"]
            Niter = int(iter)

            Tol = request.POST["tolerancia"]
            Tol = float(Tol)

            datos = jacobi(mA, Vb, Vx0, Tol, Niter)

            if not datos.get("errors"):
                # Graficar el resultado
                steps = datos["steps"]
                x_vals = list(range(len(steps)))
                y_vals = [np.linalg.norm(steps[step]) for step in steps]

                plt.figure(figsize=(10, 5))
                plt.plot(x_vals, y_vals, 'bo-', label='Jacobi')
                plt.axhline(0, color='red', lw=0.5)
                plt.title('Método de Jacobi')
                plt.xlabel('Iteraciones')
                plt.ylabel('Norma de x')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = datos["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, './metodosPage/jacobi.html', {'data': datos, 'error_message': error_message, 'plot_url': plot_url})

def gaussSeidelView(request):
    datos = {}
    error_message = None
    plot_url = None

    if request.method == 'POST':
        try:
            mA = toMatrix(request.POST["matrizA"])
            Vx0 = toVector(request.POST["vectorX0"])
            Vb = toVector(request.POST["vectorB"])

            niter = request.POST["iteraciones"]
            Niter = int(niter)

            Tol = request.POST["tolerancia"]
            Tol = float(Tol)

            datos = gaussSeidel(mA, Vb, Vx0, Tol, Niter)

            if not datos.get("errors"):
                # Graficar el resultado
                steps = datos["steps"]
                x_vals = list(range(len(steps)))
                y_vals = [np.linalg.norm(steps[step]) for step in steps]

                plt.figure(figsize=(10, 5))
                plt.plot(x_vals, y_vals, 'bo-', label='Gauss-Seidel')
                plt.axhline(0, color='red', lw=0.5)
                plt.title('Método de Gauss-Seidel')
                plt.xlabel('Iteraciones')
                plt.ylabel('Norma de x')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = datos["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, './metodosPage/gaussSeidel.html', {'data': datos, 'error_message': error_message, 'plot_url': plot_url})

def sorView(request):
    datos = {}
    error_message = None
    plot_url = None

    if request.method == 'POST':
        try:
            mA = toMatrix(request.POST["matrizA"])
            Vx0 = toVector(request.POST["vectorX0"])
            Vb = toVector(request.POST["vectorB"])
            w = request.POST["wValue"]
            W = float(w)

            niter = request.POST["iteraciones"]
            Niter = int(niter)

            Tol = request.POST["tolerancia"]
            Tol = float(Tol)

            datos = sor(mA, Vb, Vx0, W, Tol, Niter)

            if not datos.get("errors"):
                # Graficar el resultado
                iteraciones = [d[0] for d in datos["informacion"]]
                soluciones = [np.linalg.norm(d[1]) for d in datos["informacion"]]

                plt.figure(figsize=(10, 5))
                plt.plot(iteraciones, soluciones, 'bo-', label='SOR')
                plt.axhline(0, color='red', lw=0.5)
                plt.title('Método SOR')
                plt.xlabel('Iteraciones')
                plt.ylabel('Norma de x')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = datos["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, './metodosPage/sor.html', {'data': datos, 'error_message': error_message, 'plot_url': plot_url})

def toVector(vector_str):
    return [float(i) for i in vector_str.split(",")]

def splineLinealView(request):
    Coef = []
    Tracers = []
    error_message = None
    plot_url = None
    output = {"errors": []}

    if request.method == 'POST':
        try:
            x = request.POST["x"]
            X = toVector(x)
            y = request.POST["y"]
            Y = toVector(y)

            output = splineLineal(X, Y)

            if not output["errors"]:
                X, Y, x_vals, y_vals = output["results"]
                Coef = output["coef"]
                Tracers = output["tracers"]

                # Graficar los resultados
                plt.figure(figsize=(10, 5))
                plt.scatter(X, Y, color='r', label='Puntos')
                plt.plot(x_vals, y_vals, label='Interpolación lineal')

                plt.title('Método Spline Lineal')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = output["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, "./metodosPage/spline-lineal.html", {
        "coef": Coef,
        "tracers": Tracers,
        "errors": output["errors"],
        "error_message": error_message,
        "plot_url": plot_url
    })

def splineCuadraticaView(request):
    Coef = []
    Tracers = []
    error_message = None
    plot_url = None
    output = {"errors": []}

    if request.method == 'POST':
        try:
            x = request.POST["x"]
            X = toVector(x)
            y = request.POST["y"]
            Y = toVector(y)

            output = splineCuadratica(X, Y)

            if not output["errors"]:
                X, Y, x_vals, y_vals = output["results"]
                Coef = output["coef"]
                Tracers = output["tracers"]

                # Graficar los resultados
                plt.figure(figsize=(10, 5))
                plt.scatter(X, Y, color='r', label='Puntos')
                plt.plot(x_vals, y_vals, label='Interpolación cuadrática')

                plt.title('Método Spline Cuadrática')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = output["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, "./metodosPage/spline-cuadratica.html", {
        "coef": Coef,
        "tracers": Tracers,
        "errors": output["errors"],
        "error_message": error_message,
        "plot_url": plot_url
    })

def splineCubicaView(request):
    Coef = []
    Tracers = []
    error_message = None
    plot_url = None
    output = {"errors": []}

    if request.method == 'POST':
        try:
            x = request.POST["x"]
            X = toVector(x)
            y = request.POST["y"]
            Y = toVector(y)

            output = splineCubica(X, Y)

            if not output["errors"]:
                X, Y, x_vals, y_vals = output["results"]
                Coef = output["coef"]
                Tracers = output["tracers"]

                # Graficar los resultados
                plt.figure(figsize=(10, 5))
                plt.scatter(X, Y, color='r', label='Puntos')
                plt.plot(x_vals, y_vals, label='Interpolación cúbica')

                plt.title('Método Spline Cúbica')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = output["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, "./metodosPage/spline-cubica.html", {
        "coef": Coef,
        "tracers": Tracers,
        "errors": output["errors"],
        "error_message": error_message,
        "plot_url": plot_url
    })

def vandermondeView(request):
    datos = {}
    error_message = None
    plot_url = None

    if request.method == 'POST':
        try:
            vectorX = toVector(request.POST["vectorX"])
            vectorY = toVector(request.POST["vectorY"])
            datos = vandermonde(vectorX, vectorY)

            if not datos.get("errors"):
                # Graficar el polinomio
                polinomio = sympy.lambdify(sympy.Symbol('x'), datos['polinomio'])
                x_vals = np.linspace(min(vectorX), max(vectorX), 100)
                y_vals = polinomio(x_vals)

                plt.figure(figsize=(10, 5))
                plt.plot(x_vals, y_vals, 'b-', label='Polinomio Interpolante')
                plt.scatter(vectorX, vectorY, color='r', label='Puntos')
                plt.title('Método de Vandermonde')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
            else:
                error_message = datos["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, './metodosPage/vandermonde.html', {'data': datos, 'error_message': error_message, 'plot_url': plot_url})

def newton_poly(coef, x_data, x):
    n = len(x_data) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p

def newtonInterView(request):
    Coef = []
    error_message = None
    plot_url = None
    output = {}

    if request.method == 'POST':
        try:
            x = request.POST["x"]
            X = toVector(x)
            y = request.POST["y"]
            Y = toVector(y)

            output = newtonInter(X, Y)

            if "errors" not in output:
                D = output["D"]
                Coef = output["Coef"]

                # Graficar los resultados
                x_vals = np.linspace(min(X), max(X), 100)
                y_vals = [newton_poly(Coef, X, x) for x in x_vals]

                plt.figure(figsize=(10, 5))
                plt.scatter(X, Y, color='r', label='Puntos')
                plt.plot(x_vals, y_vals, label='Interpolación de Newton')

                plt.title('Método de Newton (Divided Difference)')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)

            else:
                error_message = output["errors"][0]

        except Exception as e:
            error_message = str(e)

    return render(request, "./metodosPage/newtonInter.html", {
        "datos": output,
        "error_message": error_message,
        "plot_url": plot_url
    })

#Metodos auxiliar
def toMatrix(matrixStr):
    matrixStr = matrixStr.replace(" ","")
    matrixStr = matrixStr.replace("\n","")
    rows = matrixStr.split(";")
    auxM = []
    for row in rows:
        splitedRow = row.split(",")
        auxR = []
        for num in splitedRow:
            auxR.append(float(num))
        auxM.append(auxR)
    return auxM

def toVector(vectorStr):

    splitedVector = vectorStr.split(",")
    auxV = list()
    for num in splitedVector:
        auxV.append(float(num))
    return auxV

def newtonDiffDivOutput(output):
    stringOutput = f'\n"Metodo"\n'
    stringOutput += "\nResults:\n"
    stringOutput += "\nDivided differences table:\n\n"
    rel = output["D"]
    stringOutput += '{:^7f}'.format(rel[0,0]) +"   //L \n"

    stringOutput += "\nNewton's polynomials coefficents:\n\n"
    rel = output["Coef"]

    stringOutput += "\nNewton interpolating polynomials:\n\n"
    rel = output["Coef"]
    i = 0
    while i < len(rel) :
        stringOutput += '{:^7f}'.format(rel[i,0]) +"x^3"
        stringOutput += format(rel[i,1],"+.6f") + "x^2"
        stringOutput += format(rel[i,2],"+.6f") + "x"
        stringOutput += format(rel[i,3],"+.6f") + "   //L \n"
        i += 1

    stringOutput += "\n______________________________________________________________\n"
    return stringOutput


def lagrangeView(request):
    error_message = None
    plot_url = None
    coefficients = []

    if request.method == 'POST':
        try:
            x = request.POST.get('x')
            y = request.POST.get('y')

            # Convertir a float
            X = list(map(float, x.split(',')))
            Y = list(map(float, y.split(',')))

            # Calcular los coeficientes del polinomio
            coefficients = lagrange(X, Y)

            # Graficar los resultados
            x_vals = np.linspace(min(X), max(X), 100)
            y_vals = np.polyval(coefficients, x_vals)

            plt.figure(figsize=(10, 5))
            plt.scatter(X, Y, color='r', label='Puntos')
            plt.plot(x_vals, y_vals, label='Interpolación de Lagrange')

            plt.title('Interpolación de Lagrange')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            plot_url = urllib.parse.quote(string)

        except Exception as e:
            error_message = str(e)

    return render(request, './metodosPage/lagrange.html', {
        'coefficients': coefficients,
        'error_message': error_message,
        'plot_url': plot_url
    })
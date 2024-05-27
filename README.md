# Análisis Numérico

Este repositorio contiene una implementación de varios métodos de análisis numérico desarrollados en Python y Django. La aplicación permite resolver problemas numéricos mediante diferentes métodos como el método de bisección, entre otros.

## Desarrolladores

- Juan Sebastian Camacho Palacio
- Laura Danniela Zarate Guerrero
- Felipe Uribe Correa
- Pablo Baez Santamaria

## Requisitos

Para ejecutar este proyecto, asegúrate de tener instaladas las siguientes dependencias:

```plaintext
Django==3.1.7
numpy==1.20.1
pandas==1.2.3
matplotlib==3.3.4
sympy==1.7.1
```

Puedes instalarlas utilizando el archivo `requirements.txt` incluido en el repositorio. Para ello, ejecuta:

```bash
pip install -r requirements.txt
```

## Instrucciones de uso

Sigue los siguientes pasos para ejecutar la aplicación:

1. **Clonar el repositorio:**

    ```bash
    git clone https://github.com/PabloBaezS/ProyectoFinalAnalisis
    cd Analisis_Numerico
    ```

2. **Crear y activar un entorno virtual (opcional pero recomendado):**

    ```bash
    python -m venv env
    source env/bin/activate  # En Windows usa `env\Scripts\activate`
    ```

3. **Instalar las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Aplicar las migraciones de la base de datos:**

    ```bash
    python manage.py migrate
    ```

5. **Iniciar el servidor de desarrollo:**

    ```bash
    python manage.py runserver
    ```

6. **Abrir la aplicación en el navegador:**

    Abre tu navegador web y navega a `http://127.0.0.1:8000` para acceder a la aplicación.

## Información adicional

### Métodos implementados

Este proyecto incluye implementaciones de varios métodos numéricos, tales como:

- Método de bisección
- Método de Newton-Raphson
- Método de la secante
- Entre otros

Cada método incluye una interfaz amigable donde puedes ingresar los parámetros necesarios y obtener los resultados tanto en forma de tabla como de gráficos.

### Estructura del proyecto

La estructura del proyecto es la siguiente:

```
Analisis_Numerico/
├── manage.py
├── metodos/
│   ├── admin.py
│   ├── apps.py
│   ├── __init__.py
│   ├── migrations/
│   │   ├── __init__.py
│   ├── models.py
│   ├── static/
│   ├── templates/
│   │   ├── layout.html
│   │   ├── metodosPage/
│   │       ├── biseccion.html
│   │       ├── newton.html
│   │       ├── secante.html
│   ├── tests.py
│   ├── urls.py
│   ├── views.py
├── Analisis_Numerico/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
├── requirements.txt
```

Saludos!

from setuptools import setup
try:
    import firedrake # noqa
except ImportError:
    raise Exception("Firedrake needs to be installed and activated. "
                    "Please visit firedrakeproject.org")
setup(
    name='IRKsome',
    version='0.0.1',
    author='Rob Kirby, Jorge Marchena Menendez',
    author_email='Robert_Kirby@baylor.edu',
    description='A library for fully implicit Runge-Kutta methods in Firedrake',
    long_description='',
    packages=['irksome'],
    zip_safe=False,
)

""" Checks if all the dependencies are installed
"""
import importlib


def check():
    """Checks if all the dependencies are installed
    """
    modnames = [
        "os",
        "tarfile",
        "numpy",
        "matplotlib",
        "pandas",
        "six",
        "scipy",
        "sklearn",
    ]
    flag = True
    for lib in modnames:
        try:
            globals()[lib] = importlib.import_module(lib)
        except ImportError:
            flag = False
            print("please install {}".format(lib))
            pass  # module doesn't exist, deal with it.

    if flag:
        print("All necessary modules installed")
    return

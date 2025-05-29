import os

fontsize = 25
label_surp = 5

PYTHON_PATH = "/home/fazinhan/miniconda3/envs/cavityCoupling/bin/python"
# PYTHON_PATH = "C:/Users/freak/miniforge3/envs/cavityCoupling/python.exe"
DIR = os.path.dirname(os.path.abspath(__file__))

os.system(f'{PYTHON_PATH} "{os.path.join(DIR,"g1g2_vs_t_plotter_saturation.py")}" {fontsize} {label_surp}')
os.system(f'{PYTHON_PATH} "{os.path.join(DIR,"g2_vs_g1.py")}" {fontsize} {label_surp}')
os.system(f'{PYTHON_PATH} "{os.path.join(DIR,"gyig_vs_t.py")}" {fontsize} {label_surp}')
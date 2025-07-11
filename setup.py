from setuptools import setup, find_packages
import os

setup_py_dir = os.path.dirname(os.path.realpath(__file__))
need_files = []
datadir = 'rl_kuka_robot'
print("----setup_py_dir----", setup_py_dir)

hh = setup_py_dir

for root, dirs, files in os.walk(hh):
  for fn in files:
    ext = os.path.splitext(fn)[1][1:]
    if ext and ext in 'urdf sdf xml yaml stl ini dae'.split():
      fn = root + "/" + fn
      need_files.append(fn[1 + len(hh):])

setup(
  name = 'rl_kuka_robot',
  version = '0.1',
  packages= find_packages(),
  package_data={'rl_kuka_robot': need_files},
  python_requires = '>=3.5',
  description = 'kuka robot trajectory planning using rl',
  author = 'zgj',
  author_email = 'Ghost100453@gmail.com',
  url = 'None',
)
import os
from setuptools import setup
from setuptools.command.install import install
from distutils.command.install import install as install_

class _install(install):
    def run(self):
        path = os.getcwd()
        print('请输入您的MATLAB根目录(不含单引号)：')
        print('MATLAB根目录的获取方式为：在MATLAB命令行输入\"matlabroot\"')
        root = input()
        if root[0] == path[0]:
            os.system('cd ' + root + '\extern\engines\python && python setup.py install')
        else:
            os.system(root[0] + ': && cd ' + root + '\extern\engines\python && python setup.py install')
        install_.run(self)

setup(
    name='视听导大作业1',  # 包名
    version='1.0',  # 包版
    author='hao qianyue, liu yuyang, hu kaizhe',  # 作者
    cmdclass={'install': _install},
    install_requires=[
        'matplotlib',
        'numpy',
        'pillow'
    ]
)

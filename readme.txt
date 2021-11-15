>> virtualenv -p /usr/local/bin/python3.6 venv (because of pyfftw)
>> source venv/bin/activate
>> pip install numpy scipy colossus
>> pip install mpi4py mpi4py_fft
#optionals
#>> pip install matplotlib ipython
#>> pip install jedi==0.17.2 (downgrade jedi because it clashes with ipython)

to install
>> make

>> pip install ipykernel
>> python -m ipykernel install --user --name=muscleups


to compile
>> make
>> make -f MafileMAS

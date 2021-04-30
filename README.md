# LSTM_Treasury_Yield_Curve
Treasury yield curve prediction using a Multivariate LSTM sequential model with visualization.
The solution primarily utilizes Keras and TensorFlow libraries to model US Treasury Yield Curves.

The model uses quarterly Real GDP, Real Disposable Income, US Unemployment, CPI Inflation, Fed Discount Rates and Chinese Discount Rates.  The 3 month, 6 month, 1 year, 2 year, 3 year and 10 year US Treasury Yields are generated. 

The data sources are described in the Data Sources word document.

This solution is an incremental improvement over the multivariant sequential regression solution that is also described in the FITSolutionsusa.com blog (https://www.fitsolutionsusa.com/blog/treasury-yield-predictions)

The solution leverages the following libraries -

Package                Version
---------------------- -------------------
absl-py                0.12.0
argon2-cffi            20.1.0
astunparse             1.6.3
async-generator        1.10
attrs                  20.3.0
backcall               0.2.0
bleach                 3.3.0
cachetools             4.2.1
certifi                2020.12.5
cffi                   1.14.5
chardet                4.0.0
chart-studio           1.1.0
colorama               0.4.4
cycler                 0.10.0
decorator              5.0.5
defusedxml             0.7.1
entrypoints            0.3
flatbuffers            1.12
gast                   0.3.3
google-auth            1.28.0
google-auth-oauthlib   0.4.4
google-pasta           0.2.0
grpcio                 1.32.0
h5py                   2.10.0
idna                   2.10
importlib-metadata     3.7.3
inflection             0.5.1
ipykernel              5.3.4
ipython                7.22.0
ipython-genutils       0.2.0
jedi                   0.17.0
Jinja2                 2.11.3
joblib                 1.0.1
jsonschema             3.2.0
jupyter-client         6.1.12
jupyter-core           4.7.1
jupyterlab-pygments    0.1.2
Keras                  2.4.3
Keras-Preprocessing    1.1.2
kiwisolver             1.3.1
Markdown               3.3.4
MarkupSafe             1.1.1
matplotlib             3.4.1
mistune                0.8.4
more-itertools         8.7.0
nbclient               0.5.3
nbconvert              6.0.7
nbformat               5.1.3
nest-asyncio           1.5.1
notebook               6.3.0
numpy                  1.19.5
oauthlib               3.1.0
opt-einsum             3.3.0
packaging              20.9
pandas                 1.2.4
pandocfilters          1.4.3
parso                  0.8.2
pickleshare            0.7.5
Pillow                 8.2.0
pip                    21.0.1
plotly                 4.14.3
prometheus-client      0.10.0
prompt-toolkit         3.0.17
protobuf               3.15.7
pyasn1                 0.4.8
pyasn1-modules         0.2.8
pycparser              2.20
Pygments               2.8.1
pyparsing              2.4.7
pyrsistent             0.17.3
python-dateutil        2.8.1
pytz                   2021.1
pywin32                227
pywinpty               0.5.7
PyYAML                 5.4.1
pyzmq                  20.0.0
Quandl                 3.6.1
requests               2.25.1
requests-oauthlib      1.3.0
retrying               1.3.3
rsa                    4.7.2
scikit-learn           0.24.1
scipy                  1.6.2
Send2Trash             1.5.0
setuptools             52.0.0.post20210125
six                    1.15.0
sklearn                0.0
tensorboard            2.4.1
tensorboard-plugin-wit 1.8.0
tensorflow-estimator   2.4.0
tensorflow-gpu         2.4.0
termcolor              1.1.0
terminado              0.9.4
testpath               0.4.4
threadpoolctl          2.1.0
tornado                6.1
traitlets              5.0.5
typing-extensions      3.7.4.3
urllib3                1.26.4
wcwidth                0.2.5
webencodings           0.5.1
Werkzeug               1.0.1
wheel                  0.36.2
wincertstore           0.2
wrapt                  1.12.1
zipp                   3.4.1


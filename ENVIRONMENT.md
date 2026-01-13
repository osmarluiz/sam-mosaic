# ts_annotator Environment

Environment specifications for the `ts_annotator` conda environment.

**Python Version:** 3.11.14

## Core Dependencies

### Deep Learning
| Package | Version | Channel |
|---------|---------|---------|
| torch | 2.6.0+cu124 | pypi |
| torchvision | 0.21.0+cu124 | pypi |
| torchaudio | 2.6.0+cu124 | pypi |
| pytorch-cuda | 12.4 | pytorch |
| fastai | 2.8.5 | pypi |
| sam2 | 1.1.0 | pypi |

### Machine Learning & Data Science
| Package | Version | Channel |
|---------|---------|---------|
| scikit-learn | 1.7.2 | conda |
| numpy | 1.26.4 | pypi |
| pandas | 2.3.3 | conda |
| scipy | 1.16.3 | pypi |
| tsai | 0.4.1 | pypi |
| tslearn | 0.7.0 | pypi |
| pyts | 0.13.0 | pypi |
| cleanlab | 2.7.1 | pypi |
| imbalanced-learn | 0.14.0 | pypi |
| numba | 0.63.1 | pypi |
| joblib | 1.5.2 | conda |

### NLP
| Package | Version | Channel |
|---------|---------|---------|
| spacy | 3.8.11 | pypi |
| spacy-legacy | 3.0.12 | pypi |
| spacy-loggers | 1.0.5 | pypi |

### Geospatial
| Package | Version | Channel |
|---------|---------|---------|
| rasterio | 1.4.4 | conda-forge |
| geopandas | 1.1.2 | pypi |
| shapely | 2.1.2 | conda-forge |
| pyproj | 3.7.2 | pypi |
| fiona | 1.10.1 | conda-forge |
| pyogrio | 0.12.1 | pypi |
| affine | 2.4.0 | conda-forge |

### Visualization
| Package | Version | Channel |
|---------|---------|---------|
| matplotlib | 3.10.7 | conda |
| seaborn | 0.13.2 | pypi |
| pyqtgraph | 0.14.0 | pypi |
| pillow | 12.0.0 | conda-forge |
| ipywidgets | 8.1.8 | pypi |

### Configuration & Utilities
| Package | Version | Channel |
|---------|---------|---------|
| pydantic | 2.12.5 | pypi |
| pydantic-settings | 2.12.0 | pypi |
| hydra-core | 1.3.2 | pypi |
| omegaconf | 2.3.0 | pypi |
| loguru | 0.7.3 | pypi |
| tqdm | 4.67.1 | conda |
| click | 8.3.1 | conda-forge |
| rich | 14.2.0 | pypi |
| python-dotenv | 1.2.1 | pypi |

### Project Package
| Package | Version | Channel |
|---------|---------|---------|
| sam-mosaic | 2.0.0 | pypi |
| sits | 0.1.0 | pypi |

---

## Complete Package List

Below is the complete list of all packages installed in the environment.

| Package | Version | Build | Channel |
|---------|---------|-------|---------|
| _libavif_api | 1.3.0 | h57928b3_2 | conda-forge |
| _openmp_mutex | 4.5 | 2_gnu | conda-forge |
| affine | 2.4.0 | pyhd8ed1ab_1 | conda-forge |
| annotated-types | 0.7.0 | | pypi |
| antlr4-python3-runtime | 4.9.3 | | pypi |
| aom | 3.9.1 | he0c23c2_0 | conda-forge |
| asttokens | 3.0.1 | pyhd8ed1ab_0 | conda-forge |
| attrs | 25.4.0 | pyhcf101f3_1 | conda-forge |
| beartype | 0.22.8 | | pypi |
| blas | 1.0 | mkl | conda |
| blis | 1.3.3 | | pypi |
| blosc | 1.21.6 | hfd34d9b_1 | conda-forge |
| bottleneck | 1.4.2 | py311h540bb41_1 | conda |
| brotlicffi | 1.2.0.0 | py311h885b0b7_0 | conda |
| bzip2 | 1.0.8 | h2bbff1b_6 | conda |
| ca-certificates | 2025.12.2 | haa95532_0 | conda |
| catalogue | 2.0.10 | | pypi |
| certifi | 2025.11.12 | pyhd8ed1ab_0 | conda-forge |
| cffi | 2.0.0 | py311h02ab6af_1 | conda |
| charset-normalizer | 3.4.4 | py311haa95532_0 | conda |
| cleanlab | 2.7.1 | | pypi |
| click | 8.3.1 | pyha7b4d00_1 | conda-forge |
| click-plugins | 1.1.1.2 | pyhd8ed1ab_0 | conda-forge |
| cligj | 0.7.2 | pyhd8ed1ab_2 | conda-forge |
| cloudpathlib | 0.23.0 | | pypi |
| cloudpickle | 3.1.2 | | pypi |
| colorama | 0.4.6 | pyhd8ed1ab_1 | conda-forge |
| comm | 0.2.3 | pyhe01879c_0 | conda-forge |
| confection | 0.1.5 | | pypi |
| contourpy | 1.3.3 | py311h214f63a_0 | conda |
| cuda-cccl | 12.9.27 | 0 | nvidia |
| cuda-cccl_win-64 | 12.9.27 | 0 | nvidia |
| cuda-cudart | 12.4.127 | 0 | nvidia |
| cuda-cudart-dev | 12.4.127 | 0 | nvidia |
| cuda-cupti | 12.4.127 | 0 | nvidia |
| cuda-libraries | 12.4.1 | 0 | nvidia |
| cuda-libraries-dev | 12.4.1 | 0 | nvidia |
| cuda-nvrtc | 12.4.127 | 0 | nvidia |
| cuda-nvrtc-dev | 12.4.127 | 0 | nvidia |
| cuda-nvtx | 12.4.127 | 0 | nvidia |
| cuda-opencl | 12.9.19 | 0 | nvidia |
| cuda-opencl-dev | 12.9.19 | 0 | nvidia |
| cuda-profiler-api | 12.9.79 | 0 | nvidia |
| cuda-runtime | 12.4.1 | 0 | nvidia |
| cuda-version | 12.9 | 3 | nvidia |
| cycler | 0.11.0 | pyhd3eb1b0_0 | conda |
| cymem | 2.0.13 | | pypi |
| dav1d | 1.2.1 | h2bbff1b_0 | conda |
| debugpy | 1.8.17 | py311h5dfdfe8_1 | conda-forge |
| decorator | 5.2.1 | pyhd8ed1ab_0 | conda-forge |
| executing | 2.2.1 | pyhd8ed1ab_0 | conda-forge |
| expat | 2.7.3 | h885b0b7_4 | conda |
| fastai | 2.8.5 | | pypi |
| fastcore | 1.8.17 | | pypi |
| fastdownload | 0.0.7 | | pypi |
| fastprogress | 1.0.3 | | pypi |
| fasttransform | 0.0.2 | | pypi |
| filelock | 3.20.0 | py311haa95532_0 | conda |
| fiona | 1.10.1 | py311h8db5f30_6 | conda-forge |
| fonttools | 4.61.0 | py311h02ab6af_0 | conda |
| freetype | 2.14.1 | h57928b3_0 | conda-forge |
| freexl | 2.0.0 | hf297d47_2 | conda-forge |
| fsspec | 2025.10.0 | py311h4442805_0 | conda |
| geopandas | 1.1.2 | | pypi |
| geos | 3.14.1 | hdade9fe_0 | conda-forge |
| geotiff | 1.7.4 | h73469f5_4 | conda-forge |
| giflib | 5.2.2 | h7edc060_0 | conda |
| gmp | 6.3.0 | h537511b_0 | conda |
| gmpy2 | 2.2.2 | py311h8598115_0 | conda |
| gst-plugins-base | 1.24.12 | h91a6125_1 | conda |
| gstreamer | 1.24.12 | hfb93a4f_1 | conda |
| gstreamer-orc | 0.4.41 | ha00e802_0 | conda |
| hydra-core | 1.3.2 | | pypi |
| icu | 75.1 | he0c23c2_0 | conda-forge |
| idna | 3.11 | py311haa95532_0 | conda |
| imbalanced-learn | 0.14.0 | | pypi |
| importlib-metadata | 8.7.0 | py311haa95532_0 | conda |
| iniconfig | 2.3.0 | | pypi |
| intel-openmp | 2023.1.0 | h59b6b97_46320 | conda |
| iopath | 0.1.10 | | pypi |
| ipykernel | 7.1.0 | pyh6dadd2b_0 | conda-forge |
| ipython | 9.8.0 | pyhe2676ad_0 | conda-forge |
| ipython_pygments_lexers | 1.1.1 | pyhd8ed1ab_0 | conda-forge |
| ipywidgets | 8.1.8 | | pypi |
| jedi | 0.19.2 | pyhd8ed1ab_1 | conda-forge |
| jinja2 | 3.1.6 | py311haa95532_0 | conda |
| joblib | 1.5.2 | py311haa95532_0 | conda |
| jupyter_client | 8.7.0 | pyhcf101f3_0 | conda-forge |
| jupyter_core | 5.9.1 | pyh6dadd2b_0 | conda-forge |
| jupyterlab-widgets | 3.0.16 | | pypi |
| khronos-opencl-icd-loader | 2025.07.22 | h79b28c9_0 | conda |
| kiwisolver | 1.4.9 | py311h03f52e7_0 | conda |
| krb5 | 1.21.3 | hdf4eb48_0 | conda-forge |
| lcms2 | 2.17 | hbcf6048_0 | conda-forge |
| lerc | 4.0.0 | h6470a55_1 | conda-forge |
| libabseil | 20250127.0 | cxx17_h52369b4_0 | conda |
| libarchive | 3.8.2 | gpl_h26aea39_100 | conda-forge |
| libavif16 | 1.3.0 | he916da2_2 | conda-forge |
| libblas | 3.9.0 | 20_win64_mkl | conda-forge |
| libbrotlicommon | 1.2.0 | hfd05255_1 | conda-forge |
| libbrotlidec | 1.2.0 | hfd05255_1 | conda-forge |
| libbrotlienc | 1.2.0 | hfd05255_1 | conda-forge |
| libcblas | 3.9.0 | 20_win64_mkl | conda-forge |
| libclang13 | 20.1.8 | default_hccbf073_0 | conda |
| libcublas | 12.4.5.8 | 0 | nvidia |
| libcublas-dev | 12.4.5.8 | 0 | nvidia |
| libcufft | 11.2.1.3 | 0 | nvidia |
| libcufft-dev | 11.2.1.3 | 0 | nvidia |
| libcurand | 10.3.10.19 | 0 | nvidia |
| libcurand-dev | 10.3.10.19 | 0 | nvidia |
| libcurl | 8.17.0 | h43ecb02_1 | conda-forge |
| libcusolver | 11.6.1.9 | 0 | nvidia |
| libcusolver-dev | 11.6.1.9 | 0 | nvidia |
| libcusparse | 12.3.1.170 | 0 | nvidia |
| libcusparse-dev | 12.3.1.170 | 0 | nvidia |
| libde265 | 1.0.15 | h91493d7_0 | conda-forge |
| libdeflate | 1.25 | h51727cc_0 | conda-forge |
| libexpat | 2.7.3 | h885b0b7_4 | conda |
| libffi | 3.5.2 | h52bdfb6_0 | conda-forge |
| libfreetype | 2.14.1 | h57928b3_0 | conda-forge |
| libfreetype6 | 2.14.1 | hdbac1cb_0 | conda-forge |
| libgcc | 15.2.0 | h8ee18e1_16 | conda-forge |
| libgdal-core | 3.12.1 | h4c6072a_0 | conda-forge |
| libglib | 2.86.3 | h0c9aed9_0 | conda-forge |
| libgomp | 15.2.0 | h8ee18e1_16 | conda-forge |
| libheif | 1.19.7 | gpl_h2684147_100 | conda-forge |
| libhwloc | 2.12.1 | default_h4379cf1_1003 | conda-forge |
| libhwy | 1.3.0 | ha71e874_1 | conda-forge |
| libiconv | 1.18 | hc1393d2_2 | conda-forge |
| libintl | 0.22.5 | h5728263_3 | conda-forge |
| libjpeg-turbo | 3.1.2 | hfd05255_0 | conda-forge |
| libjxl | 0.11.1 | hac9b6f3_5 | conda-forge |
| libkml | 1.3.0 | h68a222c_1022 | conda-forge |
| liblapack | 3.9.0 | 20_win64_mkl | conda-forge |
| libllvm20 | 20.1.8 | h3aa9ab2_0 | conda |
| liblzma | 5.8.1 | h2466b09_2 | conda-forge |
| liblzma-devel | 5.8.1 | h2466b09_2 | conda-forge |
| libnpp | 12.2.5.30 | 0 | nvidia |
| libnpp-dev | 12.2.5.30 | 0 | nvidia |
| libnvfatbin | 12.9.82 | 0 | nvidia |
| libnvfatbin-dev | 12.9.82 | 0 | nvidia |
| libnvjitlink | 12.4.127 | 0 | nvidia |
| libnvjitlink-dev | 12.4.127 | 0 | nvidia |
| libnvjpeg | 12.3.1.117 | 0 | nvidia |
| libnvjpeg-dev | 12.3.1.117 | 0 | nvidia |
| libopenjpeg | 2.5.4 | h02ab6af_1 | conda |
| libpng | 1.6.53 | h7351971_0 | conda-forge |
| libprotobuf | 5.29.3 | h65a231f_1 | conda |
| librttopo | 1.1.0 | haa95264_20 | conda-forge |
| libsodium | 1.0.20 | hc70643c_0 | conda-forge |
| libspatialite | 5.1.0 | gpl_h0cd62ae_119 | conda-forge |
| libsqlite | 3.51.1 | hf5d6505_0 | conda-forge |
| libssh2 | 1.11.1 | h9aa295b_0 | conda-forge |
| libtiff | 4.7.1 | h8f73337_1 | conda-forge |
| libtorch | 2.5.1 | cpu_mkl_h9002858_104 | conda |
| libuv | 1.48.0 | h827c3e9_0 | conda |
| libwebp-base | 1.6.0 | h4d5522a_0 | conda-forge |
| libwinpthread | 12.0.0.r4.gg4f2fc60ca | h57928b3_10 | conda-forge |
| libxcb | 1.17.0 | h0e4246c_0 | conda-forge |
| libxml2 | 2.15.1 | ha29bfb0_0 | conda-forge |
| libxml2-16 | 2.15.1 | h06f855e_0 | conda-forge |
| libxml2-devel | 2.15.1 | ha29bfb0_0 | conda-forge |
| libzlib | 1.3.1 | h02ab6af_0 | conda |
| llvmlite | 0.46.0 | | pypi |
| loguru | 0.7.3 | | pypi |
| lz4-c | 1.10.0 | h2466b09_1 | conda-forge |
| lzo | 2.10 | h6a83c73_1002 | conda-forge |
| markdown-it-py | 4.0.0 | | pypi |
| markupsafe | 3.0.2 | py311h827c3e9_0 | conda |
| matplotlib | 3.10.7 | py311haa95532_0 | conda |
| matplotlib-base | 3.10.7 | py311h26e45b9_0 | conda |
| matplotlib-inline | 0.2.1 | pyhd8ed1ab_0 | conda-forge |
| mdurl | 0.1.2 | | pypi |
| minizip | 4.0.10 | h9fa1bad_0 | conda-forge |
| mkl | 2023.1.0 | h6b88ed4_46358 | conda |
| mkl-service | 2.4.0 | py311h827c3e9_2 | conda |
| mpc | 1.3.1 | h827c3e9_0 | conda |
| mpfr | 4.2.1 | h56c3642_0 | conda |
| mpmath | 1.3.0 | py311haa95532_0 | conda |
| muparser | 2.3.5 | he0c23c2_0 | conda-forge |
| murmurhash | 1.0.15 | | pypi |
| nest-asyncio | 1.6.0 | pyhd8ed1ab_1 | conda-forge |
| networkx | 3.6.1 | py311haa95532_0 | conda |
| numba | 0.63.1 | | pypi |
| numexpr | 2.11.0 | py311hdb065b2_0 | conda |
| numpy | 1.26.4 | | pypi |
| omegaconf | 2.3.0 | | pypi |
| opencl-headers | 2025.07.22 | h885b0b7_0 | conda |
| openjpeg | 2.5.4 | h56d5a42_1 | conda |
| openssl | 3.6.0 | h725018a_0 | conda-forge |
| opentelemetry-api | 1.38.0 | py311haa95532_0 | conda |
| packaging | 25.0 | pyh29332c3_1 | conda-forge |
| pandas | 2.3.3 | py311h42c1672_1 | conda |
| parso | 0.8.5 | pyhcf101f3_0 | conda-forge |
| pcre2 | 10.47 | hd2b5f0e_0 | conda-forge |
| pillow | 12.0.0 | py311h17b8079_2 | conda-forge |
| pip | 25.3 | pyhc872135_0 | conda |
| platformdirs | 4.5.1 | pyhcf101f3_0 | conda-forge |
| pluggy | 1.6.0 | | pypi |
| plum-dispatch | 2.6.0 | | pypi |
| portalocker | 3.2.0 | | pypi |
| preshed | 3.0.12 | | pypi |
| proj | 9.7.1 | h7b1ce8f_0 | conda-forge |
| prompt-toolkit | 3.0.52 | pyha770c72_0 | conda-forge |
| psutil | 7.1.3 | py311hf893f09_0 | conda-forge |
| pthread-stubs | 0.3 | h3c9f919_1 | conda |
| pure_eval | 0.2.3 | pyhd8ed1ab_1 | conda-forge |
| pycparser | 2.23 | py311haa95532_0 | conda |
| pydantic | 2.12.5 | | pypi |
| pydantic-core | 2.41.5 | | pypi |
| pydantic-settings | 2.12.0 | | pypi |
| pygments | 2.19.2 | pyhd8ed1ab_0 | conda-forge |
| pyogrio | 0.12.1 | | pypi |
| pyparsing | 3.2.5 | pyhcf101f3_0 | conda-forge |
| pyproj | 3.7.2 | | pypi |
| pyqt6 | 6.6.1 | | pypi |
| pyqt6-qt6 | 6.6.1 | | pypi |
| pyqt6-sip | 13.10.3 | | pypi |
| pyqtgraph | 0.14.0 | | pypi |
| pysocks | 1.7.1 | py311haa95532_1 | conda |
| pytest | 9.0.2 | | pypi |
| python | 3.11.14 | h0159041_2_cpython | conda-forge |
| python-dateutil | 2.9.0.post0 | pyhe01879c_2 | conda-forge |
| python-dotenv | 1.2.1 | | pypi |
| python-tzdata | 2025.2 | pyhd3eb1b0_0 | conda |
| python_abi | 3.11 | 2_cp311 | conda-forge |
| pytorch-cuda | 12.4 | h3fd98bf_7 | pytorch |
| pytorch-mutex | 1.0 | cuda | pytorch |
| pyts | 0.13.0 | | pypi |
| pytz | 2025.2 | py311haa95532_0 | conda |
| pywin32 | 311 | py311hefeebc8_1 | conda-forge |
| pyyaml | 6.0.3 | py311h3f79411_0 | conda-forge |
| pyzmq | 27.1.0 | py311hb77b9c8_0 | conda-forge |
| rasterio | 1.4.4 | py311h54359c7_2 | conda-forge |
| rav1e | 0.7.1 | ha073cba_3 | conda-forge |
| requests | 2.32.5 | py311haa95532_1 | conda |
| rich | 14.2.0 | | pypi |
| sam-mosaic | 2.0.0 | | pypi |
| sam2 | 1.1.0 | | pypi |
| scikit-learn | 1.7.2 | py311h7f7e138_1 | conda |
| scipy | 1.16.3 | | pypi |
| seaborn | 0.13.2 | | pypi |
| setuptools | 80.9.0 | py311haa95532_0 | conda |
| shapely | 2.1.2 | py311h362461e_2 | conda-forge |
| sip | 6.12.0 | py311h706e071_0 | conda |
| sits | 0.1.0 | | pypi |
| six | 1.17.0 | pyhe01879c_1 | conda-forge |
| sleef | 3.5.1 | h8cc25b3_2 | conda |
| smart-open | 7.5.0 | | pypi |
| snappy | 1.2.2 | h7fa0ca8_1 | conda-forge |
| snuggs | 1.4.7 | pyhd8ed1ab_2 | conda-forge |
| spacy | 3.8.11 | | pypi |
| spacy-legacy | 3.0.12 | | pypi |
| spacy-loggers | 1.0.5 | | pypi |
| sqlite | 3.51.0 | hda9a48d_0 | conda |
| srsly | 2.5.2 | | pypi |
| stack_data | 0.6.3 | pyhd8ed1ab_1 | conda-forge |
| svt-av1 | 3.1.2 | hac47afa_0 | conda-forge |
| sympy | 1.13.1 | | pypi |
| tbb | 2021.8.0 | h59b6b97_0 | conda |
| termcolor | 3.2.0 | | pypi |
| thinc | 8.3.10 | | pypi |
| threadpoolctl | 3.5.0 | py311h4442805_1 | conda |
| tk | 8.6.15 | hf199647_0 | conda |
| torch | 2.6.0+cu124 | | pypi |
| torchaudio | 2.6.0+cu124 | | pypi |
| torchvision | 0.21.0+cu124 | | pypi |
| tornado | 6.5.3 | py311h3485c13_0 | conda-forge |
| tqdm | 4.67.1 | py311h4442805_1 | conda |
| traitlets | 5.14.3 | pyhd8ed1ab_1 | conda-forge |
| tsai | 0.4.1 | | pypi |
| tslearn | 0.7.0 | | pypi |
| typer-slim | 0.20.0 | | pypi |
| typing-extensions | 4.15.0 | h396c80c_0 | conda-forge |
| typing-inspection | 0.4.2 | | pypi |
| typing_extensions | 4.15.0 | pyhcf101f3_0 | conda-forge |
| tzdata | 2025b | h04d1e81_0 | conda |
| ucrt | 10.0.22621.0 | haa95532_0 | conda |
| uriparser | 0.9.8 | h5a68840_0 | conda-forge |
| urllib3 | 2.6.1 | py311haa95532_0 | conda |
| vc | 14.42 | haa95532_5 | conda |
| vc14_runtime | 14.44.35208 | h4927774_10 | conda |
| vs2015_runtime | 14.44.35208 | ha6b5a95_10 | conda |
| wasabi | 1.1.3 | | pypi |
| wcwidth | 0.2.14 | pyhd8ed1ab_0 | conda-forge |
| weasel | 0.4.3 | | pypi |
| wheel | 0.45.1 | py311haa95532_0 | conda |
| widgetsnbextension | 4.0.15 | | pypi |
| win32-setctime | 1.2.0 | | pypi |
| win_inet_pton | 1.1.0 | py311haa95532_1 | conda |
| wrapt | 2.0.1 | | pypi |
| x265 | 3.5 | h2d74725_3 | conda-forge |
| xerces-c | 3.3.0 | he0c23c2_0 | conda-forge |
| xorg-libxau | 1.0.12 | hba3369d_1 | conda-forge |
| xorg-libxdmcp | 1.1.5 | hba3369d_1 | conda-forge |
| xz | 5.8.1 | h208afaa_2 | conda-forge |
| xz-tools | 5.8.1 | h2466b09_2 | conda-forge |
| yaml | 0.2.5 | h6a83c73_3 | conda-forge |
| zeromq | 4.3.5 | h5bddc39_9 | conda-forge |
| zipp | 3.23.0 | py311haa95532_0 | conda |
| zlib | 1.3.1 | h02ab6af_0 | conda |
| zlib-ng | 2.3.2 | h5112557_0 | conda-forge |
| zstd | 1.5.7 | h534d264_6 | conda-forge |

---

## CUDA Information

- **CUDA Version:** 12.4
- **PyTorch CUDA:** 12.4
- **cuDNN:** Included via cuda-libraries

## Environment Location

```
C:\Users\Admin\anaconda3\envs\ts_annotator
```

## Recreation

To recreate this environment, export it with:

```bash
conda activate ts_annotator
conda env export > environment.yml
```

Or use pip to export Python packages:

```bash
pip freeze > requirements.txt
```

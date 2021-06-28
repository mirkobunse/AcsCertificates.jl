# Certification of Model Robustness in Active Class Selection

Supplementary material for our paper at ECML-PKDD 2021. Please see `supplement.pdf` for all plots that are generated here.

```bibtex
@InProceedings{bunse2021certification,
  author    = {Mirko Bunse and Katharina Morik},
  booktitle = {Europ. Conf. on Mach. Learn. and Knowledge Discovery in Databases (ECML-PKDD)},
  title     = {Certification of Model Robustness in Active Class Selection},
  year      = {2021},
  note      = {To appear},
  publisher = {Springer}
}
```


## Reproducing all plots

**Preliminaries:** You need to have `julia` and `pdflatex` installed. We conducted the experiments with Julia v1.5 and TexLive on Ubuntu.

From there on, simply calling `make` will generate a copy of the `supplement.pdf` in the `plot/` directory in about 3 hours.

**Caution:** The experiments will download [the open FACT data set](https://factdata.app.tu-dortmund.de/), which is about 5GB in size.

**Troubleshooting:** Sometimes the `Manifest.toml` file causes trouble; since the dependencies are already defined in the `Project.toml`, you should be able to safely delete the manifest file and try again without it.


## Adapting the experiments

You can alter the configurations in the `conf/` directory, e.g. try different seeds, classifiers or data sets. Calling `make` again will conduct the experiments with the altered configuration, then.


## Using our certificates elsewhere

This project can become a dependency in any other Julia project. Certificates can then be created by simply calling the constructor `Certificate(L, y_h, y; kwargs...)` with a loss function `L`, predictions `y_h`, and the validation ground-truths `y`. The keyword arguments provide additional parameters like delta and class weights; all details are given in the documentation of the construction. Many decomposable loss functions are already available through [the LossFunctions.jl package](https://github.com/JuliaML/LossFunctions.jl).

We commit ourselves to writing thin wrappers for Python and the command line to make ACS certification easily accessible in other programming languages, as well.

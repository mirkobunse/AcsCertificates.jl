# Certification of Model Robustness in Active Class Selection

Supplementary material for our submission to IAL 2021 and for our accepted paper at ECML-PKDD 2021.


## Interactive Adaptive Learning (IAL) 2021

We have added a build target for the CD diagrams presented in our IAL submission. Please see the section "Reproducing all plots" below for the details.


## ECML-PKDD 2021

Please see `supplement.pdf` for all plots we generated for our ECML-PKDD paper. Also check out [the release for this contribution](https://github.com/mirkobunse/AcsCertificates.jl/releases/tag/v0.1.0).

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

From there on, ...

- simply calling `make ecml21` will generate a copy of the `supplement.pdf` in the `plot/` directory in about 3 hours.
- calling `make ial21` will take about 1.5 hours to produce the CD diagrams of our IAL submission.

**Caution:** The experiments will download [the open FACT data set](https://factdata.app.tu-dortmund.de/), which is about 5GB in size.

**Troubleshooting:** Sometimes the `Manifest.toml` file causes trouble; since the dependencies are already defined in the `Project.toml`, you should be able to safely delete the manifest file and try again without it.


## Adapting the experiments

You can alter the configurations in the `conf/` directory, e.g. try different seeds, classifiers or data sets. Calling `make` again will conduct the experiments with the altered configuration, then.


## Using our certificates elsewhere

This project can become a dependency in any other Julia project. Certificates can then be created by simply calling the constructor `Certificate(L, y_h, y; kwargs...)` with a loss function `L`, predictions `y_h`, and the validation ground-truths `y`. The keyword arguments provide additional parameters like delta and class weights. Many decomposable loss functions are already available through [the LossFunctions.jl package](https://github.com/JuliaML/LossFunctions.jl).

```julia
# let y_val be validation labels and y_h be corresponding predictions
using AcsCertificates, LossFunctions
c = Certificate(ZeroOneLoss(), y_h, y_val)

?Certificate # inspect the documentation
```

We commit ourselves to writing thin wrappers for Python and the command line to make ACS certification easily accessible in other programming languages, as well.

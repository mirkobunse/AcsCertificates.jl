# Certification of Model Robustness in Active Class Selection

Supplementary material for two of our publications. The `supplement.pdf` contains all plots we have generated for [our ECML-PKDD paper](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_598.pdf).

```bibtex
@InProceedings{bunse2021certification,
  author    = {Mirko Bunse and Katharina Morik},
  booktitle = {Europ. Conf. on Mach. Learn. and Knowledge Discovery in Databases (ECML-PKDD)},
  title     = {Certification of Model Robustness in Active Class Selection},
  year      = {2021},
  note      = {To appear},
  publisher = {Springer}
}

@InProceedings{bunse2021active,
  author    = {Mirko Bunse and Katharina Morik},
  title     = {Active Class Selection with Uncertain Deployment Class Proportions},
  booktitle = {Workshop on Interactive Adaptive Learning},
  year      = {2021},
  note      = {To appear},
  publisher = {{CEUR} Workshop Proceedings}
}
```


## Reproducing all plots

**Preliminaries:** You need to have `julia` and `pdflatex` installed. We conducted the experiments with Julia v1.6 and TexLive on Linux.

In a terminal,

- calling `make ecml21` will generate a copy of the `supplement.pdf` in the `plot/` directory in about 3 hours.
- calling `make ial21` will take about 2 hours to produce the plots of our IAL contribution.
- calling `make` without arguments will conduct the experiments of both papers.
- calling `make -n <whatever>` will show the steps taken without actually taking them (dry run).

**Caution:** The experiments will download [the open FACT data set](https://factdata.app.tu-dortmund.de/), which is about 5GB in size.

**Troubleshooting:** Sometimes the `Manifest.toml` file causes trouble; since the dependencies are already defined in the `Project.toml`, you should be able to safely delete the manifest file and try again without it.


## Adapting the experiments

You can alter the configurations in the `conf/` directory, e.g. try different seeds, classifiers, or data sets. Calling `make` again will then conduct the experiments with the altered configuration.

More severe adaptations are possible through changing the code. To this end, the Julia modules in the `src/` directory separate concerns in the following ways:

- `AcsCertificates` is the top-level module that nests all other modules.
- `Certificates` implements our proposals: certificates for active class selection and the data acquisition strategy that is based on these certificates.
- `Data` downloads and reads the data and provides some data-related utility functions.
- `Experiments` take configuration files as inputs and produce raw results, i.e. they write all evaluations of all trials to an output CSV file.
- `Plots` take the raw results as inputs, aggregate them, and produce TEX files with code for plots.
- `SigmaTest` is a wrapper for an evaluation measure that is implemented at https://bitbucket.org/mbunse/sigma-test.


## Using our certificates elsewhere

This project can become a dependency in any other Julia project. Certificates can then be created by simply calling the constructor `Certificate(L, y_h, y; kwargs...)` with a loss function `L`, predictions `y_h`, and the validation ground-truths `y`. The keyword arguments provide additional parameters like delta and class weights. Many decomposable loss functions are already available through [the LossFunctions.jl package](https://github.com/JuliaML/LossFunctions.jl).

```julia
# let y_val be validation labels and y_h be corresponding predictions
using AcsCertificates, LossFunctions
c = Certificate(ZeroOneLoss(), y_h, y_val)

?Certificate # inspect the documentation
```

## Support

If you encounter any problem, please file a GitHub issue.

For the near future, we commit ourselves to writing thin wrappers for Python and the command line to make ACS certification easily accessible in other programming languages, as well.

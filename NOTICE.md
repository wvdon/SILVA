# Third-party code

SILVA subclasses `scDisInFact.model.scdisinfact` (encoders, classifiers, priors, MMD/KL/group-lasso losses, dataset class) and
imports it at run time. It is **not copied** into this repository; install it from the pinned commit

    pip install "git+https://github.com/ZhangLabGT/scDisInFact.git@9466ecf257d923f879cb39712c530d24826d88bc"

scDisInFact (Zhang Lab, Georgia Tech) is released under the **GNU GPL v3.0**. Please cite Zhang et al., scDisInFact
(bioRxiv 2023, doi:10.1101/2023.05.01.538975) together with this work.

Choose this repository's license with that dependency in mind; see the note in README.md.

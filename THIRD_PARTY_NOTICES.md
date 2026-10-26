# Third-party notices

This repository retains the existing root MIT license and its attribution.
That license does not replace the licenses of incorporated upstream code,
pretrained weights, or datasets.

* `models/chronos_bolt/` incorporates and adapts Chronos-Bolt pipeline/model
  code from [Chronos](https://github.com/amazon-science/chronos-forecasting)
  and [AutoGluon](https://github.com/autogluon/autogluon), whose upstream code
  is distributed under Apache License 2.0. Existing author notices are retained.
  The component quantizer and adapter are project additions to this code path.
* Benchmark data loading and time features were inherited through
  [TS-RAG](https://github.com/UConn-DSIS/TS-RAG). The existing root MIT
  attribution to the UConn DSIS Research Lab is preserved.
* The public [Chronos-Bolt-base model card](https://huggingface.co/amazon/chronos-bolt-base)
  specifies Apache License 2.0 for the pretrained model. Model files are
  downloaded separately and are not included in this repository.

The Apache License 2.0 text is included in `licenses/Apache-2.0.txt`. Consult
the corresponding upstream repositories for their complete notices and any
additional distribution terms. This cleanup does not claim new authorship of
the incorporated backbone or change its original license.

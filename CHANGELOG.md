Released
--------

0.4.3 - 17 April 2024
===================

- Fixed mistake in documentation

0.4.2 - 28 October 2023
===================

Added
-------
Feature flags (all enabled by default)
- parallel: Enables parallel iterators
- avx/sse/neon: Enables rustfft's feature flags

0.4.1 - 23 June 2023
===================

Bugfix
-------
- Fixed normalization of rfft for inputs with odd number of elements

0.4 - 15 March 2023
===================

Changed
-------
- Add support for non-standard layout input arrays.
- Add support for custom fft normalization. See `examples/fft_norm`.
- Remove specification of the minor version, use `rustfft` v6
- Remove specification of the minor version, use `realfft` v3
- Better performance

0.3.0
=====

Changed
-------
- Update of `realfft` and `realdct`

0.2.2
=====

Changed
-------
- The undefined first and last elements of real-to-complex transforms are now actively set to zero


Unreleased
-----------

0.4 - 5 August 2022
===================

Fixed
-----
- Error on arrays that are not standard layout. `ndrustfft` should now be able to deal with different `ndarray` layouts.

Changed
-------
- Add option for different normalizations. See `examples/fft_norm`. Default normalization is, as in previous versions, based on numpys normalization.


Released
--------

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


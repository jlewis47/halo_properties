# halo-properties

Requires Healpy, Numpy, Scipy, Pyplot
Compute halo properties using PHEW/FOF and RAMSES-CUDATON outputs


Setup paths to assoc_outs (Where the star/halo association files go) and to analysis_outs (where the final halo property files go)
as desired

python star_halo_latest.py <XXX> <box size> <path to sim> <name for output dir> -- This associates star particles to haloes

python halo_fesc_latest.py <XXX> <box size> <path to sim> <name for output dir> -- This computes gas and dust fesc, magnitude,
magnitude with extinction, SFR, stellar mass, beta, beta with extinction

XXX should be replaced with the last 3 numbers of the target RAMSES output folder

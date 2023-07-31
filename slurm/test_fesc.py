from halo_properties.halo_stats.halo_fesc_latest_big_boxes import compute_fesc

Tr, mags, fof_tab, gmass, mass = compute_fesc(65, assoc_mthd="stellar_peak", test=True, dilate=1)

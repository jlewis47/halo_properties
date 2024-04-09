# halo-properties

Requires Healpy, Numpy, Scipy, pyplot, mpi4py
Compute halo properties using PHEW/FOF and RAMSES-CUDATON outputs


Examples for running the code can be found in the /slurm directory


## Quick examples:

### Read the associations
You can use the functions in association/read_assoc_latest.py to access those like this:


    from halo_properties.association.read_assoc_latest import *
    from halo_properties.utils.output_paths import dataset
    outnb = 106
    dset = dataset(ll=0.2, r200=1.0, mp=False, clean=True)
    fofs, halo_star_ids, cum_star_nb = read_assoc(
            outnb,
            "CoDaIII",
            dset,
            # mass_cut=mass_cut / Mp,
            st_mass_cut=(10**7, 10**10),
            bounds=[
                [ymin, ymax],
                [xmin, xmax],
            ],
        )


in this example, I'm taking a subbox defined by xmin,xmax,ymin,ymax in cells units (0,8192)

and a stellar mass interval (the commented argument also applies a halo mass cut or interval in number of particles)


Datasets are objects for handling the parameters and getting the file paths
the clean=True argument is important as it means that stars only belong to the halo that they are the closest to
ll is the fof linking length.

This can be pretty slow if you load the whole dataset because of the number of halos (especially for the later data outputs).

If you need to run associations for snapshots or linking lengths that I don't have, you can look in halo_properties/slurm, and use run_association_stpeak.slurm (take halo centres as the local peak of stellar density)
and then also run_clean_assoc.slurm to remove duplicate stellar associations. Otherwise you can ask me and I'll queue them up for you.

### Reading stellar data for associated stars
You can read all the stellar data for the association after loading the fof list like above, using:

halo_star_nb = fofs['nstar']

for ind, halo in enumerate(fofs):

                        cur_star_ids = halo_star_ids[
                            cum_star_nb[ind]
                            - halo_star_nb[ind] :cum_star_nb[ind]
                        ]

                        cur_stars = read_specific_stars(
                            os.path.join(star_path, output_str),
                            cur_star_ids,
                            keys=["mass", "age", "Z/0.02", "x", "y", "z"],
                        )

For all the associated stars I save a custom id that I can decompose into star file and file line, so the read is fast and only of the required particles.

## Output paths

Currently all the scripts are designed to write to /lustre/orion/proj-shared/ast031/jlewis/CoDaIII_analysis, but you can change that in halo_properties/utils/output_paths.py

## Further examples

There are some more examples in the analysis scripts in halo_properties/halo_stats e.g. if you want to read the gas data


from halo_properties.halo_stats.stellar_part_fesc import compute_stellar_Tr
import argparse

Arg_parser = argparse.ArgumentParser("Associate stars and halos in full simulattion")

Arg_parser.add_argument("out_nb", type=int, help="output number")

Arg_parser.add_argument("--sub_nb", type=int, help="sub box number", default=None)

Arg_parser.add_argument(
    "--star_sub_l",
    type=int,
    help="side length of star file sub boxes",
    default=2048,
)

Arg_parser.add_argument(
    "--overd_fact",
    type=float,
    help="overdensity threshold for integration radius",
    default=50,
)

Arg_parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing files?",
    default=False,
)

args = Arg_parser.parse_args()

compute_stellar_Tr(**vars(args))

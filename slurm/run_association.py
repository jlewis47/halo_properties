from halo_properties.association.star_halo_latest import assoc_stars_to_haloes
import argparse


Arg_parser = argparse.ArgumentParser("Associate stars and halos in full simulation")

Arg_parser.add_argument(
    "nb",
    metavar="nsnap",
    type=int,
    help='snap number string, give as "XXX" so that 00001 is "001" and 00111 is "111"',
)

Arg_parser.add_argument(
    "--rtwo_fact",
    metavar="rtwo_fact",
    type=float,
    help="1.0 -> associate stellar particles within 1xR200 for all haloes",
    default=1,
)
Arg_parser.add_argument(
    "--npart_thresh",
    metavar="npart_thresh",
    type=float,
    help="dark matter particle number threshold for halos",
    default=50,
)
Arg_parser.add_argument(
    "--assoc_mthd",
    metavar="assoc_mthd",
    type=str,
    help="method for linking stars to fof",
    default="",
)
Arg_parser.add_argument(
    "--overwrite",
    action="store_true",
    help="overwrite existing files?",
    default=False,
)
Arg_parser.add_argument(
    "--binary_fof",
    action="store_true",
    help="fof in binary format?",
    default=False,
)
Arg_parser.add_argument(
    "--mp",
    action="store_true",
    help="fof with watershed segmentation by Mei Palanque",
    default=False,
)
Arg_parser.add_argument(
    "--ll",
    type=float,
    help="fof linking length parameter",
    default=0.2,
)

args = Arg_parser.parse_args()

out_nb = args.nb
rtwo_fact = args.rtwo_fact
npart_thresh = args.npart_thresh
assoc_mthd = args.assoc_mthd

assert not (args.binary_fof and args.mp), "Can't have both binary and mp segmentation"

assoc_stars_to_haloes(
    out_nb,
    npart_thresh=npart_thresh,
    rtwo_fact=rtwo_fact,
    assoc_mthd=assoc_mthd,
    overwrite=args.overwrite,
    binary_fof=args.binary_fof,
    mp_fof=args.mp,
    ll=args.ll,
)

from halo_properties.halo_stats.halo_fesc_latest_big_boxes import compute_fesc
import argparse

Arg_parser = argparse.ArgumentParser("Compute gas and stellar properties in halos")

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
    "--fesc_rad",
    metavar="fesc_rad",
    type=float,
    help="1.0 -> use association radius as integration limit for fesc computation",
    default=1,
)
Arg_parser.add_argument(
    "--ll", metavar="ll", type=float, help="linking length for fof", default=0.2
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
    help="When used, code overwrites all found data",
    default=False,
)
Arg_parser.add_argument(
    "--test",
    action="store_true",
    help="When used, code runs on one subcube and doesn't write",
    default=False,
)
Arg_parser.add_argument(
    "--dilate",
    type=int,
    help="number of times to resample grid when performing sums within r_px",
    default=8,
)
Arg_parser.add_argument(
    "--mp",
    help="Use MP's watershed segmented fof haloes",
    action="store_true",
    default=False,
)

Arg_parser.add_argument(
    "--rstar", metavar="rstar", type=float, help="rstar * fesc_rad * r200 is the radius within with fesc los can start (so if rstar <1, \
    we don't compute fesc using the stars r>rstar * fesc_rad * r200). Only accounted for when fesc_rad * r200 > 2", default=1
)

Arg_parser.add_argument(
    "--sub_nb",
    type=int,
    help="When used, run and sve result for one 512^3 cell subcube",
    default=None,
)

args = Arg_parser.parse_args()

out_nb = args.nb
rtwo_fact = args.rtwo_fact
assoc_mthd = args.assoc_mthd
ll = args.ll
overwrite = args.overwrite
dilate = args.dilate
fesc_rad = args.fesc_rad
rstar = args.rstar
sub_nb = args.sub_nb

compute_fesc(
    out_nb,
    rtwo_fact=rtwo_fact,
    fesc_rad=fesc_rad,
    assoc_mthd=assoc_mthd,
    ll=ll,
    overwrite=overwrite,
    test=args.test,
    dilate=dilate,
    mp=args.mp,
    rstar=rstar,
    subnb=sub_nb,
)

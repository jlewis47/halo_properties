from halo_properties.halo_stats.stellar_part_pties import compute_part_pties
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
    default=1
)
Arg_parser.add_argument(
    "--mbin",
    metavar="mbin",
    type=float,
    help="halo mass bin centre",
    default=1
)
Arg_parser.add_argument(
    "--mbin_width",
    metavar="mbin_width",
    type=float,
    help="halo mass bin centre",
    default=1
)

Arg_parser.add_argument(
    "--ll", metavar="ll", type=float, help="linking length for fof", default=0.2
)
Arg_parser.add_argument(
    "--assoc_mthd",
    metavar="assoc_mthd",
    type=str,
    help="method for linking stars to fof",
    default=""
)
Arg_parser.add_argument(
    "--overwrite",
    action="store_true",
    help="When used, code overwrites all found data",
    default=False
)
Arg_parser.add_argument(
    "--test",
    action="store_true",
    help="When used, code runs on one subcube and doesn't write",
    default=False
)

args = Arg_parser.parse_args()

out_nb = args.nb
rtwo_fact = args.rtwo_fact
assoc_mthd = args.assoc_mthd
ll = args.ll
overwrite = args.overwrite

compute_part_pties(
    out_nb,
    rtwo_fact=rtwo_fact,
    assoc_mthd=assoc_mthd,
    ll=ll,
    overwrite=overwrite,
    test=args.test,
    mbin=args.mbin,
    mbin_width=args.mbin_width,
)

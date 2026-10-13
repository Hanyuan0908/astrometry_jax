"""Physical constants in the unit system used throughout: au, day, Msun, mas, km/s."""
import numpy as np

# G * Msun in au^3 / day^2 (IAU 2015 nominal: GM_sun = 1.3271244e20 m^3 s^-2)
GM_SUN_AU3_DAY2 = 2.959122082855911e-4
AU_KM = 1.495978707e8
DAY_S = 86400.0
AU_PER_DAY_TO_KM_S = AU_KM / DAY_S  # multiply au/day by this to get km/s
TWO_PI = 2.0 * np.pi
DEG = np.pi / 180.0
MAS_PER_DEG = 3.6e6

# Gaia DR3 reference epoch: J2016.0 = JD 2457389.0 (TCB). t_periastron is days after this.
GAIA_REF_JD = 2457389.0

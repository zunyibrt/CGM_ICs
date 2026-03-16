"""Float-native cooling tables used by the steady-state solver."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy import interpolate

import numpy_compat  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "cooling" / "Wiersma09_CoolingTables"


def _broadcast_shape(*values):
    return np.broadcast(*[np.asarray(value, dtype=float) for value in values]).shape


def _safe_log(values):
    return np.log(np.clip(np.asarray(values, dtype=float), np.finfo(float).tiny, None))


def _evaluate_interpolator(interpolator, x, y):
    x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    points = np.column_stack([x_arr.ravel(), y_arr.ravel()])
    values = np.asarray(interpolator(points), dtype=float).reshape(x_arr.shape)
    if values.shape == ():
        return float(values)
    return values


def _closest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(target))))


class Constant_Cooling:
    """Density-independent constant cooling function."""

    def __init__(self, lambda_cgs_value: float):
        self.lambda_cgs_value = float(lambda_cgs_value)

    def lambda_cgs(self, temperature_K, nH_cm3):
        shape = _broadcast_shape(temperature_K, nH_cm3)
        return np.full(shape, self.lambda_cgs_value, dtype=float)

    def dln_lambda_dln_T(self, temperature_K, nH_cm3):
        return np.zeros(_broadcast_shape(temperature_K, nH_cm3), dtype=float)

    def dln_lambda_dln_rho(self, temperature_K, nH_cm3):
        return np.zeros(_broadcast_shape(temperature_K, nH_cm3), dtype=float)


class Wiersma_Cooling:
    """Wiersma et al. (2009) cooling tables in float-native form."""

    def __init__(self, Z2Zsun: float, z: float):
        files = []
        redshifts = []
        for table_path in sorted(DATA_DIR.glob("z_*.hdf5")):
            label = table_path.stem.removeprefix("z_")
            try:
                redshifts.append(float(label))
                files.append(table_path)
            except ValueError:
                continue

        if not files:
            raise FileNotFoundError(f"no Wiersma cooling tables found under {DATA_DIR}")

        redshifts = np.asarray(redshifts, dtype=float)
        table_path = files[_closest_index(redshifts, z)]

        with h5py.File(table_path, "r") as handle:
            helium_to_hydrogen = 10.0 ** -1.07 * (0.71553 + 0.28447 * Z2Zsun)
            hydrogen_mass_fraction = (1.0 - 0.014 * Z2Zsun) / (1.0 + 4.0 * helium_to_hydrogen)
            helium_mass_fraction = 4.0 * helium_to_hydrogen * hydrogen_mass_fraction

            helium_bins = np.asarray(handle["Metal_free"]["Helium_mass_fraction_bins"][...], dtype=float)
            helium_index = _closest_index(helium_bins, helium_mass_fraction)

            hydrogen_helium = np.asarray(handle["Metal_free"]["Net_Cooling"][helium_index, ...], dtype=float)
            temperature_bins = np.asarray(handle["Metal_free"]["Temperature_bins"][...], dtype=float)
            hydrogen_density_bins = np.asarray(handle["Metal_free"]["Hydrogen_density_bins"][...], dtype=float)
            metal_cooling = np.asarray(handle["Total_Metals"]["Net_cooling"][...], dtype=float) * float(Z2Zsun)

        log_temperature = _safe_log(temperature_bins)
        log_density = _safe_log(hydrogen_density_bins)
        cooling_table = metal_cooling + hydrogen_helium
        cooling_table = np.clip(cooling_table, np.finfo(float).tiny, None)

        self._cooling_interpolator = interpolate.RegularGridInterpolator(
            (log_temperature, log_density),
            cooling_table,
            bounds_error=False,
            fill_value=None,
        )

        dlogT = float(np.mean(np.diff(log_temperature)))
        dlogn = float(np.mean(np.diff(log_density)))
        log_cooling = np.log(cooling_table)
        dln_lambda_dln_T, dln_lambda_dln_rho = np.gradient(log_cooling, dlogT, dlogn, edge_order=2)

        self._dln_lambda_dln_T_interpolator = interpolate.RegularGridInterpolator(
            (log_temperature, log_density),
            dln_lambda_dln_T,
            bounds_error=False,
            fill_value=None,
        )
        self._dln_lambda_dln_rho_interpolator = interpolate.RegularGridInterpolator(
            (log_temperature, log_density),
            dln_lambda_dln_rho,
            bounds_error=False,
            fill_value=None,
        )

    @classmethod
    def from_config(cls, config):
        return cls(Z2Zsun=config.metallicity_solar, z=config.redshift)

    def lambda_cgs(self, temperature_K, nH_cm3):
        return _evaluate_interpolator(
            self._cooling_interpolator,
            _safe_log(temperature_K),
            _safe_log(nH_cm3),
        )

    def dln_lambda_dln_T(self, temperature_K, nH_cm3):
        return _evaluate_interpolator(
            self._dln_lambda_dln_T_interpolator,
            _safe_log(temperature_K),
            _safe_log(nH_cm3),
        )

    def dln_lambda_dln_rho(self, temperature_K, nH_cm3):
        return _evaluate_interpolator(
            self._dln_lambda_dln_rho_interpolator,
            _safe_log(temperature_K),
            _safe_log(nH_cm3),
        )


class Kartick_Cooling:
    """One-dimensional CIE cooling table used for comparison runs."""

    table_path = BASE_DIR / "cooling" / "Kartick_CIE_cooling.table"

    def __init__(self):
        table = np.genfromtxt(self.table_path, dtype=float)
        self.temperature_bins_K = table[:, 0]
        # Table stores Lambda * n_e. The original code converts this to the
        # n_H^2 Lambda convention used throughout the solver.
        self.lambda_bins_cgs = table[:, 1] * table[:, 2]

        log_temperature = _safe_log(self.temperature_bins_K)
        dlogT = np.diff(log_temperature)
        log_lambda = np.log(np.clip(self.lambda_bins_cgs, np.finfo(float).tiny, None))
        self._gradient_temperature = 0.5 * (log_temperature[1:] + log_temperature[:-1])
        self._gradient_values = np.diff(log_lambda) / dlogT

    def lambda_cgs(self, temperature_K, nH_cm3=None):
        temperature = np.asarray(temperature_K, dtype=float)
        values = np.interp(
            _safe_log(temperature),
            _safe_log(self.temperature_bins_K),
            np.log(np.clip(self.lambda_bins_cgs, np.finfo(float).tiny, None)),
        )
        return np.exp(values)

    def dln_lambda_dln_T(self, temperature_K, nH_cm3=None):
        return np.interp(_safe_log(temperature_K), self._gradient_temperature, self._gradient_values)

    def dln_lambda_dln_rho(self, temperature_K, nH_cm3=None):
        return np.zeros(np.shape(np.asarray(temperature_K, dtype=float)), dtype=float)


class DopitaSutherland_CIE:
    """Dopita & Sutherland (1996) CIE cooling table."""

    table_path = BASE_DIR / "cooling" / "DopitaSutherland_CIE.dat"

    def __init__(self, Z2Zsun: float):
        table = np.genfromtxt(self.table_path, dtype=float)
        self.temperature_bins_K = table[:, 0]
        if Z2Zsun == 1.0:
            lambda_bins = table[:, 1]
        elif Z2Zsun == 1.0 / 3.0:
            lambda_bins = table[:, 2]
        else:
            raise ValueError("DopitaSutherland_CIE only supports Z/Zsun = 1 or 1/3")

        n_i_to_n_H = 1.22 ** -1 / 0.7
        n_e_to_n_H = 1.17 ** -1 / 0.7
        self.lambda_bins_cgs = lambda_bins * n_i_to_n_H * n_e_to_n_H

        log_temperature = _safe_log(self.temperature_bins_K)
        dlogT = np.diff(log_temperature)
        log_lambda = np.log(np.clip(self.lambda_bins_cgs, np.finfo(float).tiny, None))
        self._gradient_temperature = 0.5 * (log_temperature[1:] + log_temperature[:-1])
        self._gradient_values = np.diff(log_lambda) / dlogT

    def lambda_cgs(self, temperature_K, nH_cm3=None):
        return np.exp(
            np.interp(
                _safe_log(temperature_K),
                _safe_log(self.temperature_bins_K),
                np.log(np.clip(self.lambda_bins_cgs, np.finfo(float).tiny, None)),
            )
        )

    def dln_lambda_dln_T(self, temperature_K, nH_cm3=None):
        return np.interp(_safe_log(temperature_K), self._gradient_temperature, self._gradient_values)

    def dln_lambda_dln_rho(self, temperature_K, nH_cm3=None):
        return np.zeros(np.shape(np.asarray(temperature_K, dtype=float)), dtype=float)

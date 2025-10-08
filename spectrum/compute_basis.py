# compute_basis.py
# Mallett and Yuksel 2019 の手法の追実装
# 著者らの supplemental に Matlab ベースのコードがあるのでそれを参考に実装

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linprog, minimize, Bounds, LinearConstraint

wavelengths = np.arange(390, 831, 5) # D65 + LMS 2006

M_XYZ_FROM_LMS = np.array([
    [1.94735469, -1.41445123, 0.36476327],
    [0.68990272,  0.34832189, 0.00000000],
    [0.00000000,  0.00000000, 1.93485343],
], dtype=float)

DATA_DIR = Path(".") # もし他の場所に CSV ファイルがあるなら変更すること

# ファイル名
FN_C        = "CIE_illum_C.csv"
FN_D65      = "CIE_std_illum_D65.csv"
FN_LMS2006  = "CIE_lms_cf_2deg.csv"
FN_XYZ1931  = "CIE_xyz_1931_2deg.csv"


OUT_BASIS_CSV   = "basis_rgb.csv"
OUT_XYZ_CSV     = "xyzbar.csv"
OUT_ILLUM_CSV     = "illumination.csv"

# -----------------------------
# ユーティリティ
# -----------------------------

def load_csv(fname: str):
    p = DATA_DIR / fname
    arr = np.loadtxt(p, delimiter=",", dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    N, M = arr.shape
    xs = None
    if M >= 2:
        col0 = arr[:, 0]
        if np.all(np.isfinite(col0)) and np.all(np.diff(col0) > 0):
            xs = col0
            vals = arr[:, 1:]
        else:
            vals = arr
    else : 
        vals = arr
    return xs, vals

def lms_to_xyzbar(lms: np.ndarray, M_LMS2XYZ: np.ndarray) -> np.ndarray:
    assert lms.shape[1] == 3
    return lms @ M_LMS2XYZ.T

def resample_spectrum(spectrum : np.ndarray, xs0 : np.ndarray, xs1 : np.ndarray, kind="pchip") -> np.ndarray:
    spectrum = np.asarray(spectrum)
    if spectrum.ndim == 1:
        spectrum = spectrum[: None]
    if spectrum.shape[0] != xs0.size and spectrum.shape[1] == xs0.size:
        spectrum = spectrum.T
    if spectrum.shape[0] != xs0.size:
        raise AssertionError(f"Spectrum rows ({spectrum.shape[0]}) must match xs0 length ({xs0.size}).")
    cols = spectrum.shape[1]
    out = np.zeros((xs1.size, cols), dtype=float)
    
    for j in range(cols):
        if kind == "linear":
            f = interp1d(
                xs0, 
                spectrum[:, j], 
                kind="cubic", 
                bounds_error=False, 
                fill_value=0.0, 
                assume_sorted=True
            )
            out[:, j] = f(xs1)
        else:
            f = PchipInterpolator(xs0, spectrum[:, j], extrapolate=False)
            out[:, j]= np.nan_to_num(f(xs1), nan=0.0, posinf=0.0, neginf=0.0)
    return out

def calc_XYZ(xyzbar: np.ndarray, spectrum : np.ndarray) -> np.ndarray:
    spectrum = spectrum.reshape(-1)
    return xyzbar.T @ spectrum

def calc_matr_RGBtoXYZ(
        xy_r: Tuple[float, float],
        xy_g: Tuple[float, float],
        xy_b: Tuple[float, float],
        XYZ_W: np.ndarray
    ) -> np.ndarray:
    
    x_rgb = np.array([xy_r[0], xy_g[0], xy_b[0]], dtype=float).reshape(3, 1)
    y_rgb = np.array([xy_r[1], xy_g[1], xy_b[1]], dtype=float).reshape(3, 1)

    X_rgb = x_rgb / y_rgb
    Y_rgb = np.ones((3, 1))
    Z_rgb = (1.0 - x_rgb - y_rgb) / y_rgb

    M = np.vstack([X_rgb.T, Y_rgb.T, Z_rgb.T])  # 3x3 行列
    S = np.linalg.solve(M, XYZ_W.reshape(3, 1)) # 3x1

    M_rgb2xyz = (np.hstack([X_rgb, Y_rgb, Z_rgb]) * S).T
    return M_rgb2xyz

@dataclass
class BasisSolution:
    wavelengths : np.ndarray
    basis_r     : np.ndarray
    basis_g     : np.ndarray
    basis_b     : np.ndarray
    M_RGB2XYZ   : np.ndarray
    M_XYZ2RGB   : np.ndarray


# -----------------------------
# ソルバ
# -----------------------------

def solve_basis(
        wavelength  : np.ndarray,
        xyzbar      : np.ndarray,
        M_RGB2XYZ   : np.ndarray,
        whitept     : np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    n = wavelength.size
    row0 = (whitept * xyzbar[:, 0])
    row1 = (whitept * xyzbar[:, 1])
    row2 = (whitept * xyzbar[:, 2])

    Aeq = np.zeros((9, 3 * n), dtype=float)
    Aeq[0, 0:n]     = row0; Aeq[1, 0:n]     = row1; Aeq[2, 0:n]     = row2
    Aeq[3, n:2*n]   = row0; Aeq[4, n:2*n]   = row1; Aeq[5, n:2*n]   = row2
    Aeq[6, 2*n:3*n] = row0; Aeq[7, 2*n:3*n] = row1; Aeq[8, 2*n:3*n] = row2

    beq = np.hstack([
        M_RGB2XYZ @ np.array([1.0, 0.0, 0.0]),
        M_RGB2XYZ @ np.array([0.0, 1.0, 0.0]),
        M_RGB2XYZ @ np.array([0.0, 0.0, 1.0]),
    ]).astype(float).ravel()

    #Aineq = np.hstack([np.eye(n), np.eye(n), np.eye(n)])
    Aineq = None
    #bineq = np.ones(n, dtype=float).ravel()
    bineq = None

    lb = np.zeros(3 * n, dtype=float)
    ub = np.ones(3 * n, dtype=float)
    bounds_lp = [(0.0, 1.0)] * (3 * n)
    bounds_nl = Bounds(lb, ub)

    c = np.ones(3 * n, dtype=float)
    linres = linprog(
        c=c,
        A_ub=Aineq, b_ub=bineq,
        A_eq=Aeq,   b_eq=beq,
        bounds=bounds_lp,
        method="highs",
        options={"disp": True}
    )

    if not linres.success:
        raise RuntimeError(f"linprog falied: {linres.message}")
    x0 = linres.x.copy()

    p =128.0
    

    def objective(rgb : np.ndarray) -> float:
        r = rgb[0:n]
        g = rgb[n:2*n]
        b = rgb[2*n:3*n]

        def lp(v) : return np.power(np.sum(np.abs(v) ** p), 1.0 / p)
        return lp(r) + lp(g) + lp(b)
    
    eq_con = LinearConstraint(Aeq, beq, beq)
    Aineq_nl = np.hstack([np.eye(n), np.eye(n), np.eye(n)])
    bineq_nl = np.ones(n, dtype=float)
    ineq_con = LinearConstraint(Aineq_nl, -np.inf, bineq_nl)

    opt = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds_nl,
        constraints=[eq_con, ineq_con],
        options=dict(
            maxiter=10000,
            ftol=1e-12,
            disp=True,
        ),
    )

    if not opt.success:
        print(f"[warn] SLSQP status={opt.status}, message={opt.message}")

    rgb = opt.x
    r = rgb[0:n].copy()
    g = rgb[n:2*n].copy()
    b = rgb[2*n:3*n].copy()
    return r, g, b

def check_basis(
        xyzbar      : np.ndarray,
        M_XYZ2RGB   : np.ndarray,
        basis_r     : np.ndarray,
        basis_g     : np.ndarray,
        basis_b     : np.ndarray,
        D65         : np.ndarray
    ) -> np.ndarray :
    
    RGB_d65r    = M_XYZ2RGB @ calc_XYZ(xyzbar, basis_r * D65)
    RGB_d65g    = M_XYZ2RGB @ calc_XYZ(xyzbar, basis_g * D65)
    RGB_d65b    = M_XYZ2RGB @ calc_XYZ(xyzbar, basis_b * D65)
    RGB_d65rgb  = M_XYZ2RGB @ calc_XYZ(xyzbar, (basis_r + basis_g + basis_b) * D65)

    M = np.column_stack([RGB_d65r, RGB_d65g, RGB_d65b, RGB_d65rgb])
    target = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=float)
    
    resid = M - target
    return resid

# -----------------------------
# Main
# -----------------------------
def main():
    
    # 光源情報のデータの読み込み
    # xsC, C      = load_csv(FN_C)
    xsD65, D65  = load_csv(FN_D65)

    # 光源情報のデータを wavelength のサイズと一致させるためにリサンプリング
    # C_rs    = resample_spectrum(C, xsC, wavelengths)
    D65_rs  = resample_spectrum(D65, xsD65, wavelengths)

    # LMS 関数の読み込みと等色関数への変換
    # xs_lms, lms_full = load_csv(FN_LMS2006)
    # lms_rs      = resample_spectrum(lms_full, xs_lms, wavelengths)
    # xyzbar = lms_to_xyzbar(lms_rs, M_XYZ_FROM_LMS)

    # XYZ 関数の読みこみ
    xsXYZ, XYZ  = load_csv(FN_XYZ1931)
    xyzbar      = resample_spectrum(XYZ, xsXYZ, wavelengths)
    
    np.savetxt(
        OUT_ILLUM_CSV, 
        np.column_stack([wavelengths, D65_rs]), 
        delimiter=",", 
        header="wavelength(nm), D65 illuminant", 
        comments=""
    )
    print(f"Saved: {OUT_ILLUM_CSV}")
    
    np.savetxt(
        OUT_XYZ_CSV, 
        np.column_stack([wavelengths, xyzbar]), 
        delimiter=",", 
        header="wavelength(nm), xbar, ybar, zbar", 
        comments=""
    )
    print(f"Saved: {OUT_XYZ_CSV}")
    
    # RGB <-> XYZ の行列を作成 (BT.709 色域, D65 光源)
    xy_r = (0.64, 0.33)
    xy_g = (0.30, 0.60)
    xy_b = (0.15, 0.06)

    # 白色点の計算
    XYZ_W = calc_XYZ(xyzbar, D65_rs.reshape(-1))
    M_RGB2XYZ = calc_matr_RGBtoXYZ(xy_r, xy_g, xy_b, XYZ_W)
    M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)

    basis_r, basis_g, basis_b = solve_basis(
        wavelengths, xyzbar, M_RGB2XYZ, D65_rs.reshape(-1)
    )

    resid = check_basis(xyzbar, M_XYZ2RGB, basis_r, basis_g, basis_b, D65_rs.reshape(-1))
    print("Basis residual [(RGB columns) and RGB-sum column] :\n", np.round(resid, 12))

    out = np.column_stack([basis_r, basis_g, basis_b])
    np.savetxt(
        OUT_BASIS_CSV, 
        np.column_stack([wavelengths, out]), 
        delimiter=",", 
        header="wavelength(nm), basis_r, basis_g, basis_b", 
        comments="")
    print(f"saved: {OUT_BASIS_CSV}")


if __name__  == "__main__":
    main()

    

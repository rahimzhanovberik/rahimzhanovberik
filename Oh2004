
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import os

folder = r"C:\Users\rahim\Desktop\Data set June\Spain.data"

C11_path = os.path.join(folder, "C11.img")
C22_path = os.path.join(folder, "C22.img")
C12_real_path = os.path.join(folder, "C12_real.img")
C12_imag_path = os.path.join(folder, "C12_imag.img")
theta_path = os.path.join(folder, "localIncidenceAngle.img")

def read_envi(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Ошибка при открытии файла: {path}")
    return ds.ReadAsArray(), ds

C11, ref_ds = read_envi(C11_path)
C22, _ = read_envi(C22_path)
C12_real, _ = read_envi(C12_real_path)
C12_imag, _ = read_envi(C12_imag_path)
theta_deg, _ = read_envi(theta_path)

C12 = C12_real + 1j * C12_imag
C2_matrix = np.zeros((C11.shape[0], C11.shape[1], 2, 2), dtype=np.complex64)
C2_matrix[:, :, 0, 0] = C11
C2_matrix[:, :, 0, 1] = C12
C2_matrix[:, :, 1, 0] = np.conj(C12)
C2_matrix[:, :, 1, 1] = C22

def compute_mv_vegetation_corrected(C2_matrix, theta_deg, ks):
    theta_rad = np.radians(theta_deg)
    C11 = np.real(C2_matrix[:, :, 0, 0])
    C22 = np.real(C2_matrix[:, :, 1, 1])
    C12 = C2_matrix[:, :, 0, 1]

    s0 = C11 + C22
    s1 = C11 - C22
    s2 = 2 * np.real(C12)
    s3 = -2 * np.imag(C12)

    a = 0.75
    b = -2 * (s1 - 0.5 * s2)
    c = s1**2 - s2**2 - s3**2 - s0**2
    disc = np.maximum(b**2 - 4 * a * c, 0)
    mv_vol = (-b - np.sqrt(disc)) / (2 * a)
    mv_vol = np.clip(mv_vol, 0, s0)

    S_v = np.stack([
        np.ones_like(s0),
        np.full_like(s0, 0.5),
        np.zeros_like(s0),
        np.zeros_like(s0)
    ], axis=2)
    S_vol = mv_vol[..., np.newaxis] * S_v
    S_total = np.stack([s0, s1, s2, s3], axis=2)
    S_soil = S_total - S_vol
    sigma0_vv = 0.5 * (S_soil[:, :, 0] + S_soil[:, :, 1])

    denom1 = 0.095 * (0.13 + np.sin(theta_rad)**1.5)**1.4
    denom3 = 0.11 * (np.cos(theta_rad)**2.2) * (1 - np.exp(-0.32 * ks**1.8))
    mv = (sigma0_vv / (denom1 * denom3))**(10 / 7)
    mv = np.clip(mv, 0.01, 0.50)
    return mv

wavelength = 0.056
k = 2 * np.pi / wavelength
s = 0.05
ks = k * s

mv_map = compute_mv_vegetation_corrected(C2_matrix, theta_deg, ks)

out_tif = os.path.join(folder, "soil_moisture_map.tif")
driver = gdal.GetDriverByName("GTiff")
out_ds = driver.Create(out_tif, mv_map.shape[1], mv_map.shape[0], 1, gdal.GDT_Float32)
out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
out_ds.SetProjection(ref_ds.GetProjection())
out_ds.GetRasterBand(1).WriteArray(mv_map)
out_ds.GetRasterBand(1).SetNoDataValue(-9999)
out_ds.FlushCache()
out_ds = None

plt.figure(figsize=(10, 6))
plt.imshow(mv_map, cmap='jet_r', vmin=0.01, vmax=0.5)
plt.colorbar(label="Soil Moisture (m³/m³)")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(folder, "soil_moisture_map.png"), dpi=300)
plt.close()

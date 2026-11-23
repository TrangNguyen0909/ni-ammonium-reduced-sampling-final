"""
Monthly rainfall (HadUK-Grid, Met Office, 5 km) for the 281 cohort stations, 1990-2024.

Put this script and station_coords_281.csv in the same folder, then run (from any folder):
    pip install xarray netCDF4 pyproj pandas requests
    python get_haduk_rainfall.py

Output: data/haduk_rainfall_stations_monthly.csv  (StationCode, month, rainfall_mm, raindays_ge10mm, cell_x, cell_y, cell_distance_m)
Send that CSV (small, about 120,000 rows) back for the analysis.

Two ways to obtain the netCDF files:
  A) automatic download from the CEDA archive (ROOT_URL and VERSION_DIR below), or
  B) download the files in a browser and put them in data/haduk_nc/ (the script then skips downloading).
HadUK-Grid is published under the Open Government Licence. If the download returns 401/403, log in to CEDA in the browser and
use option B.
"""
import os, sys, glob
import numpy as np, pandas as pd

YEARS = range(1990, 2025)
# 5 km monthly files of HadUK-Grid v1.3.2.ceda. Open .../5km/<variable>/mon/ in a browser and copy the version folder name
# shown there (e.g. v20250415) into VERSION_DIR. If the .nc files are listed directly in .../mon/, set VERSION_DIR = "".
ROOT_URL = "https://dap.ceda.ac.uk/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.3.2.ceda/5km/{var}/mon/{ver}/"
VERSION_DIR = "latest"
VARS = ["rainfall", "raindays10mm"]          # monthly total (mm) and number of days with >= 10 mm
FNAME = "{var}_hadukgrid_uk_5km_mon_{y}01-{y}12.nc"
HERE = os.path.dirname(os.path.abspath(__file__))
NC_DIR = os.path.join(HERE, "data", "haduk_nc"); OUT = os.path.join(HERE, "data", "haduk_rainfall_stations_monthly.csv")
META = os.path.join("results", "complete_study", "data", "station_metadata.csv")
REF = os.path.join("results", "complete_study", "data", "monthly_reference_record.csv.gz")
IRISH_GRID = "EPSG:29903"      # Irish Grid (TM75). The coordinates in the DAERA file are Irish Grid eastings/northings.
BNG = "EPSG:27700"             # British National Grid, the projection of HadUK-Grid

def download():
    import requests
    os.makedirs(NC_DIR, exist_ok=True)
    for var in VARS:
        for y in YEARS:
            fn = os.path.join(NC_DIR, FNAME.format(var=var, y=y))
            if os.path.exists(fn) and os.path.getsize(fn) > 0: continue
            url = ROOT_URL.format(var=var, ver=VERSION_DIR).replace("//", "/").replace("https:/", "https://") + FNAME.format(var=var, y=y); print("downloading", url)
            r = requests.get(url, timeout=120)
            if r.status_code != 200:
                sys.exit(f"HTTP {r.status_code} for {url}\n-> set VERSION_DIR to the folder name shown in the browser, or download the files manually into {NC_DIR}")
            if not (r.content[:3] == b"CDF" or r.content[:4] == b"\x89HDF"):          # netCDF3 / netCDF4 signature
                sys.exit(f"{url} did not return a netCDF file (probably a CEDA login page).\n-> log in to CEDA in a browser, download the files "
                         f"manually into {NC_DIR}, and run this script again")
            open(fn, "wb").write(r.content)

def main():
    import xarray as xr
    from pyproj import Transformer
    if len(glob.glob(os.path.join(NC_DIR, "*.nc"))) < len(YEARS) * len(VARS): download()
    here = os.path.dirname(os.path.abspath(__file__)); coords = os.path.join(here, "station_coords_281.csv")
    if os.path.exists(coords):                                                   # simplest: coordinates file next to this script
        meta = pd.read_csv(coords).set_index("StationCode")
    else:
        cohort = pd.read_csv(REF, usecols=["StationCode"]).StationCode.unique()
        meta = pd.read_csv(META).set_index("StationCode").loc[cohort]
    x, y = Transformer.from_crs(IRISH_GRID, BNG, always_xy=True).transform(meta.E.values, meta.N.values)
    bad = [f for f in glob.glob(os.path.join(NC_DIR, "*.nc")) if open(f, "rb").read(4)[:3] != b"CDF" and open(f, "rb").read(4) != b"\x89HDF"]
    if bad:
        sys.exit(f"{len(bad)} file(s) in {NC_DIR} are not netCDF (for example {os.path.basename(bad[0])}, {os.path.getsize(bad[0])} bytes; "
                 "a real file is about 1.8 MB). Delete them and download the files manually after logging in to CEDA.")
    tabs = []
    for var in VARS:
        rows = []
        for yy in YEARS:
            ds = xr.open_dataset(os.path.join(NC_DIR, FNAME.format(var=var, y=yy)))
            R = ds[var]; gx = ds["projection_x_coordinate"].values; gy = ds["projection_y_coordinate"].values
            land = np.isfinite(R.isel(time=0).values)                               # sea cells are missing
            jj, ii = np.nonzero(land); cx, cy = gx[ii], gy[jj]
            for code, sx, sy in zip(meta.index, x, y):
                d2 = (cx - sx) ** 2 + (cy - sy) ** 2; k = int(np.argmin(d2))        # nearest land cell
                for t, v in zip(pd.to_datetime(ds["time"].values), R.values[:, jj[k], ii[k]]):
                    rows.append({"StationCode": code, "month": f"{t.year}-{t.month:02d}", var: float(v),
                                 "cell_x": float(cx[k]), "cell_y": float(cy[k]), "cell_distance_m": float(np.sqrt(d2[k]))})
            ds.close(); print(var, yy, "done")
        tabs.append(pd.DataFrame(rows).set_index(["StationCode", "month"]))
    out = tabs[0]
    for t_, v_ in zip(tabs[1:], VARS[1:]): out = out.join(t_[[v_]])
    out = out.reset_index().sort_values(["StationCode", "month"]).rename(columns={"rainfall": "rainfall_mm", "raindays10mm": "raindays_ge10mm"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True); out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out):,} rows, {out.StationCode.nunique()} stations, months {out.month.min()}..{out.month.max()}; "
          f"max station-to-cell distance {out.cell_distance_m.max():.0f} m; missing values: "
          + ", ".join(f"{c} {int(out[c].isna().sum())}" for c in out.columns if c in ("rainfall_mm", "raindays_ge10mm")))

if __name__ == "__main__":
    main()

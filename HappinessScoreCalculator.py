import os
import pandas as pd
import geopandas as gpd

def header(t):
    print("\n" + "=" * 50)
    print(t)
    print("=" * 50)

def sanity(name, df, n_missing=10):
    header(f"{name} sanity")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nHead:")
    print(df.head())
    print("\nMissing values (top):")
    print(df.isna().sum().sort_values(ascending=False).head(n_missing))

def load_parquet(path, name):
    if not os.path.exists(path):
        header(f"{name} not found: {path}")
        return None
    try:
        df = pd.read_parquet(path)
        sanity(name, df)
        return df
    except Exception as e:
        header(f"Error loading {name}")
        print(e)
        return None

def load_gpkg(path, name):
    if not os.path.exists(path):
        header(f"{name} not found: {path}")
        return None
    try:
        gdf = gpd.read_file(path)
        sanity(name, gdf.drop(columns=["geometry"], errors="ignore"))
        return gdf
    except Exception as e:
        header(f"Error loading {name}")
        print(e)
        return None

def to_csv(df, out_name, is_geo=False, keep_geom=False):
    if df is None:
        return
    df = df.copy()
    if is_geo and not keep_geom and "geometry" in df.columns:
        df = df.drop(columns=["geometry"])
    out_path = f"{out_name}.csv"
    df.to_csv(out_path, index=False)
    header(f"Saved {out_name} to {out_path}")

def min_max_norm(s):
    s = pd.to_numeric(s, errors="coerce")
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)

def compute_happiness_from_cadastre(cadastre):
    if cadastre is None:
        header("No cadastre, cannot compute happiness")
        return None

    gdf = cadastre.copy()

    area_col = None
    for c in ["lot_area_sqm", "area_sqm", "land_area_sqm", "area", "land_area"]:
        if c in gdf.columns:
            area_col = c
            break

    if area_col is not None:
        header(f"Using existing area column: {area_col}")
        gdf["lot_area_sqm"] = pd.to_numeric(gdf[area_col], errors="coerce")
    else:
        if "geometry" not in gdf.columns:
            header("No geometry and no area column, cannot compute happiness")
            return None
        header("Computing area from geometry")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:3857")
        gdf["lot_area_sqm"] = gdf.geometry.area

    gdf["lot_area_norm"] = min_max_norm(gdf["lot_area_sqm"])
    gdf["happiness_score"] = gdf["lot_area_norm"]

    out_cols = [c for c in gdf.columns if c != "geometry"]
    out_path = "cadastre_with_happiness.csv"
    gdf[out_cols].to_csv(out_path, index=False)
    header(f"Happiness score saved to {out_path}")
    return gdf

def main():
    transactions = load_parquet("transactions.parquet", "transactions")
    gnaf = load_parquet("gnaf_prop.parquet", "gnaf_prop")
    cadastre = load_gpkg("cadastre.gpkg", "cadastre")
    roads = load_gpkg("roads.gpkg", "roads")

    to_csv(transactions, "transactions")
    to_csv(gnaf, "gnaf_prop")
    to_csv(cadastre, "cadastre", is_geo=True)
    to_csv(roads, "roads", is_geo=True)

    compute_happiness_from_cadastre(cadastre)

    header("Done")

if __name__ == "__main__":
    main()

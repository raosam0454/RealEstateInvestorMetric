import os
import pandas as pd
import geopandas as gpd
from shapely.ops import nearest_points

def header(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)

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
    header(f"Saved {out_name} -> {out_path}")

def min_max_norm(s):
    s = pd.to_numeric(s, errors="coerce")
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)

def find_area_column(df):
    candidates = [c for c in df.columns if "area" in c.lower() or "land" in c.lower()]
    if not candidates:
        return None
    # pick first numeric-ish candidate
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == object:
            return c
    return None

def find_coord_columns(df):
    cols = [c.lower() for c in df.columns]
    mapping = {c.lower(): c for c in df.columns}

    lon_candidates = [c for c in cols if "lon" in c or "x" == c or "long" in c]
    lat_candidates = [c for c in cols if "lat" in c or "y" == c]

    lon = mapping[lon_candidates[0]] if lon_candidates else None
    lat = mapping[lat_candidates[0]] if lat_candidates else None
    return lon, lat

def classify_roads(roads):
    if roads is None or roads.empty:
        return None, None

    roads = roads.copy()
    class_col = None
    for c in roads.columns:
        cl = c.lower()
        if "class" in cl or "fclass" in cl or "type" in cl or "road" in cl or "func" in cl:
            class_col = c
            break

    if class_col is None:
        return None, None

    vals = roads[class_col].astype(str).str.lower().fillna("")

    major_mask = vals.str.contains("motor") | vals.str.contains("trunk") | \
                 vals.str.contains("primary") | vals.str.contains("highway") | \
                 vals.str.contains("arterial")

    path_mask = vals.str.contains("path") | vals.str.contains("foot") | \
                vals.str.contains("cycle") | vals.str.contains("track") | \
                vals.str.contains("walk")

    major = roads[major_mask]
    paths = roads[path_mask]

    if major.empty:
        major = None
    if paths.empty:
        paths = None
    return major, paths

def nearest_distance(source_gdf, target_gdf):
    if target_gdf is None or target_gdf.empty:
        return pd.Series(0.5, index=source_gdf.index)  # neutral if nothing available
    union = target_gdf.unary_union
    dists = []
    for geom in source_gdf.geometry:
        if geom is None or geom.is_empty:
            dists.append(float("nan"))
            continue
        nearest_geom = nearest_points(geom, union)[1]
        dists.append(geom.distance(nearest_geom))
    return pd.Series(dists, index=source_gdf.index)

def compute_happiness(cadastre, roads):
    if cadastre is None or cadastre.empty:
        header("No cadastre, cannot compute happiness")
        return None

    gdf = cadastre.copy()

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")

    area_col = find_area_column(gdf)
    if area_col is not None:
        header(f"Using area-like column: {area_col}")
        gdf["lot_area_sqm"] = pd.to_numeric(gdf[area_col], errors="coerce")
    else:
        header("No area column, computing lot_area_sqm from geometry")
        gdf["lot_area_sqm"] = gdf.geometry.area

    gdf["lot_area_norm"] = min_max_norm(gdf["lot_area_sqm"])

    major, paths = classify_roads(roads.to_crs(gdf.crs)) if roads is not None and not roads.empty else (None, None)

    gdf["centroid"] = gdf.geometry.centroid
    gdf = gdf.set_geometry("centroid")

    features = []
    weights = {}

    features.append("lot_area_norm")
    weights["lot_area_norm"] = 1.0

    if major is not None:
        header("Using distance to major roads")
        d_major = nearest_distance(gdf, major.to_crs(gdf.crs))
        gdf["dist_major_norm"] = min_max_norm(d_major)
        features.append("dist_major_norm")
    if paths is not None:
        header("Using distance to paths")
        d_paths = nearest_distance(gdf, paths.to_crs(gdf.crs))
        # closer is better -> invert
        gdf["dist_path_norm"] = 1 - min_max_norm(d_paths)
        features.append("dist_path_norm")

    if len(features) > 1:
        w = 1.0 / len(features)
        for f in features:
            weights[f] = w
    else:
        weights["lot_area_norm"] = 1.0

    header(f"Using features for happiness: {features}")
    gdf["happiness_score"] = 0.0
    for f in features:
        gdf["happiness_score"] += weights[f] * gdf[f]

    gdf = gdf.set_geometry("geometry")  # restore
    gdf = gdf.drop(columns=["centroid"], errors="ignore")

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

    compute_happiness(cadastre, roads)

    header("Done")

if __name__ == "__main__":
    main()

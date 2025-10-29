import os
import time
import requests
import pandas as pd

def fetch_laureates() -> dict:
    url = "https://api.nobelprize.org/v1/laureate.json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def normalize(data: dict) -> pd.DataFrame:
    rows = []
    for l in data.get("laureates", []):
        lid = l.get("id")
        name = " ".join([l.get("firstname", ""), l.get("surname", "")]).strip()
        gender = l.get("gender")
        born = l.get("born")
        born_country = l.get("bornCountry")
        born_city = l.get("bornCity")
        prizes = l.get("prizes") or []
        if not isinstance(prizes, list):
            prizes = [prizes]
        if not prizes:
            rows.append({
                "id": lid, "name": name, "gender": gender, "bornDate": born,
                "bornCountry": born_country, "bornCity": born_city,
                "year": None, "category": None, "share": None, "motivation": None,
                "aff_org": None, "aff_city": None, "aff_country": None
            })
            continue
        for p in prizes:
            year = p.get("year")
            category = p.get("category")
            share = p.get("share")
            motivation = p.get("motivation")
            raw_affs = p.get("affiliations") or []
            # 扁平化 affiliations，过滤非字典项
            flattened_affs = []
            for item in raw_affs:
                if isinstance(item, dict):
                    flattened_affs.append(item)
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict):
                            flattened_affs.append(sub)
            if not flattened_affs:
                rows.append({
                    "id": lid, "name": name, "gender": gender, "bornDate": born,
                    "bornCountry": born_country, "bornCity": born_city,
                    "year": year, "category": category, "share": share, "motivation": motivation,
                    "aff_org": None, "aff_city": None, "aff_country": None
                })
            else:
                for a in flattened_affs:
                    rows.append({
                        "id": lid, "name": name, "gender": gender, "bornDate": born,
                        "bornCountry": born_country, "bornCity": born_city,
                        "year": year, "category": category, "share": share, "motivation": motivation,
                        "aff_org": a.get("name"), "aff_city": a.get("city"), "aff_country": a.get("country")
                    })
    return pd.DataFrame(rows)

def main():
    os.makedirs("artifacts/nobel", exist_ok=True)
    data = fetch_laureates()
    df = normalize(data)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    ts = int(time.time())
    csv_path = f"artifacts/nobel/laureates_prizes.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows -> {csv_path}")

if __name__ == "__main__":
    main()
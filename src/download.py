"""Download public road safety and population data into data/raw/.

Sources
- Department for Transport, Road Safety Data (STATS19), "last 5 years" collision and casualty files
  https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data
  Open Government Licence v3.0
- DfT Road Safety Open Data guide 2025 (code lookups)
  https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
- ONS mid-year population estimates by local authority, via the Nomis API (dataset NM_2002_1)
  https://www.nomisweb.co.uk/  Open Government Licence v3.0
"""
import pathlib, requests

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
DFT = "https://data.dft.gov.uk/road-accidents-safety-data/"
NOMIS = ("https://www.nomisweb.co.uk/api/v01/dataset/NM_2002_1.data.csv?geography=TYPE424"
         "&date=latestMINUS4,latestMINUS3,latestMINUS2,latestMINUS1,latest&gender=0&c_age=200&measures=20100"
         "&select=geography_code,geography_name,date_name,obs_value")
FILES = {
    "collisions.csv": DFT + "dft-road-casualty-statistics-collision-last-5-years.csv",
    "casualties.csv": DFT + "dft-road-casualty-statistics-casualty-last-5-years.csv",
    "data_guide.xlsx": ("https://assets.publishing.service.gov.uk/media/6ab2a71d997a4b2950cced58/"
                        "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx"),
    "population_la.csv": NOMIS,
}
HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio project; UK open data)"}

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        print("Downloading", url[:110])
        with requests.get(url, headers=HEADERS, timeout=600, stream=True) as r:
            r.raise_for_status()
            with open(RAW / name, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    print("Done:", sorted((p.name, p.stat().st_size) for p in RAW.iterdir()))

if __name__ == "__main__":
    main()

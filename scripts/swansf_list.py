"""Probe the SWAN-SF Dataverse listing (file ids + sizes)."""

import requests

URL = ("https://dataverse.harvard.edu/api/datasets/:persistentId/"
       "?persistentId=doi:10.7910/DVN/EBCFKM")

if __name__ == "__main__":
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    files = r.json()["data"]["latestVersion"]["files"]
    print(f"{len(files)} files:")
    for f in sorted(files, key=lambda x: x["dataFile"]["filename"]):
        df = f["dataFile"]
        print(f"  id={df['id']:>9}  {df['filename']:<48} {df['filesize'] / 1e6:9.1f} MB")

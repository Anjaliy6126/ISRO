import requests

MAP_KEY = "03d850d6b6b9fef427282097766c994c"

url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/world/1"

response = requests.get(url)

print("Status Code:", response.status_code)

if response.status_code == 200:
    with open("fire_data.csv", "wb") as f:
        f.write(response.content)
    print("Fire data downloaded successfully!")
else:
    print(response.text)
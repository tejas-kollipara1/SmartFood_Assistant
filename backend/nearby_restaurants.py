import os
import math
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

# =========================================
# 1. LOAD ENV & CONFIGURE API CLIENTS
# =========================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

if not PLACES_API_KEY:
    raise RuntimeError("GOOGLE_PLACES_API_KEY is not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================================
# 2. HELPER: HAVERSINE DISTANCE (KM)
# =========================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# =========================================
# 3. HELPER: PARSE BUDGET STRING
# =========================================
def parse_budget(budget_str):
    if not budget_str:
        return (None, None)
    s = budget_str.lower().replace("rs", "").replace("rupees", "").strip()
    if "under" in s:
        for token in s.split():
            if token.isdigit():
                return (0, int(token))
    if "-" in s:
        parts = s.split("-")
        if len(parts) == 2:
            try:
                return (int(parts[0]), int(parts[1]))
            except ValueError:
                pass
    for token in s.split():
        if token.isdigit():
            return (0, int(token))
    return (None, None)

# =========================================
# 4. GOOGLE PLACES NEARBY SEARCH
# =========================================
def get_nearby_restaurants_from_google(lat, lon, craving, radius_m=8000, max_results=20):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius": radius_m,
        "keyword": craving,
        "type": "restaurant",
        "key": PLACES_API_KEY,
    }
    restaurants = []

    while len(restaurants) < max_results:
        response = requests.get(url, params=params)
        data = response.json()

        status = data.get("status")
        if status == "ZERO_RESULTS":
            return []
        if status != "OK":
            raise RuntimeError(f"Places API error: {status} - {data.get('error_message')}")

        for place in data.get("results", []):
            try:
                loc = place["geometry"]["location"]
                p_lat, p_lon = loc["lat"], loc["lng"]
                place_id = place.get("place_id")

                maps_url = f"https://www.google.com/maps/search/?api=1&query={p_lat},{p_lon}"
                if place_id:
                    maps_url += f"&query_place_id={place_id}"

                restaurants.append({
                    "name": place.get("name", "Unknown"),
                    "rating": place.get("rating"),
                    "user_ratings_total": place.get("user_ratings_total", 0),
                    "price_level": place.get("price_level"),
                    "address": place.get("vicinity", ""),
                    "latitude": p_lat,
                    "longitude": p_lon,
                    "distance_km": round(haversine_km(lat, lon, p_lat, p_lon), 1),
                    "maps_url": maps_url,
                })

                if len(restaurants) >= max_results:
                    return restaurants
            except KeyError:
                continue

        next_token = data.get("next_page_token")
        if not next_token:
            break
        params["pagetoken"] = next_token
        # Google requires a short delay before using the token
        import time
        time.sleep(2)

    return restaurants

# =========================================
# 5. OPENAI — RECOMMENDATIONS (FIXED FOR OPENAI 1.X)
# =========================================
def get_restaurant_recommendations(location, craving, budget_str):
    lat, lon = location
    min_budget, max_budget = parse_budget(budget_str)

    places = get_nearby_restaurants_from_google(lat, lon, craving)
    if not places:
        return "No restaurants found for your craving. Try something else!"

    places_json = json.dumps(places, indent=2, ensure_ascii=False)

    prompt = f"""You are a friendly restaurant recommendation assistant in India.

User wants: {craving}
Budget: {budget_str} INR
Location: Hyderabad area

Here are real restaurants nearby:
{places_json}

Please recommend the TOP 5 in beautiful Markdown format like this:

**1. Restaurant Name**
   • Cuisine: South Indian / Chinese / etc.
   • Approx cost: ₹300–500 (based on price_level)
   • Distance: 2.1 km
   • Why: High rating, matches your craving perfectly
   • Google Maps → {places[0]['maps_url']}

At the end say:
My top recommendation is #1 because...

Keep it short, fun and in Indian style."""

    # THIS IS THE ONLY CHANGE NEEDED FOR OPENAI 1.X+
    response = client.chat.completions.create(
        model="gpt-4o-mini",           # or gpt-3.5-turbo if you want cheaper
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()

# =========================================
# 6. MAIN
# =========================================
def main():
    print("SMART FOOD FINDER — Restaurant Recommendation AI\n")
    location = (17.3850, 78.4867)  # Hyderabad
    print(f"Location set to Hyderabad\n")

    craving = input("What are you craving? (biryani, dosa, pizza etc): ").strip()
    budget = input("Budget? (under 300, 200-600, etc): ").strip()

    print("\nSearching the best spots for you...\n")
    try:
        result = get_restaurant_recommendations(location, craving, budget)
        print(result)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
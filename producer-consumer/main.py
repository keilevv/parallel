import threading
import urllib.request
import urllib.error
import urllib.parse
import json
import time
import csv
from collections import deque
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(threadName)s] %(message)s')

class ThreadSafeQueue:
    def __init__(self, max_size):
        self.queue = deque()
        self.max_size = max_size
        self.condition = threading.Condition()
        self.is_shutdown = False

    def put(self, item):
        with self.condition:
            while len(self.queue) >= self.max_size:
                logging.debug("Queue full, waiting to put...")
                self.condition.wait()
            self.queue.append(item)
            logging.debug("Data added, notifying consumers...")
            self.condition.notify()

    def get(self):
        with self.condition:
            while len(self.queue) == 0 and not self.is_shutdown:
                logging.debug("Queue empty, waiting to get...")
                self.condition.wait()
            if self.is_shutdown and len(self.queue) == 0:
                return None
            item = self.queue.popleft()
            logging.debug("Data removed, notifying producers...")
            self.condition.notify()
            return item

    def shutdown(self):
        with self.condition:
            self.is_shutdown = True
            self.condition.notify_all()

def expand_zip_codes(pattern):
    pattern = pattern.lower()
    if 'x' not in pattern:
        return [pattern]
    
    prefix = pattern.split('x')[0]
    num_x = pattern.count('x')
    
    start = int(prefix + '0' * num_x)
    end = int(prefix + '9' * num_x)
    return [str(v).zfill(len(pattern)) for v in range(start, end + 1)]

def fetch_json(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

# Global APIs Caches and Locks
weather_cache = {}
air_cache = {}
weather_lock = threading.Lock()
air_lock = threading.Lock()
nominatim_lock = threading.Lock()

def get_coordinates(zip_code):
    # Use Nominatim (OpenStreetMap) API to fetch latitude and longitude dynamically
    # Nominatim requires a descriptive User-Agent and limits to 1 request per second
    url = f"https://nominatim.openstreetmap.org/search?postalcode={urllib.parse.quote(zip_code)}&country=Colombia&format=json"
    
    try:
        with nominatim_lock:
            req = urllib.request.Request(url, headers={'User-Agent': 'producer-consumer-test-script/1.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                
            # Respect OpenStreetMap's acceptable use policy (1 req/sec max strictly enforced across all threads)
            time.sleep(1)
            
            if data and len(data) > 0:
                res = data[0]
                lat = float(res.get('lat', 0))
                lon = float(res.get('lon', 0))
                if lat and lon:
                    return lat, lon
                
    except Exception as e:
        logging.error(f"Geocoding API error for {zip_code}: {e}")
        
    # If the geocoding fails (e.g., zip code not found in OpenStreetMap DB), fallback to department region coordinates
    mappings = {
        "05": (6.2518, -75.5636), "08": (10.9685, -74.7813), "11": (4.6097, -74.0817), 
        "13": (10.3997, -75.4794), "15": (5.5353, -73.3678), "17": (5.0689, -75.5174), 
        "18": (1.6144, -75.6062), "19": (2.4382, -76.6132), "20": (10.4631, -73.2532), 
        "23": (8.748, -75.8814), "25": (4.5981, -74.0758), "27": (5.6947, -76.6611), 
        "41": (2.9273, -75.2819), "44": (11.5444, -72.9072), "47": (11.2408, -74.199), 
        "50": (4.142, -73.6266), "52": (1.2136, -77.2811), "54": (7.8939, -72.5078), 
        "63": (4.5339, -75.6811), "66": (4.8133, -75.6961), "68": (7.1254, -73.1198), 
        "70": (9.3047, -75.3978), "73": (4.4389, -75.2322), "76": (3.4372, -76.5225), 
        "81": (7.0847, -70.7591), "85": (5.3378, -72.3959), "86": (1.1492, -76.6464), 
        "88": (12.5847, -81.7006), "91": (-4.2153, -69.9406), "94": (3.8653, -67.9239), 
        "95": (2.5732, -72.6459), "97": (1.2532, -70.2346), "99": (6.1822, -67.47)
    }
    prefix2 = zip_code[:2]
    if prefix2 in mappings:
        return mappings[prefix2]
        
    return 4.6097, -74.0817

def fetch_weather_history(lat, lon):
    with weather_lock:
        if (lat, lon) in weather_cache:
            logging.info(f"CACHE HIT: Returning cached Weather for ({lat}, {lon})")
            return weather_cache[(lat, lon)]
            
    logging.info(f"API REQUEST: Fetching Weather from Open-Meteo for ({lat}, {lon})")
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2024-01-01&end_date=2024-01-01&hourly=temperature_2m,precipitation,wind_speed_10m"
    res = fetch_json(url)
    
    with weather_lock:
        weather_cache[(lat, lon)] = res
    return res

def fetch_air_quality_history(lat, lon):
    with air_lock:
        if (lat, lon) in air_cache:
            logging.info(f"CACHE HIT: Returning cached Air Quality for ({lat}, {lon})")
            return air_cache[(lat, lon)]
            
    logging.info(f"API REQUEST: Fetching Air Quality from Open-Meteo for ({lat}, {lon})")
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&start_date=2024-01-01&end_date=2024-01-01&hourly=pm10,pm2_5,ozone"
    res = fetch_json(url)
    
    with air_lock:
        air_cache[(lat, lon)] = res
    return res

def process_zip_pattern(pattern):
    zips = expand_zip_codes(pattern)
    # To avoid overloading the API in tests, we limit to first 3 if there are too many, or keep full
    return zips

class Producer(threading.Thread):
    def __init__(self, queue, zip_codes):
        super().__init__(name="Producer")
        self.queue = queue
        self.zip_codes = zip_codes

    def run(self):
        for zc in self.zip_codes:
            logging.info(f"Resolving coordinates for {zc}")
            lat, lon = get_coordinates(zc)
            if lat is None or lon is None:
                logging.warning(f"Could not find coordinates for {zc}, skipping")
                continue
            
            weather = fetch_weather_history(lat, lon)
            air = fetch_air_quality_history(lat, lon)
            
            payload = {
                'zip': zc,
                'lat': lat,
                'lon': lon,
                'weather': weather,
                'air': air
            }
            logging.info(f"Putting data for {zc} into queue")
            self.queue.put(payload)
        
        logging.info("Producer finished fetching")

class Consumer(threading.Thread):
    def __init__(self, queue, writer, csv_lock, name="Consumer"):
        super().__init__(name=name)
        self.queue = queue
        self.writer = writer
        self.csv_lock = csv_lock

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                logging.info(f"{self.name} received shutdown signal")
                break
            
            logging.info(f"{self.name} processing data for {item['zip']}")
            self.aggregate_and_save(item)

    def aggregate_and_save(self, item):
        weather = item.get('weather')
        air = item.get('air')
        if not weather or not air or 'hourly' not in weather or 'hourly' not in air:
            logging.error(f"Incomplete data for {item['zip']}")
            return

        w_hourly = weather['hourly']
        a_hourly = air['hourly']

        times = w_hourly.get('time', [])
        
        # We lock the CSV write so rows don't get mixed up if multiple consumers process concurrently
        with self.csv_lock:
            for i in range(len(times)):
                t = times[i]
                temp = w_hourly['temperature_2m'][i] if i < len(w_hourly['temperature_2m']) else None
                precip = w_hourly['precipitation'][i] if i < len(w_hourly['precipitation']) else None
                wind = w_hourly['wind_speed_10m'][i] if i < len(w_hourly['wind_speed_10m']) else None
                
                pm10 = a_hourly.get('pm10', [])[i] if i < len(a_hourly.get('pm10', [])) else None
                pm25 = a_hourly.get('pm2_5', [])[i] if i < len(a_hourly.get('pm2_5', [])) else None
                ozone = a_hourly.get('ozone', [])[i] if i < len(a_hourly.get('ozone', [])) else None
                
                self.writer.writerow([
                    t, item['zip'], item['lat'], item['lon'],
                    temp, precip, wind, pm10, pm25, ozone
                ])

def run_serial(zip_codes, out_file):
    logging.info("--- Starting Serial Execution ---")
    start_time = time.time()
    
    with open(out_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Region (ZIP)', 'Latitude', 'Longitude', 
                         'Temperature (C)', 'Precipitation (mm)', 'Wind Speed (km/h)', 
                         'PM10', 'PM2.5', 'Ozone'])
        
        for zc in zip_codes:
            logging.info(f"Resolving coordinates for {zc}")
            lat, lon = get_coordinates(zc)
            if lat is None or lon is None:
                logging.warning(f"Could not find coordinates for {zc}, skipping")
                continue
            
            weather = fetch_weather_history(lat, lon)
            air = fetch_air_quality_history(lat, lon)
            
            logging.info(f"Saving data for {zc}")
            item = {'zip': zc, 'lat': lat, 'lon': lon, 'weather': weather, 'air': air}
            
            # Aggregate and save (inline logic)
            if weather and air and 'hourly' in weather and 'hourly' in air:
                w_hourly = weather['hourly']
                a_hourly = air['hourly']
                times = w_hourly.get('time', [])
                for i in range(len(times)):
                    writer.writerow([
                        times[i], zc, lat, lon,
                        w_hourly['temperature_2m'][i] if i < len(w_hourly['temperature_2m']) else None,
                        w_hourly['precipitation'][i] if i < len(w_hourly['precipitation']) else None,
                        w_hourly['wind_speed_10m'][i] if i < len(w_hourly['wind_speed_10m']) else None,
                        a_hourly.get('pm10', [])[i] if i < len(a_hourly.get('pm10', [])) else None,
                        a_hourly.get('pm2_5', [])[i] if i < len(a_hourly.get('pm2_5', [])) else None,
                        a_hourly.get('ozone', [])[i] if i < len(a_hourly.get('ozone', [])) else None
                    ])
                    
    end_time = time.time()
    elapsed = end_time - start_time
    logging.info(f"--- Serial Execution Finished in {elapsed:.2f} seconds ---")
    return elapsed

def run_threaded(zip_codes, out_file):
    logging.info("--- Starting Threaded Execution ---")
    start_time = time.time()
    
    import math
    import os
    
    total_zips = len(zip_codes)
    cpu_cores = os.cpu_count() or 4
    
    # Intelligently assign Producers based on the workload size and system cores
    # We allocate 1 producer per ~10 zip codes, capping it at a max of (cpu_cores * 4) since these are I/O bound tasks
    NUM_PRODUCERS = max(1, min(total_zips // 10 + 1, cpu_cores * 4))
    
    # Intelligently assign Consumers based on the Producers (ratio of 4:1)
    NUM_CONSUMERS = max(1, NUM_PRODUCERS // 4)
    
    logging.info(f"System detected {cpu_cores} cores. Dynamically provisioning {NUM_PRODUCERS} Producers and {NUM_CONSUMERS} Consumers for processing {total_zips} locations.")
    
    queue = ThreadSafeQueue(max_size=NUM_PRODUCERS * 10) # Dynamic capacity scaling for parallel pool
    
    chunk_size = math.ceil(total_zips / NUM_PRODUCERS)
    if chunk_size == 0: chunk_size = 1
    chunks = [zip_codes[i:i + chunk_size] for i in range(0, total_zips, chunk_size)]
    
    producers = []
    for idx, chunk in enumerate(chunks):
        p = Producer(queue, chunk)
        p.name = f"Producer-{idx+1}"
        p.start()
        producers.append(p)
        
    csv_lock = threading.Lock()
    with open(out_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Region (ZIP)', 'Latitude', 'Longitude', 
                         'Temperature (C)', 'Precipitation (mm)', 'Wind Speed (km/h)', 
                         'PM10', 'PM2.5', 'Ozone'])
        
        consumers = []
        for i in range(NUM_CONSUMERS):
            c = Consumer(queue, writer, csv_lock, name=f"Consumer-{i+1}")
            c.start()
            consumers.append(c)
        
        for p in producers:
            p.join()
            
        queue.shutdown()  # Sends None or wakes up consumer
        
        for c in consumers:
            c.join()
            
    end_time = time.time()
    elapsed = end_time - start_time
    logging.info(f"--- Threaded Execution Finished in {elapsed:.2f} seconds ---")
    return elapsed

if __name__ == "__main__":
    import re
    
    while True:
        pattern = input("Enter the zip code pattern (e.g., 70xxxx for Sucre, 13xxxx for Bolivar): ").strip().lower()
        if len(pattern) == 6 and re.match(r'^[0-9x]+$', pattern):
            break
        print("Invalid input. Please enter exactly a 6-character string containing only numbers and 'x'.")
    
    patterns = [pattern]
    
    all_zips = []
    for p in patterns:
        all_zips.extend(expand_zip_codes(p))
    
    # Apply no limits as requested by user to map EVERY zip code
    print(f"Total zip codes to check: {len(all_zips)}")
    test_zips = all_zips
    
    logging.info(f"Targets starting with: {test_zips[:5]}... ({len(test_zips)} total)")
    
    serial_time = run_serial(test_zips, "weather_data_serial.csv")
    threaded_time = run_threaded(test_zips, "weather_data_threaded.csv")
    
    print("\n" + "="*40)
    print("PERFORMANCE REPORT")
    print("="*40)
    print(f"Serial Execution Time:   {serial_time:.2f} s")
    print(f"Threaded Execution Time: {threaded_time:.2f} s")
    
    if threaded_time < serial_time:
        print(f"Speedup: {serial_time/threaded_time:.2f}x")
    else:
        print("No speedup observed. Network I/O or GIL may have dominated.")
        
    print("="*40)
    print("""
========================================
DISCUSSION:
========================================
Think of our weather pipeline like a team project:
- The Producers are the "Data Gatherers" (fetching weather/air JSONs from Open-Meteo).
- The Consumers are the "Writers" (copying that data into the final CSV file).
- The Queue is the "Inbox Folder" (it can only hold a few JSONs at once so the Writers don't get overwhelmed).

Why do we need Condition Variables?
If the Inbox is full, the Gatherers stop fetching and go to sleep. If the Inbox is empty, the Writers go to sleep because there's nothing to write. Condition variables are the "alarm clock" they ring to wake each other up!

1. Deadlocks: 
Imagine if the Inbox is full, all Gatherers fall asleep waiting for a Writer to process the JSONs. But all Writers are asleep too! If nobody rings the alarm to wake the others, everyone sleeps forever. That's a deadlock!

2. Missed Signals:
Imagine a Gatherer rings the alarm, but a Writer was distracted putting on their headphones. They miss the signal entirely! To fix this, Writers always double-check the Inbox before going back to sleep (`while len(queue) == 0: wait()`).

3. Race Conditions:
Imagine two Writers try to grab the exact same JSON dataset at the exact same split-second to write it to the CSV. They crash and corrupt the file! Our `with condition:` lock acts like a traffic cop, making sure only one Writer touches the Inbox at a time.

Conclusion:
Because fetching from Open-Meteo over the internet takes forever, having many Gatherers and Writers working together (Threading) makes the whole pipeline much faster than having just one script doing everything top-to-bottom.
""")

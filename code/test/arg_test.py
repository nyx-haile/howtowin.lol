import requests
import time

class redis():
    def __init__(self):
        self.data = {}
        self.expiry = {}
    def get(self, key, default=0):
        if self.expiry.get(key) is not None:
            if self.expiry[key] < time.time():
                del self.data[key]
                del self.expiry[key]
                return default
        return self.data.get(key, default)
    def set(self, key, value):
        self.data[key] = value
    def incr(self, key):
        self.data[key] = self.data.get(key, 0) + 1
    def expire(self, key, value):
        self.expiry[key] = value

def ratelimit(func, *args):
     #define two lists. sec_rq stores the number of requests in the last second (max 20)
     #min_rq stores the number of requests in the last 2 minutes (max 100)
     now = int(time.time())
     ts = now  % 100
     sec_rq = redis_client.get(f"sec_rq{ts}")
     min_rq = redis_client.get(f"min_rq") % 100
     time.sleep(max(redis_client.get(f"min_rq{min_rq}") + 120 - now, 0))
     if sec_rq <= 20:
        redis_client.incr(f"sec_rq{ts}")
        redis_client.expire(f"sec_rq{ts}", now + 1)
        redis_client.incr("min_rq")
        redis_client.set(f"min_rq{min_rq}", now)
        return func(*args)
     else:
         time.sleep(1)
         return ratelimit(func, *args)

redis_client = redis()

def request(url, headers):
    return ratelimit(requests.get, url, headers)

jobs = {}
def tfunc(a):
    print(a)
    jobs[a] = "done"
#print (redis_client.data)


#for i in range(200):
    #ratelimit(tfunc, i)

#for i in range(200):
    #assert jobs.get(i) == "done"

#print (redis_client.data)
print(request("https://www.google.com", {}))

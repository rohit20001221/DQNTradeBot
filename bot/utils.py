import redis, time

def wait():
    r = redis.Redis(host="db")

    while r.get("start") != "START":
        time.sleep(0.5)

        continue
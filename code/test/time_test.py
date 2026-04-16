import time
from datetime import datetime


def main():
    reps = 100000
    for _ in range(reps):
        int(time.time() % 120)

    t1 = time.thread_time()
    p1 = time.process_time()

    for _ in range(reps):
        c = datetime.now()
        int(c.strftime('%S'))

    t2 = time.thread_time() - t1
    p2 = time.process_time() - p1

    for _ in range(reps):
        int(datetime.now().timestamp()) % 60

    t3 = time.thread_time() - t1 - t2
    p3 = time.process_time() - p1 - p2

    print(f"Method 1:\nThread Time {t1}, Process Time {p1}")
    print(f"Method 2:\nThread Time {t2}, Process Time {p2}")
    print(f"Method 3:\nThread Time {t3}, Process Time {p3}")


if __name__ == "__main__":
    main()

import max30102
import time

m = max30102.MAX30102()

def read_sensor():
        while True:
            start = time.time()
            red, ir = m.read_sequential(100)
            print("Time colected:", time.time() - start)
            print(f'red: {red}')

if __name__ == "__main__":
    read_sensor()


#sending 2 int values and recieving results from arduino
import serial
import struct
ser=serial.Serial('/dev/ttyUSB0',115200)
ser.flush()
a="30"
b='-23'

while True:
    ser.write(str.encode(a+","+b))
    rS=ser.readline().decode().rstrip()
    print(rS)
    
    
    
    